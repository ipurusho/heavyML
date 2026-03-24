#!/usr/bin/env python3
"""Training loop for the Siamese MLP band similarity model.

Usage:
    python -m model.train                     # from project root
    python model/train.py                     # also works

Steps:
    1. Load and preprocess feature matrix from data/processed/ma_ab_features.csv
    2. Split similarity pairs into train/val/test (or load pre-split CSVs)
    3. Train Siamese MLP with InfoNCE loss, Adam + cosine LR decay
    4. Early stopping on validation loss
    5. Evaluate on test set: Recall@10, genre purity, MRR
    6. Save best checkpoint + all band embeddings
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `model.*` imports work
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model.tower import BandEncoder
from model.loss import InfoNCELoss
from model.dataset import SimilarBandDataset, preprocess_features
from model.evaluate import (
    evaluate_model,
    predict_top_k_from_embeddings,
    load_true_similar,
    load_lastfm_similar,
    load_band_genres,
    print_comparison_table,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_DIR / "model" / "checkpoints"

FEATURES_CSV = PROCESSED_DIR / "ma_ab_features.csv"
PAIRS_CSV = PROCESSED_DIR / "ma_similar_artists.csv"
METAL_BANDS_CSV = RAW_DIR / "metal_bands.csv"
LASTFM_CSV = PROCESSED_DIR / "lastfm_similar_artists.csv"

# Pre-split CSVs (if they exist, use them; otherwise split on-the-fly)
TRAIN_PAIRS_CSV = PROCESSED_DIR / "train_pairs.csv"
VAL_PAIRS_CSV = PROCESSED_DIR / "val_pairs.csv"
TEST_PAIRS_CSV = PROCESSED_DIR / "test_pairs.csv"

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
EMBEDDINGS_CSV = PROCESSED_DIR / "siamese_embeddings.csv"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_DIM = 64
EMBED_DIM = 32
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 10          # early stopping patience
TEMPERATURE = 0.07
NOISE_STD = 0.01       # Gaussian noise regularisation
MIN_SCORE = 0          # minimum similarity score to include pairs


# ---------------------------------------------------------------------------
# Train / Val / Test split
# ---------------------------------------------------------------------------

def split_pairs(
    pairs_csv: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    """Split similarity pairs into train/val/test CSVs.

    Splits by *anchor band* so that a band's pairs don't leak across splits.
    Saves to data/processed/{train,val,test}_pairs.csv and returns the paths.
    """
    df = pd.read_csv(pairs_csv)

    rng = np.random.RandomState(seed)
    anchor_ids = df["band_id"].unique()
    rng.shuffle(anchor_ids)

    n = len(anchor_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(anchor_ids[:n_train])
    val_ids = set(anchor_ids[n_train : n_train + n_val])
    test_ids = set(anchor_ids[n_train + n_val :])

    train_df = df[df["band_id"].isin(train_ids)]
    val_df = df[df["band_id"].isin(val_ids)]
    test_df = df[df["band_id"].isin(test_ids)]

    train_df.to_csv(TRAIN_PAIRS_CSV, index=False)
    val_df.to_csv(VAL_PAIRS_CSV, index=False)
    test_df.to_csv(TEST_PAIRS_CSV, index=False)

    print(f"  Split {len(df):,} pairs across {n:,} anchor bands:")
    print(f"    Train: {len(train_df):,} pairs ({len(train_ids):,} anchors)")
    print(f"    Val:   {len(val_df):,} pairs ({len(val_ids):,} anchors)")
    print(f"    Test:  {len(test_df):,} pairs ({len(test_ids):,} anchors)")

    return TRAIN_PAIRS_CSV, VAL_PAIRS_CSV, TEST_PAIRS_CSV


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: BandEncoder,
    loss_fn: InfoNCELoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for anchor_feat, pos_feat in loader:
        anchor_feat = anchor_feat.to(device)
        pos_feat = pos_feat.to(device)

        # Forward: encode both through shared tower
        anchor_emb = model(anchor_feat)
        pos_emb = model(pos_feat)

        loss = loss_fn(anchor_emb, pos_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: BandEncoder,
    loss_fn: InfoNCELoss,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Validate. Returns mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for anchor_feat, pos_feat in loader:
        anchor_feat = anchor_feat.to(device)
        pos_feat = pos_feat.to(device)

        anchor_emb = model(anchor_feat)
        pos_emb = model(pos_feat)

        loss = loss_fn(anchor_emb, pos_emb)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def compute_all_embeddings(
    model: BandEncoder,
    feature_matrix: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Compute embeddings for all bands."""
    model.eval()
    features_t = torch.from_numpy(feature_matrix).float()
    embeddings = []

    for start in range(0, len(features_t), batch_size):
        batch = features_t[start : start + batch_size].to(device)
        emb = model(batch).cpu().numpy()
        embeddings.append(emb)

    return np.vstack(embeddings)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("heavyML — Siamese MLP Training")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ------------------------------------------------------------------
    # 1. Load and preprocess features
    # ------------------------------------------------------------------
    print("\n--- Loading features ---")
    if not FEATURES_CSV.exists():
        print(f"ERROR: Feature file not found: {FEATURES_CSV}")
        print("Run pipeline step 02 first.")
        sys.exit(1)

    band_ids, feature_matrix, feature_names = preprocess_features(FEATURES_CSV, METAL_BANDS_CSV)
    n_bands, input_dim = feature_matrix.shape
    print(f"  Bands: {n_bands:,}")
    print(f"  Feature dims: {input_dim} ({', '.join(feature_names)})")

    # ------------------------------------------------------------------
    # 2. Prepare train/val/test pair splits
    # ------------------------------------------------------------------
    print("\n--- Preparing pair splits ---")
    if not PAIRS_CSV.exists():
        print(f"ERROR: Similarity pairs not found: {PAIRS_CSV}")
        print("Run pipeline step 03 first.")
        sys.exit(1)

    # Check pair count
    total_pairs = len(pd.read_csv(PAIRS_CSV))
    if total_pairs < 100:
        warnings.warn(
            f"Only {total_pairs} similarity pairs found. This is very few "
            "for contrastive learning. Results will be unreliable. "
            "Consider waiting for the scraper to collect more pairs.",
            stacklevel=2,
        )
    elif total_pairs < 1000:
        print(f"  WARNING: Only {total_pairs:,} pairs. Model quality may be limited.")

    if TRAIN_PAIRS_CSV.exists() and VAL_PAIRS_CSV.exists() and TEST_PAIRS_CSV.exists():
        print("  Using existing train/val/test splits.")
        train_path, val_path, test_path = TRAIN_PAIRS_CSV, VAL_PAIRS_CSV, TEST_PAIRS_CSV
    else:
        print("  No pre-existing splits found. Splitting on-the-fly ...")
        train_path, val_path, test_path = split_pairs(PAIRS_CSV)

    # ------------------------------------------------------------------
    # 3. Create datasets and dataloaders
    # ------------------------------------------------------------------
    print("\n--- Creating datasets ---")
    train_ds = SimilarBandDataset(
        feature_matrix, band_ids, train_path,
        noise_std=NOISE_STD, min_score=MIN_SCORE,
    )
    val_ds = SimilarBandDataset(
        feature_matrix, band_ids, val_path,
        noise_std=0.0, min_score=MIN_SCORE,
    )
    test_ds = SimilarBandDataset(
        feature_matrix, band_ids, test_path,
        noise_std=0.0, min_score=MIN_SCORE,
    )
    val_ds.training_mode = False
    test_ds.training_mode = False

    print(f"  Train pairs: {len(train_ds):,}")
    print(f"  Val pairs:   {len(val_ds):,}")
    print(f"  Test pairs:  {len(test_ds):,}")
    print(f"  Input dim:   {train_ds.input_dim}")

    if len(train_ds) == 0:
        print("\nERROR: No training pairs available after filtering.")
        print("Check that similarity pair band IDs overlap with feature matrix band IDs.")
        sys.exit(1)

    # Adjust batch size if dataset is small
    effective_batch_size = min(BATCH_SIZE, len(train_ds))
    if effective_batch_size < BATCH_SIZE:
        print(f"  NOTE: Batch size reduced from {BATCH_SIZE} to {effective_batch_size} "
              f"(dataset has only {len(train_ds)} pairs)")

    # InfoNCE needs batch_size >= 2 for negatives
    if effective_batch_size < 2:
        print("\nERROR: Need at least 2 pairs per batch for in-batch negatives.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=effective_batch_size, shuffle=True,
        drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=effective_batch_size, shuffle=False,
        drop_last=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=effective_batch_size, shuffle=False,
        drop_last=False, num_workers=0,
    )

    # ------------------------------------------------------------------
    # 4. Initialise model, loss, optimizer, scheduler
    # ------------------------------------------------------------------
    print("\n--- Initialising model ---")
    # Audio dims = 7 numeric + 12 key one-hot + 1 scale = 20
    N_AUDIO_DIMS = 20

    model = BandEncoder(
        input_dim=train_ds.input_dim,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        dropout=DROPOUT,
        n_audio_dims=N_AUDIO_DIMS,
    ).to(device)

    n_genre_dims = train_ds.input_dim - N_AUDIO_DIMS
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: two-branch ({N_AUDIO_DIMS} audio + {n_genre_dims} genre) -> {HIDDEN_DIM} -> {EMBED_DIM}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Dropout: {DROPOUT}")

    loss_fn = InfoNCELoss(temperature=TEMPERATURE, learnable_temperature=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )

    print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"  Scheduler: CosineAnnealingLR (T_max={MAX_EPOCHS})")
    print(f"  Loss: InfoNCE (tau={TEMPERATURE})")
    print(f"  Batch size: {effective_batch_size}")
    print(f"  Max epochs: {MAX_EPOCHS} (patience={PATIENCE})")

    # ------------------------------------------------------------------
    # 5. Training loop with early stopping
    # ------------------------------------------------------------------
    print("\n--- Training ---")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, loss_fn, train_loader, optimizer, device)
        val_loss = validate(model, loss_fn, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        elapsed = time.time() - epoch_start
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
        })

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "input_dim": train_ds.input_dim,
                "hidden_dim": HIDDEN_DIM,
                "embed_dim": EMBED_DIM,
                "dropout": DROPOUT,
                "n_audio_dims": N_AUDIO_DIMS,
                "feature_names": feature_names,
            }, BEST_MODEL_PATH)
            improved = " *"
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch:3d}/{MAX_EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"lr={lr:.2e}  "
            f"({elapsed:.1f}s){improved}"
        )

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (best was epoch {best_epoch})")
            break

    total_time = time.time() - start_time
    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Save training history
    history_path = CHECKPOINT_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {history_path}")

    # ------------------------------------------------------------------
    # 6. Load best model and evaluate
    # ------------------------------------------------------------------
    print("\n--- Evaluation ---")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Test set loss
    test_loss = validate(model, loss_fn, test_loader, device)
    print(f"  Test loss: {test_loss:.4f}")

    # Compute all embeddings
    print("\n  Computing embeddings for all bands ...")
    all_embeddings = compute_all_embeddings(model, feature_matrix, device)
    print(f"  Embedding shape: {all_embeddings.shape}")

    # Load ground truth
    true_similar = load_true_similar(test_path)
    band_genres = load_band_genres(METAL_BANDS_CSV) if METAL_BANDS_CSV.exists() else {}

    # Load Last.fm data if available
    lastfm_similar = None
    if LASTFM_CSV.exists():
        try:
            lastfm_similar = load_lastfm_similar(LASTFM_CSV, top_k=10)
            print(f"  Last.fm data loaded: {len(lastfm_similar):,} bands")
        except Exception as e:
            print(f"  WARNING: Could not load Last.fm data: {e}")

    # Get test query IDs (bands that appear as anchors in the test set)
    test_pairs = pd.read_csv(test_path)
    if "band_id_a" in test_pairs.columns:
        test_pairs = test_pairs.rename(columns={"band_id_a": "band_id", "band_id_b": "similar_band_id"})
    test_query_ids = test_pairs["band_id"].unique()
    # Filter to bands that exist in our feature matrix
    valid_test_ids = np.array([
        bid for bid in test_query_ids if bid in {int(b) for b in band_ids}
    ])
    print(f"  Test query bands: {len(valid_test_ids):,}")

    if len(valid_test_ids) == 0:
        print("  WARNING: No valid test query bands. Skipping retrieval metrics.")
    else:
        # Predict top-10 for each test query
        predicted_ids = predict_top_k_from_embeddings(
            valid_test_ids, band_ids, all_embeddings, k=10
        )

        # Evaluate
        siamese_results = evaluate_model(
            "Siamese MLP",
            valid_test_ids,
            predicted_ids,
            true_similar,
            band_genres,
            lastfm_similar,
            k=10,
        )

        print("\n  Siamese MLP Results:")
        for metric, value in siamese_results.items():
            print(f"    {metric}: {value:.4f}")

        # ------------------------------------------------------------------
        # 7. Compare with baseline if it exists
        # ------------------------------------------------------------------
        all_results = {"Siamese MLP": siamese_results}

        # Check for baseline predictions file
        baseline_emb_csv = PROCESSED_DIR / "baseline_embeddings.csv"
        if baseline_emb_csv.exists():
            print("\n  Loading baseline embeddings for comparison ...")
            try:
                bl_df = pd.read_csv(baseline_emb_csv)
                bl_ids = bl_df.iloc[:, 0].values
                bl_emb = bl_df.iloc[:, 1:].values.astype(np.float32)

                bl_pred = predict_top_k_from_embeddings(
                    valid_test_ids, bl_ids, bl_emb, k=10
                )
                baseline_results = evaluate_model(
                    "Cosine Baseline",
                    valid_test_ids,
                    bl_pred,
                    true_similar,
                    band_genres,
                    lastfm_similar,
                    k=10,
                )
                all_results["Cosine Baseline"] = baseline_results
            except Exception as e:
                print(f"  WARNING: Could not load baseline: {e}")

        print_comparison_table(all_results)

    # ------------------------------------------------------------------
    # 8. Save embeddings
    # ------------------------------------------------------------------
    print("--- Saving embeddings ---")
    emb_cols = [f"emb_{i}" for i in range(EMBED_DIM)]
    emb_df = pd.DataFrame(all_embeddings, columns=emb_cols)
    emb_df.insert(0, "band_id", band_ids)
    emb_df.to_csv(EMBEDDINGS_CSV, index=False)
    print(f"  Saved: {EMBEDDINGS_CSV}")
    print(f"  Shape: {emb_df.shape}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Best model:  {BEST_MODEL_PATH}")
    print(f"  Embeddings:  {EMBEDDINGS_CSV}")
    print(f"  History:     {CHECKPOINT_DIR / 'training_history.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
