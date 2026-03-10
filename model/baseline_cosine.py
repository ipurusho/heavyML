#!/usr/bin/env python3
"""baseline_cosine.py — Cosine similarity baseline for band recommendation.

Computes pairwise cosine similarity between all band audio fingerprints
and evaluates against ground-truth similarity labels (MA pairs) and
held-out Last.fm data.

This is the "dumb" baseline that the Siamese model must beat. If cosine
similarity on raw features already captures band similarity well, the
neural model adds no value.

Usage:
    python model/baseline_cosine.py
    python model/baseline_cosine.py --top-k 20
    python model/baseline_cosine.py --no-lastfm

Input:
    data/processed/feature_matrix.npy        — (N_bands x D) feature matrix
    data/processed/feature_band_ids.csv      — band_id index mapping
    data/processed/val_pairs.csv             — validation pairs
    data/processed/test_pairs.csv            — test pairs
    data/processed/lastfm_similar_artists.csv — (optional) Last.fm held-out eval
    data/raw/metal_bands.csv                 — (optional) genre info for genre purity

Output:
    data/processed/baseline_predictions.csv  — top-K predictions per band
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

FEATURE_MATRIX = PROCESSED_DIR / "feature_matrix.npy"
BAND_IDS_CSV = PROCESSED_DIR / "feature_band_ids.csv"
VAL_PAIRS_CSV = PROCESSED_DIR / "val_pairs.csv"
TEST_PAIRS_CSV = PROCESSED_DIR / "test_pairs.csv"
LASTFM_CSV = PROCESSED_DIR / "lastfm_similar_artists.csv"
METAL_BANDS_CSV = RAW_DIR / "metal_bands.csv"

OUTPUT_PREDICTIONS = PROCESSED_DIR / "baseline_predictions.csv"

DEFAULT_TOP_K = 10


def load_data():
    """Load feature matrix, band IDs, and validation/test pairs."""
    print("[LOAD] Loading feature matrix and band IDs ...")

    if not FEATURE_MATRIX.exists():
        print(f"  ERROR: {FEATURE_MATRIX} not found. Run step 04 first.")
        sys.exit(1)
    if not BAND_IDS_CSV.exists():
        print(f"  ERROR: {BAND_IDS_CSV} not found. Run step 04 first.")
        sys.exit(1)

    matrix = np.load(FEATURE_MATRIX)
    band_ids_df = pd.read_csv(BAND_IDS_CSV)
    band_ids = band_ids_df["band_id"].values

    print(f"  Feature matrix: {matrix.shape}")
    print(f"  Band IDs: {len(band_ids):,}")

    # Load val and test pairs
    val_pairs = None
    test_pairs = None

    if VAL_PAIRS_CSV.exists():
        val_pairs = pd.read_csv(VAL_PAIRS_CSV)
        print(f"  Val pairs: {len(val_pairs):,}")
    else:
        print(f"  Val pairs: not found ({VAL_PAIRS_CSV})")

    if TEST_PAIRS_CSV.exists():
        test_pairs = pd.read_csv(TEST_PAIRS_CSV)
        print(f"  Test pairs: {len(test_pairs):,}")
    else:
        print(f"  Test pairs: not found ({TEST_PAIRS_CSV})")

    return matrix, band_ids, val_pairs, test_pairs


def load_genre_map() -> dict[int, str] | None:
    """Load primary genre for each band from metal_bands.csv."""
    if not METAL_BANDS_CSV.exists():
        return None

    try:
        df = pd.read_csv(
            METAL_BANDS_CSV,
            usecols=["Band ID", "Genre"],
            dtype={"Band ID": "str", "Genre": "str"},
        )
        df["Band ID"] = pd.to_numeric(df["Band ID"], errors="coerce")
        df = df.dropna(subset=["Band ID"])
        df["Band ID"] = df["Band ID"].astype("int64")
    except (ValueError, KeyError, OverflowError):
        return None

    def extract_primary(genre_str: str) -> str:
        if pd.isna(genre_str) or not genre_str.strip():
            return "Unknown"
        first = genre_str.split("/")[0].split(",")[0].strip()
        core = [
            "Death", "Black", "Thrash", "Doom", "Power", "Heavy",
            "Progressive", "Folk", "Symphonic", "Gothic", "Groove",
            "Speed", "Metalcore", "Deathcore", "Grindcore", "Sludge",
            "Stoner", "Industrial", "Nu", "Alternative",
        ]
        for cg in core:
            if cg.lower() in first.lower():
                return cg
        return "Other"

    return {int(row["Band ID"]): extract_primary(row["Genre"]) for _, row in df.iterrows()}


def load_lastfm_pairs(band_ids_set: set) -> pd.DataFrame | None:
    """Load Last.fm similar artists as held-out evaluation data.

    Returns pairs where both bands are in our feature matrix.
    """
    if not LASTFM_CSV.exists():
        return None

    print(f"\n[LASTFM] Loading held-out Last.fm pairs from {LASTFM_CSV} ...")
    df = pd.read_csv(LASTFM_CSV)
    print(f"  Total Last.fm pairs: {len(df):,}")

    # Columns: band_id, band_name, similar_name, similar_band_id, match_score
    # Filter to pairs where both bands are in our feature matrix
    # similar_band_id may be NaN for bands not found in MA
    df = df.dropna(subset=["similar_band_id"]).copy()
    df["band_id"] = pd.to_numeric(df["band_id"], errors="coerce")
    df["similar_band_id"] = pd.to_numeric(df["similar_band_id"], errors="coerce")
    df = df.dropna(subset=["band_id", "similar_band_id"]).copy()
    df["band_id"] = df["band_id"].astype(int)
    df["similar_band_id"] = df["similar_band_id"].astype(int)

    # Filter to bands in our feature matrix
    mask = df["band_id"].isin(band_ids_set) & df["similar_band_id"].isin(band_ids_set)
    df = df[mask].copy()
    print(f"  Valid Last.fm pairs (both bands in matrix): {len(df):,}")

    if len(df) == 0:
        return None

    # Rename for consistency
    df = df.rename(columns={
        "band_id": "band_id_a",
        "similar_band_id": "band_id_b",
        "match_score": "score",
    })

    return df[["band_id_a", "band_id_b", "score"]]


def compute_cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    print(f"\n[COSINE] Computing pairwise cosine similarity for {matrix.shape[0]:,} bands ...")
    sim_matrix = cosine_similarity(matrix)

    # Zero out self-similarity (diagonal)
    np.fill_diagonal(sim_matrix, -1.0)

    print(f"  Similarity matrix shape: {sim_matrix.shape}")
    print(f"  Non-diagonal min: {sim_matrix[sim_matrix > -1].min():.4f}")
    print(f"  Non-diagonal max: {sim_matrix[sim_matrix > -1].max():.4f}")
    print(f"  Non-diagonal mean: {sim_matrix[sim_matrix > -1].mean():.4f}")

    return sim_matrix


def get_top_k_predictions(
    sim_matrix: np.ndarray, band_ids: np.ndarray, top_k: int
) -> dict[int, list[int]]:
    """For each band, get the top-K most similar bands by cosine similarity.

    Returns:
        dict mapping band_id -> list of top-K similar band_ids (ordered by similarity)
    """
    print(f"\n[PREDICT] Generating top-{top_k} predictions per band ...")
    predictions = {}

    for i in range(len(band_ids)):
        # Get top-K indices (excluding self)
        sim_scores = sim_matrix[i]
        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        top_band_ids = band_ids[top_indices].tolist()
        predictions[int(band_ids[i])] = top_band_ids

    print(f"  Generated predictions for {len(predictions):,} bands")
    return predictions


def evaluate_recall_at_k(
    predictions: dict[int, list[int]],
    ground_truth_pairs: pd.DataFrame,
    k: int,
    label: str = "",
) -> float:
    """Compute Recall@K for the predictions against ground truth pairs.

    For each source band in ground truth, compute the fraction of its
    true similar bands that appear in the model's top-K predictions.
    Average across all source bands.

    Args:
        predictions: dict mapping band_id -> list of top-K predicted similar bands
        ground_truth_pairs: DataFrame with columns [band_id_a, band_id_b, score]
        k: number of top predictions to consider
        label: label for printing

    Returns:
        Mean Recall@K across all source bands
    """
    if ground_truth_pairs is None or len(ground_truth_pairs) == 0:
        return float("nan")

    # Build ground truth: band_id_a -> set of true similar band_ids
    gt_dict = {}
    for _, row in ground_truth_pairs.iterrows():
        src = int(row["band_id_a"])
        tgt = int(row["band_id_b"])
        if src not in gt_dict:
            gt_dict[src] = set()
        gt_dict[src].add(tgt)

    recalls = []
    for src_band, true_similar in gt_dict.items():
        if src_band not in predictions:
            # Band not in feature matrix — skip
            continue

        pred_top_k = set(predictions[src_band][:k])
        if len(true_similar) == 0:
            continue

        hits = len(pred_top_k & true_similar)
        recall = hits / len(true_similar)
        recalls.append(recall)

    if not recalls:
        return float("nan")

    mean_recall = np.mean(recalls)
    return mean_recall


def evaluate_genre_purity_at_k(
    predictions: dict[int, list[int]],
    genre_map: dict[int, str] | None,
    k: int,
) -> float:
    """Compute Genre Purity@K: fraction of top-K predictions sharing the source's genre.

    For each band, look at the top-K predicted similar bands and compute
    what fraction share the same primary genre as the source band.
    Average across all bands.
    """
    if genre_map is None:
        return float("nan")

    purities = []
    for src_band, pred_bands in predictions.items():
        src_genre = genre_map.get(src_band)
        if src_genre is None or src_genre in ("Unknown", "Other"):
            continue

        top_k_bands = pred_bands[:k]
        if not top_k_bands:
            continue

        same_genre = sum(1 for b in top_k_bands if genre_map.get(b) == src_genre)
        purity = same_genre / len(top_k_bands)
        purities.append(purity)

    if not purities:
        return float("nan")

    return np.mean(purities)


def save_predictions(
    predictions: dict[int, list[int]],
    sim_matrix: np.ndarray,
    band_ids: np.ndarray,
    top_k: int,
):
    """Save top-K predictions to CSV."""
    print(f"\n[SAVE] Writing predictions to {OUTPUT_PREDICTIONS} ...")

    band_id_to_idx = {int(b): i for i, b in enumerate(band_ids)}
    rows = []

    for src_band, pred_bands in predictions.items():
        src_idx = band_id_to_idx[src_band]
        for rank, tgt_band in enumerate(pred_bands[:top_k], start=1):
            tgt_idx = band_id_to_idx[tgt_band]
            sim_score = float(sim_matrix[src_idx, tgt_idx])
            rows.append({
                "source_band_id": src_band,
                "predicted_band_id": tgt_band,
                "rank": rank,
                "cosine_similarity": round(sim_score, 6),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"  Saved {len(df):,} prediction rows ({len(predictions):,} bands x top-{top_k})")


def main():
    parser = argparse.ArgumentParser(
        description="Cosine similarity baseline for band recommendation"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of top similar bands to predict (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--no-lastfm", action="store_true",
        help="Skip Last.fm held-out evaluation"
    )
    args = parser.parse_args()
    top_k = args.top_k

    print("=" * 70)
    print("heavyML Baseline: Cosine Similarity")
    print("=" * 70)

    # ---- 1. Load data ----
    matrix, band_ids, val_pairs, test_pairs = load_data()
    genre_map = load_genre_map()
    band_ids_set = set(int(b) for b in band_ids)

    lastfm_pairs = None
    if not args.no_lastfm:
        lastfm_pairs = load_lastfm_pairs(band_ids_set)

    # ---- 2. Compute cosine similarity ----
    sim_matrix = compute_cosine_similarity(matrix)

    # ---- 3. Get top-K predictions ----
    predictions = get_top_k_predictions(sim_matrix, band_ids, top_k)

    # ---- 4. Evaluate ----
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS (top-K = {top_k})")
    print("=" * 70)

    # Recall@K on val set
    val_recall = evaluate_recall_at_k(predictions, val_pairs, top_k, "Val")
    print(f"\n  Recall@{top_k} (Val set):   {val_recall:.4f}" if not np.isnan(val_recall)
          else f"\n  Recall@{top_k} (Val set):   N/A (no val pairs)")

    # Recall@K on test set
    test_recall = evaluate_recall_at_k(predictions, test_pairs, top_k, "Test")
    print(f"  Recall@{top_k} (Test set):  {test_recall:.4f}" if not np.isnan(test_recall)
          else f"  Recall@{top_k} (Test set):  N/A (no test pairs)")

    # Recall@K on Last.fm held-out
    if lastfm_pairs is not None:
        lastfm_recall = evaluate_recall_at_k(predictions, lastfm_pairs, top_k, "Last.fm")
        print(f"  Recall@{top_k} (Last.fm):   {lastfm_recall:.4f}" if not np.isnan(lastfm_recall)
              else f"  Recall@{top_k} (Last.fm):   N/A")

    # Genre Purity@K
    genre_purity = evaluate_genre_purity_at_k(predictions, genre_map, top_k)
    print(f"\n  Genre Purity@{top_k}:        {genre_purity:.4f}" if not np.isnan(genre_purity)
          else f"\n  Genre Purity@{top_k}:        N/A (no genre data)")

    # Additional Recall breakdowns
    if val_pairs is not None and len(val_pairs) > 0:
        for k_val in [1, 5, 10, 20, 50]:
            if k_val > top_k:
                break
            r = evaluate_recall_at_k(predictions, val_pairs, k_val, f"Val@{k_val}")
            if not np.isnan(r):
                print(f"  Recall@{k_val:<3d} (Val):       {r:.4f}")

    # ---- 5. Save predictions ----
    save_predictions(predictions, sim_matrix, band_ids, top_k)

    # ---- 6. Summary ----
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)
    print(f"  Model:              Cosine similarity on {matrix.shape[1]}-dim audio features")
    print(f"  Bands:              {len(band_ids):,}")
    print(f"  Top-K:              {top_k}")
    if not np.isnan(val_recall):
        print(f"  Val Recall@{top_k}:     {val_recall:.4f}")
    if not np.isnan(test_recall):
        print(f"  Test Recall@{top_k}:    {test_recall:.4f}")
    if not np.isnan(genre_purity):
        print(f"  Genre Purity@{top_k}:   {genre_purity:.4f}")
    print()
    print("  This is the baseline the Siamese model must beat.")
    print("  If Recall@10 > 0.10, cosine on raw features already captures")
    print("  meaningful similarity. The neural model should push this higher.")
    print("=" * 70)


if __name__ == "__main__":
    main()
