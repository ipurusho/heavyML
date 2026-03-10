#!/usr/bin/env python3
"""04_feature_matrix.py — Build the feature matrix for model training.

Takes the band-level audio features (step 02) and similarity pairs (step 03),
filters to bands that have both features AND similarity labels, normalizes
numeric features, one-hot encodes categorical features, and outputs a numpy
feature matrix plus index/pair CSVs.

Usage:
    python pipeline/04_feature_matrix.py

Input:
    data/processed/ma_ab_features.csv     — 16k bands with audio features
    data/processed/ma_similar_artists.csv  — similarity pairs from MA scraper

Output:
    data/processed/feature_matrix.npy     — (N_bands x D_features) numpy array
    data/processed/feature_band_ids.csv   — band_id index mapping (row i -> band_id)
    data/processed/valid_pairs.csv        — filtered similarity pairs (both bands in matrix)
    data/processed/scaler_params.npz      — StandardScaler fit params for inference
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

FEATURES_CSV = PROCESSED_DIR / "ma_ab_features.csv"
SIMILAR_CSV = PROCESSED_DIR / "ma_similar_artists.csv"

OUTPUT_MATRIX = PROCESSED_DIR / "feature_matrix.npy"
OUTPUT_BAND_IDS = PROCESSED_DIR / "feature_band_ids.csv"
OUTPUT_VALID_PAIRS = PROCESSED_DIR / "valid_pairs.csv"
OUTPUT_SCALER = PROCESSED_DIR / "scaler_params.npz"


# ---------------------------------------------------------------------------
# Numeric and categorical feature definitions
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "average_loudness",
    "bpm",
    "danceability",
    "dynamic_complexity",
    "mfcc_zero_mean",
    "onset_rate",
    "tuning_frequency",
]

# 12 possible keys for one-hot encoding
KEY_CLASSES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

# key_scale: binary encoding (major=1, minor=0)
SCALE_MAP = {"major": 1, "minor": 0}


def load_features() -> pd.DataFrame:
    """Load the band-level audio features CSV."""
    print(f"[LOAD] Loading features from {FEATURES_CSV} ...")
    if not FEATURES_CSV.exists():
        print(f"  ERROR: {FEATURES_CSV} not found. Run step 02 first.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    print(f"  Loaded {len(df):,} bands with {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # Validate required columns
    required = ["ma_band_id"] + NUMERIC_FEATURES + ["key_key", "key_scale"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing required columns: {missing}")
        sys.exit(1)

    return df


def load_similar_pairs() -> pd.DataFrame:
    """Load the MA similar artists CSV."""
    print(f"\n[LOAD] Loading similarity pairs from {SIMILAR_CSV} ...")
    if not SIMILAR_CSV.exists():
        print(f"  ERROR: {SIMILAR_CSV} not found. Run step 03 first.")
        sys.exit(1)

    df = pd.read_csv(SIMILAR_CSV)
    print(f"  Loaded {len(df):,} similarity pairs")
    print(f"  Columns: {list(df.columns)}")

    # Normalize column names: the scraper outputs band_id, similar_band_id, similar_band_name, score
    col_map = {}
    if "band_id" in df.columns and "band_id_a" not in df.columns:
        col_map["band_id"] = "band_id_a"
    if "similar_band_id" in df.columns and "band_id_b" not in df.columns:
        col_map["similar_band_id"] = "band_id_b"
    if col_map:
        df = df.rename(columns=col_map)

    # Ensure required columns exist
    for col in ["band_id_a", "band_id_b", "score"]:
        if col not in df.columns:
            print(f"  ERROR: Missing column '{col}' after renaming.")
            print(f"  Available columns: {list(df.columns)}")
            sys.exit(1)

    # Keep only the columns we need
    keep_cols = ["band_id_a", "band_id_b", "score"]
    extra = [c for c in df.columns if c not in keep_cols]
    if extra:
        print(f"  Dropping extra columns: {extra}")
    df = df[keep_cols].copy()

    # Ensure numeric types
    df["band_id_a"] = pd.to_numeric(df["band_id_a"], errors="coerce")
    df["band_id_b"] = pd.to_numeric(df["band_id_b"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["band_id_a", "band_id_b"]).copy()
    df["band_id_a"] = df["band_id_a"].astype(int)
    df["band_id_b"] = df["band_id_b"].astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    """Build the normalized feature matrix from band-level features.

    Returns:
        matrix: (N_bands x D_features) numpy array
        band_ids_df: DataFrame with band_id index mapping
        scaler: fitted StandardScaler (for inference-time use)
    """
    print("\n[BUILD] Constructing feature matrix ...")

    # ---- 1. Clean data: drop rows with NaN in any numeric feature ----
    n_before = len(df)
    df = df.dropna(subset=NUMERIC_FEATURES + ["key_key", "key_scale"]).copy()
    n_after = len(df)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after:,} bands with missing features")

    # ---- 2. Sort by band_id for deterministic ordering ----
    df = df.sort_values("ma_band_id").reset_index(drop=True)

    # ---- 3. Extract and normalize numeric features ----
    numeric_data = df[NUMERIC_FEATURES].values.astype(np.float64)
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_data)
    print(f"  Numeric features: {len(NUMERIC_FEATURES)} columns, StandardScaler fitted")

    # ---- 4. One-hot encode key_key (12 possible keys) ----
    key_values = df["key_key"].values
    key_onehot = np.zeros((len(df), len(KEY_CLASSES)), dtype=np.float64)
    key_to_idx = {k: i for i, k in enumerate(KEY_CLASSES)}
    for i, k in enumerate(key_values):
        if k in key_to_idx:
            key_onehot[i, key_to_idx[k]] = 1.0
        else:
            # Unknown key — leave as all zeros (will be rare)
            print(f"  WARNING: Unknown key '{k}' for band at row {i}")
    n_unknown_keys = (key_onehot.sum(axis=1) == 0).sum()
    if n_unknown_keys > 0:
        print(f"  {n_unknown_keys} bands with unrecognized key values")
    print(f"  Key one-hot: {len(KEY_CLASSES)} columns ({KEY_CLASSES})")

    # ---- 5. Binary encode key_scale (major=1, minor=0) ----
    scale_values = df["key_scale"].map(SCALE_MAP).values.astype(np.float64).reshape(-1, 1)
    n_unknown_scales = np.isnan(scale_values).sum()
    if n_unknown_scales > 0:
        print(f"  WARNING: {n_unknown_scales} bands with unrecognized scale values")
        scale_values = np.nan_to_num(scale_values, nan=0.0)
    print(f"  Scale binary: 1 column (major=1, minor=0)")

    # ---- 6. Concatenate all features ----
    matrix = np.hstack([numeric_scaled, key_onehot, scale_values])
    print(f"\n  Final feature matrix shape: {matrix.shape}")
    print(f"    {len(NUMERIC_FEATURES)} numeric + {len(KEY_CLASSES)} key_onehot + 1 scale = {matrix.shape[1]} total dims")

    # ---- 7. Build band_id index ----
    band_ids_df = df[["ma_band_id"]].copy()
    band_ids_df = band_ids_df.rename(columns={"ma_band_id": "band_id"})
    band_ids_df = band_ids_df.reset_index(drop=True)

    return matrix, band_ids_df, scaler


def filter_valid_pairs(pairs: pd.DataFrame, valid_band_ids: set) -> pd.DataFrame:
    """Filter similarity pairs to only those where BOTH bands are in the feature matrix."""
    print("\n[FILTER] Filtering pairs to bands with features ...")
    print(f"  Total pairs before filter: {len(pairs):,}")
    print(f"  Bands in feature matrix: {len(valid_band_ids):,}")

    mask = pairs["band_id_a"].isin(valid_band_ids) & pairs["band_id_b"].isin(valid_band_ids)
    valid = pairs[mask].copy()
    print(f"  Valid pairs (both bands have features): {len(valid):,}")

    if len(valid) == 0:
        print("  WARNING: No valid pairs found! The scraper may still be running.")
        print("  The script will still output empty files for downstream compatibility.")

    return valid


def print_stats(matrix: np.ndarray, band_ids: pd.DataFrame, valid_pairs: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("FEATURE MATRIX STATS")
    print("=" * 70)
    print(f"  Total bands in matrix:     {matrix.shape[0]:,}")
    print(f"  Feature dimensionality:    {matrix.shape[1]}")
    print(f"  Total valid pairs:         {len(valid_pairs):,}")

    if len(valid_pairs) > 0:
        # Pairs per band distribution
        pairs_per_band = valid_pairs.groupby("band_id_a").size()
        print(f"\n  Pairs per source band:")
        print(f"    mean:   {pairs_per_band.mean():.1f}")
        print(f"    median: {pairs_per_band.median():.1f}")
        print(f"    min:    {pairs_per_band.min()}")
        print(f"    max:    {pairs_per_band.max()}")
        print(f"    std:    {pairs_per_band.std():.1f}")

        # Unique bands in pairs
        unique_a = valid_pairs["band_id_a"].nunique()
        unique_b = valid_pairs["band_id_b"].nunique()
        all_unique = pd.concat([valid_pairs["band_id_a"], valid_pairs["band_id_b"]]).nunique()
        print(f"\n  Unique source bands (band_id_a):  {unique_a:,}")
        print(f"  Unique target bands (band_id_b):  {unique_b:,}")
        print(f"  Unique bands in any pair:         {all_unique:,}")

        # Score distribution
        print(f"\n  Similarity score distribution:")
        print(f"    mean:   {valid_pairs['score'].mean():.1f}")
        print(f"    median: {valid_pairs['score'].median():.1f}")
        print(f"    min:    {valid_pairs['score'].min()}")
        print(f"    max:    {valid_pairs['score'].max()}")

    # Feature matrix stats
    print(f"\n  Feature matrix value ranges:")
    feature_names = (
        NUMERIC_FEATURES
        + [f"key_{k}" for k in KEY_CLASSES]
        + ["scale_major"]
    )
    for i, name in enumerate(feature_names):
        col = matrix[:, i]
        print(f"    {name:25s}  mean={col.mean():+.3f}  std={col.std():.3f}  "
              f"min={col.min():.3f}  max={col.max():.3f}")


def main():
    print("=" * 70)
    print("heavyML Pipeline Step 04: Feature Matrix Construction")
    print("=" * 70)

    # ---- 1. Load data ----
    features_df = load_features()
    pairs_df = load_similar_pairs()

    # ---- 2. Build feature matrix ----
    matrix, band_ids_df, scaler = build_feature_matrix(features_df)

    # ---- 3. Filter valid pairs ----
    valid_band_ids = set(band_ids_df["band_id"].values)
    valid_pairs = filter_valid_pairs(pairs_df, valid_band_ids)

    # ---- 4. Save outputs ----
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[SAVE] Writing outputs ...")

    # Feature matrix
    np.save(OUTPUT_MATRIX, matrix)
    print(f"  Feature matrix:  {OUTPUT_MATRIX}  shape={matrix.shape}")

    # Band ID index
    band_ids_df.to_csv(OUTPUT_BAND_IDS, index=False)
    print(f"  Band ID index:   {OUTPUT_BAND_IDS}  rows={len(band_ids_df):,}")

    # Valid pairs
    valid_pairs.to_csv(OUTPUT_VALID_PAIRS, index=False)
    print(f"  Valid pairs:     {OUTPUT_VALID_PAIRS}  rows={len(valid_pairs):,}")

    # Scaler params (for inference-time normalization)
    np.savez(
        OUTPUT_SCALER,
        mean=scaler.mean_,
        scale=scaler.scale_,
        feature_names=NUMERIC_FEATURES,
    )
    print(f"  Scaler params:   {OUTPUT_SCALER}")

    # ---- 5. Print stats ----
    print_stats(matrix, band_ids_df, valid_pairs)

    print("\n" + "=" * 70)
    print("Step 04 complete. Ready for step 05 (train/val/test split).")
    print("=" * 70)


if __name__ == "__main__":
    main()
