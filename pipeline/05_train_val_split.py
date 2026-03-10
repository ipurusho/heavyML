#!/usr/bin/env python3
"""05_train_val_split.py — Split valid similarity pairs into train/val/test sets.

Splits the valid pairs (from step 04) into 70/15/15 train/val/test sets.
Ensures no source band leakage: a band that appears as band_id_a in the
train set will NOT appear as band_id_a in val or test (it may appear as
band_id_b, which is fine — we only prevent leakage of the query side).

If genre information is available (from metal_bands.csv), the split is
stratified by primary genre to ensure genre balance across sets.

Usage:
    python pipeline/05_train_val_split.py

Input:
    data/processed/valid_pairs.csv        — filtered similarity pairs
    data/processed/feature_band_ids.csv   — band_id index mapping
    data/raw/metal_bands.csv              — (optional) genre info for stratification

Output:
    data/processed/train_pairs.csv
    data/processed/val_pairs.csv
    data/processed/test_pairs.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

VALID_PAIRS_CSV = PROCESSED_DIR / "valid_pairs.csv"
BAND_IDS_CSV = PROCESSED_DIR / "feature_band_ids.csv"
METAL_BANDS_CSV = RAW_DIR / "metal_bands.csv"

OUTPUT_TRAIN = PROCESSED_DIR / "train_pairs.csv"
OUTPUT_VAL = PROCESSED_DIR / "val_pairs.csv"
OUTPUT_TEST = PROCESSED_DIR / "test_pairs.csv"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42


def load_valid_pairs() -> pd.DataFrame:
    """Load the valid pairs CSV from step 04."""
    print(f"[LOAD] Loading valid pairs from {VALID_PAIRS_CSV} ...")
    if not VALID_PAIRS_CSV.exists():
        print(f"  ERROR: {VALID_PAIRS_CSV} not found. Run step 04 first.")
        sys.exit(1)

    df = pd.read_csv(VALID_PAIRS_CSV)
    print(f"  Loaded {len(df):,} valid pairs")
    print(f"  Columns: {list(df.columns)}")

    if len(df) == 0:
        print("  WARNING: No valid pairs found. The MA scraper may still be running.")
        print("  Creating empty split files for downstream compatibility.")

    return df


def load_genre_map() -> dict[int, str] | None:
    """Load primary genre for each band from metal_bands.csv.

    Returns a dict mapping band_id -> primary_genre, or None if the file
    is not available.
    """
    if not METAL_BANDS_CSV.exists():
        print(f"\n[GENRE] {METAL_BANDS_CSV} not found — will use random split")
        return None

    print(f"\n[GENRE] Loading genre info from {METAL_BANDS_CSV} ...")
    try:
        df = pd.read_csv(
            METAL_BANDS_CSV,
            usecols=["Band ID", "Genre"],
            dtype={"Band ID": "str", "Genre": "str"},
        )
        df["Band ID"] = pd.to_numeric(df["Band ID"], errors="coerce")
        df = df.dropna(subset=["Band ID"])
        df["Band ID"] = df["Band ID"].astype("int64")
    except (ValueError, KeyError, OverflowError) as e:
        print(f"  WARNING: Could not load genre info: {e}")
        return None

    # Extract primary genre: take first genre before any "/" or ","
    # e.g. "Progressive Death Metal/Folk Metal" -> "Death Metal"
    # But we want a coarser categorization — extract the core subgenre
    def extract_primary_genre(genre_str: str) -> str:
        if pd.isna(genre_str) or not genre_str.strip():
            return "Unknown"
        # Take the first genre listed (before / or ,)
        first = genre_str.split("/")[0].split(",")[0].strip()
        # Extract the core genre keyword for stratification
        # We want broad categories: Death, Black, Thrash, Doom, Power, Heavy, etc.
        core_genres = [
            "Death", "Black", "Thrash", "Doom", "Power", "Heavy",
            "Progressive", "Folk", "Symphonic", "Gothic", "Groove",
            "Speed", "Metalcore", "Deathcore", "Grindcore", "Sludge",
            "Stoner", "Industrial", "Nu", "Alternative", "Djent",
            "Post", "Avant-garde", "Experimental", "Crust", "Punk",
        ]
        for cg in core_genres:
            if cg.lower() in first.lower():
                return cg
        return "Other"

    genre_map = {}
    for _, row in df.iterrows():
        genre_map[int(row["Band ID"])] = extract_primary_genre(row["Genre"])

    # Print genre distribution
    from collections import Counter
    genre_counts = Counter(genre_map.values())
    print(f"  Loaded genres for {len(genre_map):,} bands")
    print(f"  Top genres: {genre_counts.most_common(10)}")

    return genre_map


def split_by_source_band(
    pairs: pd.DataFrame,
    genre_map: dict[int, str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split pairs into train/val/test ensuring no source band leakage.

    Strategy:
    1. Get unique source bands (band_id_a)
    2. Split SOURCE BANDS into train/val/test groups (stratified by genre if possible)
    3. Assign all pairs for each source band to its group

    This prevents a source band from appearing in both train and val/test,
    which would cause data leakage (the model sees queries from the same band
    during training and evaluation).
    """
    if len(pairs) == 0:
        empty = pd.DataFrame(columns=pairs.columns)
        return empty, empty.copy(), empty.copy()

    source_bands = pairs["band_id_a"].unique()
    print(f"\n[SPLIT] Splitting {len(source_bands):,} source bands ...")
    print(f"  Target ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")

    # Handle very small datasets gracefully
    if len(source_bands) < 3:
        print(f"  WARNING: Only {len(source_bands)} source bands — too few for 3-way split.")
        print(f"  Putting all pairs in train set, with empty val/test.")
        empty = pd.DataFrame(columns=pairs.columns)
        return pairs.copy(), empty, empty.copy()

    if len(source_bands) < 10:
        print(f"  WARNING: Only {len(source_bands)} source bands — using simple split.")
        # Simple sequential split for tiny datasets
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(source_bands)
        n_val = max(1, int(len(source_bands) * VAL_RATIO))
        n_test = max(1, int(len(source_bands) * TEST_RATIO))
        n_train = len(source_bands) - n_val - n_test
        if n_train < 1:
            n_train = 1
            n_val = max(1, (len(source_bands) - 1) // 2)
            n_test = len(source_bands) - 1 - n_val
        train_bands = set(source_bands[:n_train])
        val_bands = set(source_bands[n_train:n_train + n_val])
        test_bands = set(source_bands[n_train + n_val:])

        train = pairs[pairs["band_id_a"].isin(train_bands)].copy()
        val = pairs[pairs["band_id_a"].isin(val_bands)].copy()
        test = pairs[pairs["band_id_a"].isin(test_bands)].copy()
        return train, val, test

    # Build stratification labels for source bands
    strat_labels = None
    if genre_map is not None:
        genres = [genre_map.get(int(b), "Unknown") for b in source_bands]
        # Check if we have enough bands per genre for stratification
        from collections import Counter
        genre_counts = Counter(genres)

        # Stratification requires at least 2 samples per class
        # Collapse rare genres into "Other" to make stratification work
        min_count = 2
        rare_genres = {g for g, c in genre_counts.items() if c < min_count}
        if rare_genres:
            genres = [g if g not in rare_genres else "Other" for g in genres]
            genre_counts = Counter(genres)

        # Final check: still need >= 2 per class for stratified split
        too_small = {g for g, c in genre_counts.items() if c < 2}
        if too_small:
            # Collapse everything too small into "Other"
            genres = [g if g not in too_small else "Other" for g in genres]

        strat_labels = genres
        print(f"  Using genre-stratified split ({len(set(genres))} genre groups)")

    # First split: train vs (val+test)
    val_test_ratio = VAL_RATIO + TEST_RATIO  # 0.30
    try:
        train_bands, valtest_bands = train_test_split(
            source_bands,
            test_size=val_test_ratio,
            random_state=RANDOM_SEED,
            stratify=strat_labels,
        )
    except ValueError:
        # Stratification failed (too few samples per class) — fall back to random
        print("  WARNING: Genre stratification failed — falling back to random split")
        train_bands, valtest_bands = train_test_split(
            source_bands,
            test_size=val_test_ratio,
            random_state=RANDOM_SEED,
        )
        strat_labels = None

    # Second split: val vs test (50/50 of the remaining 30%)
    if strat_labels is not None:
        # Get strat labels for valtest bands
        band_to_genre = dict(zip(source_bands, strat_labels))
        valtest_strat = [band_to_genre[b] for b in valtest_bands]
        # Collapse for this smaller subset
        from collections import Counter
        vt_counts = Counter(valtest_strat)
        too_small = {g for g, c in vt_counts.items() if c < 2}
        valtest_strat = [g if g not in too_small else "Other" for g in valtest_strat]
        try:
            val_bands, test_bands = train_test_split(
                valtest_bands,
                test_size=TEST_RATIO / val_test_ratio,  # 0.15/0.30 = 0.5
                random_state=RANDOM_SEED,
                stratify=valtest_strat,
            )
        except ValueError:
            val_bands, test_bands = train_test_split(
                valtest_bands,
                test_size=TEST_RATIO / val_test_ratio,
                random_state=RANDOM_SEED,
            )
    else:
        val_bands, test_bands = train_test_split(
            valtest_bands,
            test_size=TEST_RATIO / val_test_ratio,
            random_state=RANDOM_SEED,
        )

    # Convert to sets for fast lookup
    train_set = set(train_bands)
    val_set = set(val_bands)
    test_set = set(test_bands)

    # Assign pairs to splits
    train = pairs[pairs["band_id_a"].isin(train_set)].copy()
    val = pairs[pairs["band_id_a"].isin(val_set)].copy()
    test = pairs[pairs["band_id_a"].isin(test_set)].copy()

    return train, val, test


def print_split_stats(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
):
    """Print statistics about the train/val/test split."""
    total = len(train) + len(val) + len(test)

    print("\n" + "=" * 70)
    print("SPLIT STATISTICS")
    print("=" * 70)

    if total == 0:
        print("  No pairs to split. Scraper may still be running.")
        return

    print(f"  {'Set':<10} {'Pairs':>8} {'%':>8} {'Source bands':>14} {'Target bands':>14}")
    print(f"  {'-' * 54}")

    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        n_pairs = len(df)
        pct = n_pairs / total * 100 if total > 0 else 0
        n_src = df["band_id_a"].nunique() if n_pairs > 0 else 0
        n_tgt = df["band_id_b"].nunique() if n_pairs > 0 else 0
        print(f"  {name:<10} {n_pairs:>8,} {pct:>7.1f}% {n_src:>14,} {n_tgt:>14,}")

    print(f"  {'Total':<10} {total:>8,} {'100.0%':>8}")

    # Check for source band leakage
    if len(train) > 0 and len(val) > 0:
        train_src = set(train["band_id_a"].unique())
        val_src = set(val["band_id_a"].unique())
        test_src = set(test["band_id_a"].unique()) if len(test) > 0 else set()
        leak_val = train_src & val_src
        leak_test = train_src & test_src
        leak_valtest = val_src & test_src

        print(f"\n  Source band leakage check:")
        print(f"    Train-Val overlap:   {len(leak_val)} bands {'(CLEAN)' if len(leak_val) == 0 else '(LEAK!)'}")
        print(f"    Train-Test overlap:  {len(leak_test)} bands {'(CLEAN)' if len(leak_test) == 0 else '(LEAK!)'}")
        print(f"    Val-Test overlap:    {len(leak_valtest)} bands {'(CLEAN)' if len(leak_valtest) == 0 else '(LEAK!)'}")

    # Pairs per source band distribution for each split
    print(f"\n  Pairs per source band:")
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        if len(df) > 0:
            ppb = df.groupby("band_id_a").size()
            print(f"    {name:<6} mean={ppb.mean():.1f}  median={ppb.median():.1f}  "
                  f"min={ppb.min()}  max={ppb.max()}")


def main():
    print("=" * 70)
    print("heavyML Pipeline Step 05: Train/Val/Test Split")
    print("=" * 70)

    # ---- 1. Load data ----
    pairs = load_valid_pairs()
    genre_map = load_genre_map()

    # ---- 2. Split ----
    train, val, test = split_by_source_band(pairs, genre_map)

    # ---- 3. Save ----
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[SAVE] Writing split files ...")
    train.to_csv(OUTPUT_TRAIN, index=False)
    print(f"  Train: {OUTPUT_TRAIN}  ({len(train):,} pairs)")
    val.to_csv(OUTPUT_VAL, index=False)
    print(f"  Val:   {OUTPUT_VAL}  ({len(val):,} pairs)")
    test.to_csv(OUTPUT_TEST, index=False)
    print(f"  Test:  {OUTPUT_TEST}  ({len(test):,} pairs)")

    # ---- 4. Print stats ----
    print_split_stats(train, val, test)

    print("\n" + "=" * 70)
    print("Step 05 complete. Ready for model training or baseline evaluation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
