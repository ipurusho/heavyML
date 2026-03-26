#!/usr/bin/env python3
"""02c_album_ab_features.py — Aggregate AcousticBrainz features per album (release group).

Takes the album linkage from step 02b and the release_group -> recording mapping,
looks up AcousticBrainz features for each recording, and aggregates them per
release group to produce album-level audio fingerprints.

Usage:
    python pipeline/02c_album_ab_features.py

Input:
    data/processed/ma_album_mb_linkage.csv   — step 02b output (MA band -> MB release group)
    data/processed/mb_rg_recording_map.csv   — step 02b output (release_group_id -> recording_id)
    data/musicbrainz/recording               — MB recording table (id -> gid/MBID)
    data/acousticbrainz/*                    — AB feature CSVs

Output:
    data/processed/ma_album_ab_features.csv  — album-level audio features
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MB_DIR = DATA_DIR / "musicbrainz"
AB_DIR = DATA_DIR / "acousticbrainz"
PROCESSED_DIR = DATA_DIR / "processed"

ALBUM_LINKAGE_CSV = PROCESSED_DIR / "ma_album_mb_linkage.csv"
RG_RECORDING_MAP_CSV = PROCESSED_DIR / "mb_rg_recording_map.csv"
OUTPUT_CSV = PROCESSED_DIR / "ma_album_ab_features.csv"

# ---------------------------------------------------------------------------
# AcousticBrainz feature configuration (same as 02_ab_features.py)
# ---------------------------------------------------------------------------
AB_FEATURE_FILES = {
    "lowlevel": "acousticbrainz_lowlevel.csv",
    "rhythm": "acousticbrainz_rhythm.csv",
    "tonal": "acousticbrainz_tonal.csv",
}

AB_LOWLEVEL_COLS = ["mbid", "average_loudness", "dynamic_complexity", "mfcc_zero_mean"]
AB_RHYTHM_COLS = ["mbid", "bpm", "danceability", "onset_rate"]
AB_TONAL_COLS = ["mbid", "key_key", "key_scale", "tuning_frequency"]

NUMERIC_FEATURES = [
    "average_loudness", "dynamic_complexity", "mfcc_zero_mean",
    "bpm", "danceability", "onset_rate", "tuning_frequency",
]

CATEGORICAL_FEATURES = ["key_key", "key_scale"]


# ===================================================================
# STEP 1: Load inputs
# ===================================================================

def load_album_linkage() -> pd.DataFrame:
    """Load the album linkage CSV from step 02b."""
    print(f"[LOAD] Loading album linkage: {ALBUM_LINKAGE_CSV} ...")
    if not ALBUM_LINKAGE_CSV.exists():
        print(f"  ERROR: {ALBUM_LINKAGE_CSV} not found. Run step 02b first.")
        sys.exit(1)

    df = pd.read_csv(ALBUM_LINKAGE_CSV)
    print(f"  Album linkage rows: {len(df):,}")
    print(f"  Unique bands: {df['ma_band_id'].nunique():,}")
    print(f"  Unique release groups: {df['mb_release_group_id'].nunique():,}")
    return df


def load_rg_recording_map() -> pd.DataFrame:
    """Load the release_group -> recording_id mapping from step 02b."""
    print(f"\n[LOAD] Loading RG-recording map: {RG_RECORDING_MAP_CSV} ...")
    if not RG_RECORDING_MAP_CSV.exists():
        print(f"  ERROR: {RG_RECORDING_MAP_CSV} not found. Run step 02b first.")
        sys.exit(1)

    df = pd.read_csv(RG_RECORDING_MAP_CSV,
                      dtype={"rg_id": "int64", "recording_id": "int64"})
    print(f"  RG-recording pairs: {len(df):,}")
    return df


def load_recording_mbid_map(recording_ids: set[int]) -> pd.DataFrame:
    """Load MB recording table to map recording_id -> recording MBID (gid).

    Only loads the IDs we need for memory efficiency.
    """
    path = MB_DIR / "recording"
    print(f"\n[MB:recording] Loading {path} ...")
    df = pd.read_csv(
        path, sep="\t", header=None, usecols=[0, 1],
        names=["recording_id", "recording_mbid"],
        dtype={"recording_id": "int64", "recording_mbid": "str"},
        na_filter=False, quoting=3,
    )
    print(f"  Total recordings: {len(df):,}")

    df = df[df["recording_id"].isin(recording_ids)].copy()
    print(f"  Filtered to our recording IDs: {len(df):,}")
    return df


# ===================================================================
# STEP 2: Load AcousticBrainz features (reused from 02_ab_features.py)
# ===================================================================

def _find_ab_csv(feature_type: str) -> Path | None:
    """Find an AB feature CSV, trying various naming conventions."""
    norm_name = AB_FEATURE_FILES[feature_type]
    if (AB_DIR / norm_name).exists():
        return AB_DIR / norm_name

    for p in AB_DIR.glob(f"*-{feature_type}.csv"):
        return p
    for p in AB_DIR.glob(f"*{feature_type}*.csv"):
        return p

    return None


def _load_ab_csv_filtered(
    path: Path, recording_mbids: set[str], keep_cols: list[str]
) -> pd.DataFrame:
    """Load an AB feature CSV in chunks, filtering to our recording MBIDs."""
    chunks = []
    chunk_size = 500_000

    header_df = pd.read_csv(path, nrows=0)
    all_cols = list(header_df.columns)

    col_map = {}
    for desired in keep_cols:
        if desired in all_cols:
            col_map[desired] = desired
        else:
            for actual in all_cols:
                if actual.lower() == desired.lower():
                    col_map[desired] = actual
                    break

    if "mbid" not in col_map:
        for c in all_cols:
            if "mbid" in c.lower() or "recording" in c.lower():
                col_map["mbid"] = c
                break

    if "mbid" not in col_map:
        print(f"    WARNING: Could not find mbid column. Available: {all_cols}")
        return pd.DataFrame(columns=keep_cols)

    usecols = [col_map.get(c, c) for c in keep_cols if c in col_map]
    if "submission_offset" in all_cols and "submission_offset" not in usecols:
        usecols.append("submission_offset")
    mbid_col = col_map["mbid"]

    reader = pd.read_csv(path, usecols=usecols, chunksize=chunk_size, low_memory=False)

    for chunk in tqdm(reader, desc=f"    Reading {path.name}", unit="chunk"):
        if mbid_col != "mbid":
            chunk = chunk.rename(columns={mbid_col: "mbid"})
        mask = chunk["mbid"].isin(recording_mbids)
        filtered = chunk[mask]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame(columns=keep_cols)

    result = pd.concat(chunks, ignore_index=True)

    if "submission_offset" in result.columns:
        result = result.sort_values("submission_offset").drop_duplicates(
            subset=["mbid"], keep="first"
        )
        result = result.drop(columns=["submission_offset"])

    reverse_map = {v: k for k, v in col_map.items() if v != k}
    if reverse_map:
        result = result.rename(columns=reverse_map)

    final_cols = [c for c in keep_cols if c in result.columns]
    return result[final_cols]


def load_ab_features(recording_mbids: set[str]) -> pd.DataFrame:
    """Load AB feature CSVs and filter to our recording MBIDs."""
    print("\n[AB] Loading AcousticBrainz feature CSVs ...")

    # Check all AB files exist first
    for ftype in AB_FEATURE_FILES:
        path = _find_ab_csv(ftype)
        if path is None:
            print(f"  ERROR: Missing {ftype} feature CSV in {AB_DIR}/")
            print("  Run step 02 first to ensure AB data is downloaded.")
            sys.exit(1)

    ll_path = _find_ab_csv("lowlevel")
    print(f"\n  Loading lowlevel: {ll_path} ...")
    ll_df = _load_ab_csv_filtered(ll_path, recording_mbids, AB_LOWLEVEL_COLS)
    print(f"    Matched recordings: {len(ll_df):,}")

    rh_path = _find_ab_csv("rhythm")
    print(f"\n  Loading rhythm: {rh_path} ...")
    rh_df = _load_ab_csv_filtered(rh_path, recording_mbids, AB_RHYTHM_COLS)
    print(f"    Matched recordings: {len(rh_df):,}")

    tn_path = _find_ab_csv("tonal")
    print(f"\n  Loading tonal: {tn_path} ...")
    tn_df = _load_ab_csv_filtered(tn_path, recording_mbids, AB_TONAL_COLS)
    print(f"    Matched recordings: {len(tn_df):,}")

    print("\n  Merging feature tables ...")
    features = ll_df.merge(rh_df, on="mbid", how="inner")
    features = features.merge(tn_df, on="mbid", how="inner")
    print(f"  Recordings with all 3 feature types: {len(features):,}")

    return features


# ===================================================================
# STEP 3: Aggregate features per release group (album)
# ===================================================================

def aggregate_album_features(
    album_linkage: pd.DataFrame,
    rg_recording: pd.DataFrame,
    rec_mbid_map: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate AB features per release group.

    For numeric features: mean across all recordings in the release group.
    For categorical features (key_key, key_scale): mode (most common value).
    """
    print("\n[AGGREGATE] Building album-level features ...")

    # 1. Map recording_id -> recording MBID in the rg_recording table
    rg_rec = rg_recording.merge(rec_mbid_map, on="recording_id", how="inner")
    print(f"  RG-recording pairs with MBIDs: {len(rg_rec):,}")

    # 2. Join with AB features on recording MBID
    rg_features = rg_rec.merge(features, left_on="recording_mbid", right_on="mbid", how="inner")
    print(f"  RG-recording-feature rows: {len(rg_features):,}")
    print(f"  Release groups with features: {rg_features['rg_id'].nunique():,}")

    if len(rg_features) == 0:
        print("  WARNING: No features matched any recordings.")
        return pd.DataFrame()

    # 3. Coerce numeric columns
    for col in NUMERIC_FEATURES:
        if col in rg_features.columns:
            rg_features[col] = pd.to_numeric(rg_features[col], errors="coerce")

    # 4. Aggregate numeric features: mean per release group
    print("  Computing release-group-level means for numeric features ...")
    available_numeric = [c for c in NUMERIC_FEATURES if c in rg_features.columns]
    if available_numeric:
        rg_numeric = rg_features.groupby("rg_id")[available_numeric].mean()
    else:
        rg_numeric = pd.DataFrame(index=rg_features["rg_id"].unique())

    # 5. Aggregate categorical features: mode per release group
    print("  Computing release-group-level modes for categorical features ...")
    available_cat = [c for c in CATEGORICAL_FEATURES if c in rg_features.columns]
    cat_aggs = {}
    for col in available_cat:
        mode_series = rg_features.groupby("rg_id")[col].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
        )
        cat_aggs[col] = mode_series

    if cat_aggs:
        rg_cat = pd.DataFrame(cat_aggs)
    else:
        rg_cat = pd.DataFrame(index=rg_features["rg_id"].unique())

    # 6. Count AB-matched recordings per release group
    ab_counts = rg_features.groupby("rg_id").size().rename("n_ab_matched")

    # 7. Combine
    rg_result = (
        ab_counts.to_frame()
        .join(rg_numeric, how="left")
        .join(rg_cat, how="left")
    )
    rg_result = rg_result.reset_index()

    # 8. Merge with album linkage to get MA band info and RG names
    album_info = album_linkage[[
        "ma_band_id", "ma_name", "mb_release_group_id",
        "mb_release_group_name", "n_recordings",
    ]].drop_duplicates()

    # We need to join on rg_id (int) but album_linkage has mb_release_group_id (gid/uuid)
    # Build rg_id -> rg_gid mapping from the rg_recording + album_linkage
    # Actually, we need the rg_id -> rg_gid mapping. Let's load it from release_group table.
    # Or better: we can join album_linkage on rg_gid via a lookup.

    # Load a minimal rg_id -> rg_gid mapping
    rg_id_gid = _load_rg_id_gid_map(set(rg_result["rg_id"].unique()))
    rg_result = rg_result.merge(rg_id_gid, on="rg_id", how="left")

    # Now join with album_info on rg_gid = mb_release_group_id
    final = rg_result.merge(
        album_info,
        left_on="rg_gid", right_on="mb_release_group_id",
        how="inner"
    )

    # Select and order output columns
    lead_cols = [
        "ma_band_id", "ma_name", "mb_release_group_id",
        "mb_release_group_name", "n_recordings", "n_ab_matched",
    ]
    feature_cols = sorted([c for c in final.columns if c not in lead_cols
                           and c not in ["rg_id", "rg_gid"]])
    final = final[lead_cols + feature_cols]
    final = final.sort_values(["ma_band_id", "mb_release_group_name"])

    return final


def _load_rg_id_gid_map(rg_ids: set[int]) -> pd.DataFrame:
    """Load minimal release_group id -> gid mapping."""
    path = MB_DIR / "release_group"
    print(f"\n[MB:release_group] Loading id->gid mapping ...")
    df = pd.read_csv(
        path, sep="\t", header=None, usecols=[0, 1],
        names=["rg_id", "rg_gid"],
        dtype={"rg_id": "int64", "rg_gid": "str"},
        na_filter=False, quoting=3,
    )
    df = df[df["rg_id"].isin(rg_ids)].copy()
    print(f"  Mapped {len(df):,} release group IDs to GIDs")
    return df


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 70)
    print("heavyML Pipeline Step 02c: Album-Level AcousticBrainz Features")
    print("=" * 70)

    # --- 1. Load album linkage ---
    print("\n--- Step 1: Load album linkage ---")
    album_linkage = load_album_linkage()

    # --- 2. Load RG -> recording mapping ---
    print("\n--- Step 2: Load RG-recording map ---")
    rg_recording = load_rg_recording_map()

    # Filter rg_recording to only release groups in our album linkage
    # (album_linkage has mb_release_group_id as gid, rg_recording has rg_id as int)
    # We need the rg_id -> gid mapping to filter
    our_rg_gids = set(album_linkage["mb_release_group_id"].unique())
    print(f"  Release groups in album linkage: {len(our_rg_gids):,}")

    # Load rg gid -> id mapping for filtering
    rg_gid_to_id = _load_rg_id_gid_map(set(rg_recording["rg_id"].unique()))
    rg_gid_to_id_filtered = rg_gid_to_id[rg_gid_to_id["rg_gid"].isin(our_rg_gids)]
    our_rg_ids = set(rg_gid_to_id_filtered["rg_id"].unique())
    rg_recording = rg_recording[rg_recording["rg_id"].isin(our_rg_ids)].copy()
    print(f"  Filtered RG-recording pairs: {len(rg_recording):,}")

    # --- 3. Resolve recording IDs to MBIDs ---
    print("\n--- Step 3: Resolve recording IDs to MBIDs ---")
    recording_ids = set(rg_recording["recording_id"].unique())
    print(f"  Unique recording IDs to resolve: {len(recording_ids):,}")
    rec_mbid_map = load_recording_mbid_map(recording_ids)

    # --- 4. Load AB features ---
    print("\n--- Step 4: Load AcousticBrainz features ---")
    recording_mbids = set(rec_mbid_map["recording_mbid"].unique())
    print(f"  Looking up {len(recording_mbids):,} recording MBIDs in AB data ...")
    features = load_ab_features(recording_mbids)

    if len(features) == 0:
        print("\n[ERROR] No AB features matched any recording MBIDs.")
        sys.exit(1)

    # --- 5. Aggregate to album level ---
    print("\n--- Step 5: Aggregate features per release group ---")
    result = aggregate_album_features(album_linkage, rg_recording, rec_mbid_map, features)

    if len(result) == 0:
        print("\n[ERROR] No album-level features produced.")
        sys.exit(1)

    # --- 6. Save output ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OUTPUT] Saved: {OUTPUT_CSV}")
    print(f"  Rows: {len(result):,}")
    print(f"  Columns: {list(result.columns)}")

    # --- 7. Coverage report ---
    print()
    print("=" * 70)
    print("COVERAGE REPORT")
    print("=" * 70)

    total_rgs_linkage = album_linkage["mb_release_group_id"].nunique()
    total_rgs_features = result["mb_release_group_id"].nunique()
    total_bands = result["ma_band_id"].nunique()

    print(f"  Release groups in linkage:         {total_rgs_linkage:,}")
    print(f"  -> with AB features:               {total_rgs_features:,}  "
          f"({total_rgs_features / total_rgs_linkage * 100:.1f}%)")
    print(f"  Unique bands with album features:  {total_bands:,}")
    print()

    # Per-album stats
    recs_per_album = result["n_ab_matched"]
    print(f"  AB-matched recordings per album:")
    print(f"    mean:   {recs_per_album.mean():.1f}")
    print(f"    median: {recs_per_album.median():.1f}")
    print(f"    min:    {recs_per_album.min()}")
    print(f"    max:    {recs_per_album.max()}")

    # Feature distributions
    print()
    print("  Feature distributions (album-level averages):")
    for col in NUMERIC_FEATURES:
        if col in result.columns:
            vals = result[col].dropna()
            if len(vals) > 0:
                print(f"    {col:25s}  mean={vals.mean():.3f}  "
                      f"std={vals.std():.3f}  "
                      f"min={vals.min():.3f}  max={vals.max():.3f}")

    if "key_key" in result.columns:
        print()
        print("  Most common album keys:")
        key_counts = result["key_key"].value_counts().head(5)
        for key, count in key_counts.items():
            print(f"    {key}: {count:,} albums")

    # Verdict
    print()
    if total_rgs_features >= 50_000:
        print("VERDICT: Excellent album coverage (>50k albums with features).")
    elif total_rgs_features >= 20_000:
        print("VERDICT: Good album coverage (20k-50k). Sufficient for album-level model.")
    elif total_rgs_features >= 10_000:
        print("VERDICT: Marginal album coverage (10k-20k).")
    else:
        print("VERDICT: Low album coverage (<10k). May need alternative feature sources.")

    print()
    print("=" * 70)
    print("Step 02c complete. Album-level features ready for downstream use.")
    print("=" * 70)


if __name__ == "__main__":
    main()
