#!/usr/bin/env python3
"""02_ab_features.py — Bridge MA band IDs to AcousticBrainz audio features.

Linkage chain:
  MA Band ID → (step 01 linkage) → MB Artist MBID
    → artist table (gid → id)
    → artist_credit_name table (artist = artist.id → artist_credit)
    → recording table (artist_credit → gid = recording MBID)
    → AcousticBrainz feature CSVs (keyed by recording MBID)

MusicBrainz TSV schemas (headerless, tab-separated, column indices):
  artist:              id(0), gid(1), name(2), sort_name(3), ... (19 cols)
  artist_credit_name:  artist_credit(0), position(1), artist(2), name(3), join_phrase(4)
  recording:           id(0), gid(1), name(2), artist_credit(3), length(4), ... (9 cols)

AcousticBrainz feature CSVs (with headers, comma-separated):
  lowlevel: mbid, submission_offset, average_loudness, dynamic_complexity, mfcc_zero_mean
  rhythm:   mbid, submission_offset, bpm, ..., danceability, onset_rate
  tonal:    mbid, submission_offset, key_key, key_scale, tuning_frequency, ...
"""

import subprocess
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

LINKAGE_CSV = PROCESSED_DIR / "ma_mb_linkage.csv"
OUTPUT_CSV = PROCESSED_DIR / "ma_ab_features.csv"

# ---------------------------------------------------------------------------
# MusicBrainz dump configuration
# ---------------------------------------------------------------------------
MB_DUMP_URL = (
    "https://data.metabrainz.org/pub/musicbrainz/data/fullexport/"
    "20260307-002311/mbdump.tar.bz2"
)
MB_TARBALL = MB_DIR / "mbdump.tar.bz2"

# Tables we need beyond what step 01 already extracted
# Step 01 extracted: artist, url, l_artist_url
# We additionally need: artist_credit_name, recording
MB_EXTRA_TABLES = ["artist_credit_name", "recording"]

# ---------------------------------------------------------------------------
# AcousticBrainz feature dump configuration
# ---------------------------------------------------------------------------
# The AB download page (https://acousticbrainz.org/download) provides feature
# CSVs as a ~3GB compressed archive. The exact filenames follow the pattern:
#   acousticbrainz-lowlevel-features-YYYYMMDD-{type}.csv
# where type is one of: lowlevel, rhythm, tonal.
#
# Since the site may be flaky (project shut down 2022), we provide multiple
# possible download strategies:
#   1. Direct download from acousticbrainz.org/download (official, may be down)
#   2. Manual download instructions for the user
#
# The feature CSVs are ~3GB compressed / ~10GB uncompressed total.
# Each has ~29.5M rows with headers.
#
# Expected filenames after extraction (we normalise to these names):
AB_FEATURE_FILES = {
    "lowlevel": "acousticbrainz_lowlevel.csv",
    "rhythm": "acousticbrainz_rhythm.csv",
    "tonal": "acousticbrainz_tonal.csv",
}

# Columns we want from each AB feature file
AB_LOWLEVEL_COLS = ["mbid", "average_loudness", "dynamic_complexity", "mfcc_zero_mean"]
AB_RHYTHM_COLS = ["mbid", "bpm", "danceability", "onset_rate"]
AB_TONAL_COLS = ["mbid", "key_key", "key_scale", "tuning_frequency"]


# ===================================================================
# STEP 1: Ensure required MusicBrainz tables are present
# ===================================================================

def ensure_mb_tables():
    """Download and extract additional MB tables if not already present."""
    missing = [t for t in MB_EXTRA_TABLES if not (MB_DIR / t).exists()]
    if not missing:
        print("[MB] All required tables already present.")
        for t in MB_EXTRA_TABLES:
            path = MB_DIR / t
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {t}: {size_mb:.0f} MB")
        return

    print(f"[MB] Missing tables: {missing}")
    print(f"[MB] Need to download mbdump.tar.bz2 (~6.6 GB) and extract: {missing}")

    # Download tarball if not present
    if not MB_TARBALL.exists():
        print(f"[MB] Downloading {MB_DUMP_URL} ...")
        print(f"[MB] This is ~6.6 GB — it will take a while.")
        try:
            subprocess.run(
                ["curl", "-L", "--progress-bar", "-o", str(MB_TARBALL), MB_DUMP_URL],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"[MB] ERROR: curl failed with exit code {e.returncode}")
            print(f"[MB] You can manually download from: {MB_DUMP_URL}")
            print(f"[MB] Place it at: {MB_TARBALL}")
            sys.exit(1)
        except FileNotFoundError:
            print("[MB] ERROR: curl not found. Install curl or manually download.")
            print(f"[MB] URL: {MB_DUMP_URL}")
            print(f"[MB] Save to: {MB_TARBALL}")
            sys.exit(1)
    else:
        print(f"[MB] Tarball already present: {MB_TARBALL}")

    # Extract only the tables we need
    print(f"[MB] Extracting {missing} from tarball ...")
    members_to_extract = [f"mbdump/{t}" for t in missing]

    try:
        subprocess.run(
            [
                "tar", "-xjf", str(MB_TARBALL),
                "-C", str(MB_DIR),
                "--strip-components=1",
            ] + members_to_extract,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[MB] ERROR: tar extraction failed: {e}")
        sys.exit(1)

    # Verify extraction
    for t in missing:
        path = MB_DIR / t
        if not path.exists():
            print(f"[MB] ERROR: Failed to extract {t}")
            sys.exit(1)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"[MB] Extracted {t}: {size_mb:.0f} MB")

    # Optionally clean up tarball to save space
    print(f"[MB] Removing tarball to save disk space ...")
    MB_TARBALL.unlink(missing_ok=True)
    print(f"[MB] Done.")


# ===================================================================
# STEP 2: Ensure AcousticBrainz feature CSVs are present
# ===================================================================

def _find_ab_csv(feature_type: str) -> Path | None:
    """Find an AB feature CSV, trying various naming conventions."""
    # Our normalised name
    norm_name = AB_FEATURE_FILES[feature_type]
    if (AB_DIR / norm_name).exists():
        return AB_DIR / norm_name

    # Official dump naming: acousticbrainz-lowlevel-features-YYYYMMDD-{type}.csv
    for p in AB_DIR.glob(f"*-{feature_type}.csv"):
        return p
    for p in AB_DIR.glob(f"*{feature_type}*.csv"):
        return p

    return None


def ensure_ab_features():
    """Check that AcousticBrainz feature CSVs exist, or print download instructions."""
    AB_DIR.mkdir(parents=True, exist_ok=True)

    all_present = True
    for ftype, fname in AB_FEATURE_FILES.items():
        path = _find_ab_csv(ftype)
        if path is None:
            all_present = False
            print(f"[AB] MISSING: {ftype} feature CSV (expected in {AB_DIR}/)")
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"[AB] Found {ftype}: {path.name} ({size_mb:.0f} MB)")

    if all_present:
        print("[AB] All feature CSVs present.")
        return True

    print()
    print("=" * 70)
    print("ACOUSTICBRAINZ FEATURE DUMPS NOT FOUND")
    print("=" * 70)
    print()
    print("AcousticBrainz shut down in 2022 but made data dumps available.")
    print("You need the 3 feature CSV files (~3 GB compressed total).")
    print()
    print("OPTION 1 — Download from official site (may still work):")
    print("  Visit: https://acousticbrainz.org/download")
    print("  Look for 'Feature dumps' section (~3 GB)")
    print("  Download and extract to: data/acousticbrainz/")
    print()
    print("OPTION 2 — If the site is down, try the MetaBrainz FTP:")
    print("  ftp://ftp.acousticbrainz.org/pub/acousticbrainz/")
    print("  Or: https://data.metabrainz.org/pub/acousticbrainz/")
    print()
    print("OPTION 3 — Check archive.org for mirrors:")
    print("  Search: https://archive.org/search?query=acousticbrainz")
    print()
    print("After downloading, rename or symlink the CSVs to:")
    for ftype, fname in AB_FEATURE_FILES.items():
        print(f"  {AB_DIR / fname}")
    print()
    print("The CSVs should have these columns (with headers):")
    print("  lowlevel: mbid, submission_offset, average_loudness, dynamic_complexity, mfcc_zero_mean")
    print("  rhythm:   mbid, submission_offset, bpm, ..., danceability, onset_rate")
    print("  tonal:    mbid, submission_offset, key_key, key_scale, tuning_frequency, ...")
    print()
    print("Re-run this script after placing the files.")
    print("=" * 70)
    return False


# ===================================================================
# STEP 3: Load step-01 linkage (MA band ID → MB Artist MBID)
# ===================================================================

def load_linkage() -> pd.DataFrame:
    """Load the step-01 linkage CSV."""
    print(f"\n[LINKAGE] Loading {LINKAGE_CSV} ...")
    df = pd.read_csv(LINKAGE_CSV, dtype={"ma_band_id": "int64", "mbid": "str"})
    # Deduplicate: keep one MBID per ma_band_id (step 01 may have dupes)
    df = df.drop_duplicates(subset=["ma_band_id"], keep="first")
    print(f"  Unique MA bands with MBID: {len(df):,}")
    return df[["ma_band_id", "ma_name", "mbid"]]


# ===================================================================
# STEP 4: Build artist MBID → recording MBIDs mapping via MB tables
# ===================================================================

def load_artist_id_map() -> pd.DataFrame:
    """Load MB artist table: mbid (gid) → artist_id (internal int)."""
    path = MB_DIR / "artist"
    print(f"\n[MB:artist] Loading {path} ...")
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["artist_id", "mbid"],
        dtype={"artist_id": "int64", "mbid": "str"},
        na_filter=False,
        quoting=3,  # QUOTE_NONE
    )
    print(f"  Total artists: {len(df):,}")
    return df


def load_artist_credit_name() -> pd.DataFrame:
    """Load MB artist_credit_name table.

    Schema: artist_credit(0), position(1), artist(2), name(3), join_phrase(4)
    We need: artist(2) → artist_credit(0)
    """
    path = MB_DIR / "artist_credit_name"
    print(f"\n[MB:artist_credit_name] Loading {path} ...")
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 2],
        names=["artist_credit_id", "artist_id"],
        dtype={"artist_credit_id": "int64", "artist_id": "int64"},
        na_filter=False,
        quoting=3,
    )
    print(f"  Total artist-credit-name rows: {len(df):,}")
    return df


def load_recording() -> pd.DataFrame:
    """Load MB recording table.

    Schema: id(0), gid(1), name(2), artist_credit(3), length(4), ...
    We need: artist_credit(3) → gid(1) (recording MBID)
    """
    path = MB_DIR / "recording"
    print(f"\n[MB:recording] Loading {path} ...")
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[1, 3],
        names=["recording_mbid", "artist_credit_id"],
        dtype={"recording_mbid": "str", "artist_credit_id": "int64"},
        na_filter=False,
        quoting=3,
    )
    print(f"  Total recordings: {len(df):,}")
    return df


def build_artist_to_recordings(linkage: pd.DataFrame) -> pd.DataFrame:
    """Build mapping: ma_band_id → set of recording MBIDs.

    Join chain:
      linkage.mbid → artist.gid → artist.id
        → artist_credit_name.artist → artist_credit_name.artist_credit
        → recording.artist_credit → recording.gid
    """
    # 1. Load artist table, map MBID → internal artist ID
    artist_df = load_artist_id_map()

    # 2. Filter to only our 46k artist MBIDs for memory efficiency
    our_mbids = set(linkage["mbid"].unique())
    artist_df = artist_df[artist_df["mbid"].isin(our_mbids)].copy()
    print(f"  Artists matching our linkage: {len(artist_df):,}")

    # 3. Merge linkage with artist to get internal IDs
    #    linkage has: ma_band_id, ma_name, mbid
    #    artist_df has: artist_id, mbid
    band_artist = linkage.merge(artist_df, on="mbid", how="inner")
    print(f"  Bands with artist_id: {len(band_artist):,}")

    # 4. Load artist_credit_name: artist_id → artist_credit_id
    acn = load_artist_credit_name()

    # Filter to only our artist IDs
    our_artist_ids = set(band_artist["artist_id"].unique())
    acn = acn[acn["artist_id"].isin(our_artist_ids)].copy()
    print(f"  Artist credit names for our artists: {len(acn):,}")

    # 5. Join: band_artist → acn on artist_id
    band_credit = band_artist[["ma_band_id", "artist_id"]].merge(
        acn, on="artist_id", how="inner"
    )
    print(f"  Band-credit pairs: {len(band_credit):,}")

    # 6. Load recording: artist_credit_id → recording MBID
    rec_df = load_recording()

    # Filter to only our artist_credit_ids
    our_credit_ids = set(band_credit["artist_credit_id"].unique())
    rec_df = rec_df[rec_df["artist_credit_id"].isin(our_credit_ids)].copy()
    print(f"  Recordings for our credits: {len(rec_df):,}")

    # 7. Join: band_credit → recording on artist_credit_id
    band_rec = band_credit[["ma_band_id", "artist_credit_id"]].merge(
        rec_df, on="artist_credit_id", how="inner"
    )
    print(f"  Band-recording pairs: {len(band_rec):,}")

    # 8. Deduplicate (same recording can appear via multiple credits)
    band_rec = band_rec[["ma_band_id", "recording_mbid"]].drop_duplicates()
    print(f"  Unique band-recording pairs: {len(band_rec):,}")

    n_bands = band_rec["ma_band_id"].nunique()
    n_recs = band_rec["recording_mbid"].nunique()
    print(f"  Unique bands with recordings: {n_bands:,}")
    print(f"  Unique recording MBIDs: {n_recs:,}")

    return band_rec


# ===================================================================
# STEP 5: Load and merge AcousticBrainz features
# ===================================================================

def load_ab_features(recording_mbids: set[str]) -> pd.DataFrame:
    """Load AB feature CSVs and filter to our recording MBIDs.

    Returns a single DataFrame with one row per recording MBID,
    containing all audio features.
    """
    print("\n[AB] Loading AcousticBrainz feature CSVs ...")

    # --- Lowlevel ---
    ll_path = _find_ab_csv("lowlevel")
    print(f"\n  Loading lowlevel: {ll_path} ...")
    ll_df = _load_ab_csv_filtered(ll_path, recording_mbids, AB_LOWLEVEL_COLS)
    print(f"    Matched recordings: {len(ll_df):,}")

    # --- Rhythm ---
    rh_path = _find_ab_csv("rhythm")
    print(f"\n  Loading rhythm: {rh_path} ...")
    rh_df = _load_ab_csv_filtered(rh_path, recording_mbids, AB_RHYTHM_COLS)
    print(f"    Matched recordings: {len(rh_df):,}")

    # --- Tonal ---
    tn_path = _find_ab_csv("tonal")
    print(f"\n  Loading tonal: {tn_path} ...")
    tn_df = _load_ab_csv_filtered(tn_path, recording_mbids, AB_TONAL_COLS)
    print(f"    Matched recordings: {len(tn_df):,}")

    # Merge all three on mbid (inner join — only keep recordings with all 3)
    print("\n  Merging feature tables ...")
    features = ll_df.merge(rh_df, on="mbid", how="inner")
    features = features.merge(tn_df, on="mbid", how="inner")
    print(f"  Recordings with all 3 feature types: {len(features):,}")

    return features


def _load_ab_csv_filtered(
    path: Path, recording_mbids: set[str], keep_cols: list[str]
) -> pd.DataFrame:
    """Load an AB feature CSV in chunks, filtering to our recording MBIDs.

    AB CSVs have ~29.5M rows. Reading the full file into memory is expensive,
    so we use chunked reading and filter on-the-fly.
    """
    chunks = []
    chunk_size = 500_000

    # Read the header first to discover column names
    header_df = pd.read_csv(path, nrows=0)
    all_cols = list(header_df.columns)

    # Map our desired columns to actual column names (handle naming variations)
    col_map = {}
    for desired in keep_cols:
        # Try exact match first
        if desired in all_cols:
            col_map[desired] = desired
        else:
            # Try case-insensitive match
            for actual in all_cols:
                if actual.lower() == desired.lower():
                    col_map[desired] = actual
                    break

    if "mbid" not in col_map:
        # mbid might be called 'recording_mbid' or similar
        for c in all_cols:
            if "mbid" in c.lower() or "recording" in c.lower():
                col_map["mbid"] = c
                break

    if "mbid" not in col_map:
        print(f"    WARNING: Could not find mbid column. Available: {all_cols}")
        return pd.DataFrame(columns=keep_cols)

    # Determine which columns to actually read
    # Always include submission_offset for deduplication (multiple submissions per recording)
    usecols = [col_map.get(c, c) for c in keep_cols if c in col_map]
    if "submission_offset" in all_cols and "submission_offset" not in usecols:
        usecols.append("submission_offset")
    mbid_col = col_map["mbid"]

    reader = pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in tqdm(reader, desc=f"    Reading {path.name}", unit="chunk"):
        # Normalise mbid column name
        if mbid_col != "mbid":
            chunk = chunk.rename(columns={mbid_col: "mbid"})

        # Filter to our recordings
        mask = chunk["mbid"].isin(recording_mbids)
        filtered = chunk[mask]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame(columns=keep_cols)

    result = pd.concat(chunks, ignore_index=True)

    # If there are multiple submissions per recording, keep the first (offset=0)
    if "submission_offset" in result.columns:
        result = result.sort_values("submission_offset").drop_duplicates(
            subset=["mbid"], keep="first"
        )
        result = result.drop(columns=["submission_offset"])

    # Rename columns back to our desired names
    reverse_map = {v: k for k, v in col_map.items() if v != k}
    if reverse_map:
        result = result.rename(columns=reverse_map)

    # Keep only the columns we want
    final_cols = [c for c in keep_cols if c in result.columns]
    return result[final_cols]


# ===================================================================
# STEP 6: Aggregate features to band level and output
# ===================================================================

def aggregate_band_features(
    band_rec: pd.DataFrame, features: pd.DataFrame, linkage: pd.DataFrame
) -> pd.DataFrame:
    """Join band-recording mapping with AB features, aggregate per band.

    For numeric features: compute mean across all recordings for the band.
    For categorical features (key_key, key_scale): compute mode (most common).

    Output columns:
      ma_band_id, ma_name, n_recordings, n_ab_matched,
      avg_loudness, dynamic_complexity, mfcc_zero_mean,
      bpm, danceability, onset_rate,
      key_key, key_scale, tuning_frequency
    """
    print("\n[AGGREGATE] Joining band-recordings with AB features ...")

    # Join: band_rec (ma_band_id, recording_mbid) + features (mbid, ...)
    merged = band_rec.merge(
        features, left_on="recording_mbid", right_on="mbid", how="inner"
    )
    print(f"  Band-recording-feature rows: {len(merged):,}")

    bands_with_features = merged["ma_band_id"].nunique()
    print(f"  Bands with at least 1 AB match: {bands_with_features:,}")

    # Count recordings per band (total and AB-matched)
    rec_counts = band_rec.groupby("ma_band_id").size().rename("n_recordings")
    ab_counts = merged.groupby("ma_band_id").size().rename("n_ab_matched")

    # Numeric feature columns to average
    numeric_cols = [
        "average_loudness", "dynamic_complexity", "mfcc_zero_mean",
        "bpm", "danceability", "onset_rate", "tuning_frequency",
    ]
    # Only include columns that actually exist
    numeric_cols = [c for c in numeric_cols if c in merged.columns]

    # Categorical columns — take mode
    cat_cols = [c for c in ["key_key", "key_scale"] if c in merged.columns]

    # Coerce numeric columns
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Aggregate numeric: mean per band
    print("  Computing band-level means for numeric features ...")
    if numeric_cols:
        band_numeric = merged.groupby("ma_band_id")[numeric_cols].mean()
    else:
        band_numeric = pd.DataFrame(index=merged["ma_band_id"].unique())

    # Aggregate categorical: mode per band
    print("  Computing band-level modes for categorical features ...")
    cat_aggs = {}
    for col in cat_cols:
        # mode() returns a Series; take first element
        mode_series = merged.groupby("ma_band_id")[col].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
        )
        cat_aggs[col] = mode_series

    if cat_aggs:
        band_cat = pd.DataFrame(cat_aggs)
    else:
        band_cat = pd.DataFrame(index=merged["ma_band_id"].unique())

    # Combine everything
    result = (
        rec_counts.to_frame()
        .join(ab_counts, how="left")
        .join(band_numeric, how="left")
        .join(band_cat, how="left")
    )
    result["n_ab_matched"] = result["n_ab_matched"].fillna(0).astype(int)
    result = result.reset_index()

    # Add band name from linkage
    name_map = linkage[["ma_band_id", "ma_name"]].drop_duplicates(subset=["ma_band_id"])
    result = result.merge(name_map, on="ma_band_id", how="left")

    # Reorder columns: ma_band_id, ma_name, counts, then features
    lead_cols = ["ma_band_id", "ma_name", "n_recordings", "n_ab_matched"]
    feature_cols = [c for c in result.columns if c not in lead_cols]
    result = result[lead_cols + sorted(feature_cols)]

    # Filter to only bands that have AB data
    result = result[result["n_ab_matched"] > 0].copy()
    result = result.sort_values("ma_band_id")

    return result


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 70)
    print("heavyML Pipeline Step 02: AcousticBrainz Feature Extraction")
    print("=" * 70)

    # --- 1. Ensure MB tables ---
    print("\n--- Step 1: MusicBrainz tables ---")
    ensure_mb_tables()

    # --- 2. Ensure AB feature CSVs ---
    print("\n--- Step 2: AcousticBrainz feature CSVs ---")
    ab_ready = ensure_ab_features()
    if not ab_ready:
        print("\n[ABORT] Cannot proceed without AcousticBrainz feature CSVs.")
        print("         Follow the instructions above and re-run.")
        sys.exit(1)

    # --- 3. Load step-01 linkage ---
    print("\n--- Step 3: Load MA → MB linkage ---")
    linkage = load_linkage()

    # --- 4. Build artist → recording mapping ---
    print("\n--- Step 4: Build artist → recording MBIDs ---")
    band_rec = build_artist_to_recordings(linkage)

    if len(band_rec) == 0:
        print("\n[ERROR] No band-recording pairs found. Check MB table integrity.")
        sys.exit(1)

    # --- 5. Load AB features ---
    print("\n--- Step 5: Load AcousticBrainz features ---")
    recording_mbids = set(band_rec["recording_mbid"].unique())
    print(f"  Looking up {len(recording_mbids):,} recording MBIDs in AB data ...")
    features = load_ab_features(recording_mbids)

    if len(features) == 0:
        print("\n[ERROR] No AB features matched any recording MBIDs.")
        print("  This could mean:")
        print("  1. The AB CSVs are not the correct feature dumps")
        print("  2. The recording MBIDs don't overlap with AB coverage")
        sys.exit(1)

    # --- 6. Aggregate to band level ---
    print("\n--- Step 6: Aggregate to band-level features ---")
    result = aggregate_band_features(band_rec, features, linkage)

    # --- 7. Save output ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OUTPUT] Saved: {OUTPUT_CSV}")
    print(f"  Rows: {len(result):,}")
    print(f"  Columns: {list(result.columns)}")

    # --- 8. Coverage report ---
    print()
    print("=" * 70)
    print("COVERAGE REPORT")
    print("=" * 70)
    total_linkage = linkage["ma_band_id"].nunique()
    total_with_rec = band_rec["ma_band_id"].nunique()
    total_with_ab = len(result)

    print(f"  MA bands with MBID (step 01):      {total_linkage:,}")
    print(f"  → with MB recordings:              {total_with_rec:,}  "
          f"({total_with_rec/total_linkage*100:.1f}%)")
    print(f"  → with AB audio features:          {total_with_ab:,}  "
          f"({total_with_ab/total_linkage*100:.1f}%)")
    print()

    # Recording-level stats
    total_recs = band_rec["recording_mbid"].nunique()
    matched_recs = len(features)
    print(f"  Unique recording MBIDs:            {total_recs:,}")
    print(f"  → matched in AcousticBrainz:       {matched_recs:,}  "
          f"({matched_recs/total_recs*100:.1f}%)")
    print()

    # Feature stats
    numeric_cols = [c for c in result.columns if c not in
                    ["ma_band_id", "ma_name", "n_recordings", "n_ab_matched",
                     "key_key", "key_scale"]]
    if numeric_cols:
        print("  Feature distributions (band-level averages):")
        for col in numeric_cols:
            if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
                vals = result[col].dropna()
                if len(vals) > 0:
                    print(f"    {col:25s}  mean={vals.mean():.3f}  "
                          f"std={vals.std():.3f}  "
                          f"min={vals.min():.3f}  max={vals.max():.3f}")

    if "key_key" in result.columns:
        print()
        print("  Most common keys:")
        key_counts = result["key_key"].value_counts().head(5)
        for key, count in key_counts.items():
            print(f"    {key}: {count:,} bands")

    # Verdict
    print()
    if total_with_ab >= 20_000:
        print("VERDICT: Excellent coverage (>20k bands with audio features).")
        print("  Proceed to step 03 (similarity labels) and step 04 (feature matrix).")
    elif total_with_ab >= 10_000:
        print("VERDICT: Good coverage (10k-20k bands). Sufficient for model training.")
    elif total_with_ab >= 5_000:
        print("VERDICT: Marginal coverage (5k-10k bands). Model quality may be limited.")
    else:
        print("VERDICT: Low coverage (<5k bands). Consider alternative feature sources.")


if __name__ == "__main__":
    main()
