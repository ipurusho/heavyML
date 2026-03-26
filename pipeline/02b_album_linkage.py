#!/usr/bin/env python3
"""02b_album_linkage.py — Link MA albums to MusicBrainz release groups.

Builds the album-level linkage chain:
  MA Band ID → (step 01 linkage) → MB Artist MBID
    → artist table (gid → id)
    → artist_credit_name table (artist = artist.id → artist_credit)
    → release_group table (artist_credit → id, name)
    → release table (release_group → id)
    → medium table (release → id)
    → track table (medium → recording)
    → recording table (id → gid = recording MBID)

Groups recordings by release_group (= album concept) and outputs
a mapping from MA bands to MB release groups with recording counts.

MusicBrainz TSV schemas (headerless, tab-separated, column indices):
  artist:              id(0), gid(1), name(2), sort_name(3), ... (19 cols)
  artist_credit_name:  artist_credit(0), position(1), artist(2), name(3), join_phrase(4)
  release_group:       id(0), gid(1), name(2), artist_credit(3), type(4), ...
  release:             id(0), gid(1), name(2), artist_credit(3), release_group(4), status(5), ...
  medium:              id(0), release(1), position(2), format(3), name(4), ...
  track:               id(0), gid(1), recording(2), medium(3), position(4), number(5), name(6), artist_credit(7), length(8), ...
  recording:           id(0), gid(1), name(2), artist_credit(3), length(4), ... (9 cols)

Usage:
    python pipeline/02b_album_linkage.py

Input:
    data/processed/ma_mb_linkage.csv          — step 01 output
    data/musicbrainz/{artist,artist_credit_name,release_group,release,medium,track,recording}

Output:
    data/processed/ma_album_mb_linkage.csv    — MA band → MB release group linkage
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
PROCESSED_DIR = DATA_DIR / "processed"

LINKAGE_CSV = PROCESSED_DIR / "ma_mb_linkage.csv"
DISCOGRAPHY_CSV = DATA_DIR / "raw" / "all_bands_discography.csv"
OUTPUT_CSV = PROCESSED_DIR / "ma_album_mb_linkage.csv"

# MusicBrainz dump configuration (same URL as 02_ab_features.py)
MB_DUMP_URL = (
    "https://data.metabrainz.org/pub/musicbrainz/data/fullexport/"
    "20260307-002311/mbdump.tar.bz2"
)
MB_TARBALL = MB_DIR / "mbdump.tar.bz2"

# Tables needed for album-level linkage (beyond step 01)
MB_ALBUM_TABLES = ["artist_credit_name", "recording", "release_group", "release", "medium", "track"]


# ===================================================================
# STEP 1: Ensure required MusicBrainz tables are present
# ===================================================================

def ensure_mb_tables():
    """Download and extract album-related MB tables if not already present."""
    missing = [t for t in MB_ALBUM_TABLES if not (MB_DIR / t).exists()]
    if not missing:
        print("[MB] All required album tables already present.")
        for t in MB_ALBUM_TABLES:
            path = MB_DIR / t
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {t}: {size_mb:.0f} MB")
        return

    print(f"[MB] Missing tables: {missing}")
    print(f"[MB] Need to extract from mbdump.tar.bz2: {missing}")

    # Download tarball if not present
    if not MB_TARBALL.exists():
        print(f"[MB] Downloading {MB_DUMP_URL} ...")
        print("[MB] This is ~6.6 GB -- it will take a while.")
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
            sys.exit(1)
    else:
        print(f"[MB] Tarball already present: {MB_TARBALL}")

    # Extract only the missing tables
    print(f"[MB] Extracting {missing} from tarball ...")
    members = [f"mbdump/{t}" for t in missing]

    try:
        subprocess.run(
            ["tar", "-xjf", str(MB_TARBALL), "-C", str(MB_DIR), "--strip-components=1"]
            + members,
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

    # Clean up tarball to save space
    print("[MB] Removing tarball to save disk space ...")
    MB_TARBALL.unlink(missing_ok=True)
    print("[MB] Done.")


# ===================================================================
# STEP 2: Load step-01 linkage (MA band ID -> MB Artist MBID)
# ===================================================================

def load_linkage() -> pd.DataFrame:
    """Load the step-01 linkage CSV."""
    print(f"\n[LINKAGE] Loading {LINKAGE_CSV} ...")
    if not LINKAGE_CSV.exists():
        print(f"  ERROR: {LINKAGE_CSV} not found. Run step 01 first.")
        sys.exit(1)

    df = pd.read_csv(LINKAGE_CSV, dtype={"ma_band_id": "int64", "mbid": "str"})
    df = df.drop_duplicates(subset=["ma_band_id"], keep="first")
    print(f"  Unique MA bands with MBID: {len(df):,}")
    return df[["ma_band_id", "ma_name", "mbid"]]


# ===================================================================
# STEP 3: Load MusicBrainz tables
# ===================================================================

def load_artist_id_map() -> pd.DataFrame:
    """Load MB artist table: mbid (gid) -> artist_id (internal int)."""
    path = MB_DIR / "artist"
    print(f"\n[MB:artist] Loading {path} ...")
    df = pd.read_csv(
        path, sep="\t", header=None, usecols=[0, 1],
        names=["artist_id", "mbid"],
        dtype={"artist_id": "int64", "mbid": "str"},
        na_filter=False, quoting=3,
    )
    print(f"  Total artists: {len(df):,}")
    return df


def load_artist_credit_name() -> pd.DataFrame:
    """Load MB artist_credit_name: artist(2) -> artist_credit(0)."""
    path = MB_DIR / "artist_credit_name"
    print(f"\n[MB:artist_credit_name] Loading {path} ...")
    df = pd.read_csv(
        path, sep="\t", header=None, usecols=[0, 2],
        names=["artist_credit_id", "artist_id"],
        dtype={"artist_credit_id": "int64", "artist_id": "int64"},
        na_filter=False, quoting=3,
    )
    print(f"  Total rows: {len(df):,}")
    return df


def load_release_group() -> pd.DataFrame:
    """Load MB release_group: id(0), gid(1), name(2), artist_credit(3), type(4).

    We need: artist_credit(3) -> id(0), gid(1), name(2)
    """
    path = MB_DIR / "release_group"
    print(f"\n[MB:release_group] Loading {path} ...")
    df = pd.read_csv(
        path, sep="\t", header=None, usecols=[0, 1, 2, 3],
        names=["rg_id", "rg_gid", "rg_name", "artist_credit_id"],
        dtype={"rg_id": "int64", "rg_gid": "str", "rg_name": "str", "artist_credit_id": "int64"},
        na_filter=False, quoting=3,
    )
    print(f"  Total release groups: {len(df):,}")
    return df


def load_release() -> pd.DataFrame:
    """Load MB release: id(0), release_group(4).

    We need: release_group(4) -> id(0)
    """
    path = MB_DIR / "release"
    print(f"\n[MB:release] Loading {path} ...")
    df = pd.read_csv(
        path, sep="\t", header=None, usecols=[0, 4],
        names=["release_id", "rg_id"],
        dtype={"release_id": "int64", "rg_id": "int64"},
        na_filter=False, quoting=3,
    )
    print(f"  Total releases: {len(df):,}")
    return df


def load_medium() -> pd.DataFrame:
    """Load MB medium: id(0), release(1).

    We need: release(1) -> id(0)
    """
    path = MB_DIR / "medium"
    print(f"\n[MB:medium] Loading {path} ...")
    df = pd.read_csv(
        path, sep="\t", header=None, usecols=[0, 1],
        names=["medium_id", "release_id"],
        dtype={"medium_id": "int64", "release_id": "int64"},
        na_filter=False, quoting=3,
    )
    print(f"  Total mediums: {len(df):,}")
    return df


def load_track() -> pd.DataFrame:
    """Load MB track: id(0), recording(2), medium(3).

    We need: medium(3) -> recording(2)
    """
    path = MB_DIR / "track"
    print(f"\n[MB:track] Loading {path} ...")
    df = pd.read_csv(
        path, sep="\t", header=None, usecols=[2, 3],
        names=["recording_id", "medium_id"],
        dtype={"recording_id": "int64", "medium_id": "int64"},
        na_filter=False, quoting=3,
    )
    print(f"  Total tracks: {len(df):,}")
    return df


def load_recording() -> pd.DataFrame:
    """Load MB recording: id(0), gid(1), name(2), artist_credit(3).

    We need: id(0) -> gid(1) (recording MBID)
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
    return df


# ===================================================================
# STEP 4: Build the album-level linkage chain
# ===================================================================

def build_album_linkage(linkage: pd.DataFrame) -> pd.DataFrame:
    """Build the full chain: MA Band -> MB Artist -> release_group -> recordings.

    Join chain:
      linkage.mbid -> artist.gid -> artist.id
        -> artist_credit_name.artist -> artist_credit_name.artist_credit
        -> release_group.artist_credit -> release_group.(id, gid, name)
        -> release.release_group -> release.id
        -> medium.release -> medium.id
        -> track.medium -> track.recording
        -> recording.id -> recording.gid (recording MBID)

    Groups by (ma_band_id, release_group) and counts recordings.
    """
    # 1. Load artist table, filter to our MBIDs
    artist_df = load_artist_id_map()
    our_mbids = set(linkage["mbid"].unique())
    artist_df = artist_df[artist_df["mbid"].isin(our_mbids)].copy()
    print(f"  Artists matching our linkage: {len(artist_df):,}")

    # 2. Merge linkage with artist to get internal IDs
    band_artist = linkage.merge(artist_df, on="mbid", how="inner")
    print(f"  Bands with artist_id: {len(band_artist):,}")

    # 3. Load artist_credit_name, filter to our artist IDs
    acn = load_artist_credit_name()
    our_artist_ids = set(band_artist["artist_id"].unique())
    acn = acn[acn["artist_id"].isin(our_artist_ids)].copy()
    print(f"  Artist credit names for our artists: {len(acn):,}")

    # 4. Join band_artist -> acn on artist_id to get artist_credit_ids
    band_credit = band_artist[["ma_band_id", "artist_id"]].merge(
        acn, on="artist_id", how="inner"
    )
    our_credit_ids = set(band_credit["artist_credit_id"].unique())
    print(f"  Band-credit pairs: {len(band_credit):,}")
    print(f"  Unique artist_credit_ids: {len(our_credit_ids):,}")

    # 5. Load release_group, filter to our artist_credit_ids
    rg_df = load_release_group()
    rg_df = rg_df[rg_df["artist_credit_id"].isin(our_credit_ids)].copy()
    print(f"  Release groups for our credits: {len(rg_df):,}")

    # 6. Join band_credit -> release_group on artist_credit_id
    band_rg = band_credit[["ma_band_id", "artist_credit_id"]].merge(
        rg_df[["rg_id", "rg_gid", "rg_name", "artist_credit_id"]],
        on="artist_credit_id", how="inner"
    )
    band_rg = band_rg[["ma_band_id", "rg_id", "rg_gid", "rg_name"]].drop_duplicates()
    print(f"  Band-release_group pairs: {len(band_rg):,}")

    # 7. Load release, filter to our release_group IDs
    release_df = load_release()
    our_rg_ids = set(band_rg["rg_id"].unique())
    release_df = release_df[release_df["rg_id"].isin(our_rg_ids)].copy()
    print(f"  Releases for our release groups: {len(release_df):,}")

    # 8. Load medium, filter to our release IDs
    medium_df = load_medium()
    our_release_ids = set(release_df["release_id"].unique())
    medium_df = medium_df[medium_df["release_id"].isin(our_release_ids)].copy()
    print(f"  Mediums for our releases: {len(medium_df):,}")

    # 9. Load track, filter to our medium IDs
    track_df = load_track()
    our_medium_ids = set(medium_df["medium_id"].unique())
    track_df = track_df[track_df["medium_id"].isin(our_medium_ids)].copy()
    print(f"  Tracks for our mediums: {len(track_df):,}")

    # 10. Build release_group -> recording chain via release -> medium -> track
    # release_group -> release
    rg_release = release_df[["release_id", "rg_id"]]
    # release -> medium
    release_medium = medium_df[["medium_id", "release_id"]]
    # medium -> track (recording_id)
    medium_track = track_df[["recording_id", "medium_id"]]

    # Chain: rg -> release -> medium -> track
    print("\n[JOIN] Building release_group -> recording chain ...")
    rg_medium = rg_release.merge(release_medium, on="release_id", how="inner")
    print(f"  rg-medium pairs: {len(rg_medium):,}")

    rg_track = rg_medium.merge(medium_track, on="medium_id", how="inner")
    print(f"  rg-track pairs: {len(rg_track):,}")

    # Deduplicate: same recording can appear on multiple releases of the same rg
    rg_recording = rg_track[["rg_id", "recording_id"]].drop_duplicates()
    print(f"  Unique rg-recording pairs: {len(rg_recording):,}")

    # 11. Count recordings per release_group
    rg_rec_counts = rg_recording.groupby("rg_id").size().rename("n_recordings").reset_index()
    print(f"  Release groups with recordings: {len(rg_rec_counts):,}")

    # 12. Merge back: band_rg -> rg_rec_counts
    result = band_rg.merge(rg_rec_counts, on="rg_id", how="inner")
    print(f"  Band-rg pairs with recordings: {len(result):,}")

    # 13. Also resolve recording_id -> recording MBID for downstream use
    #     (We store the rg-level linkage; recording MBIDs will be resolved in 02c)

    # 14. Rename columns for output
    result = result.rename(columns={
        "rg_gid": "mb_release_group_id",
        "rg_name": "mb_release_group_name",
    })

    # Add ma_name from linkage
    name_map = linkage[["ma_band_id", "ma_name"]].drop_duplicates(subset=["ma_band_id"])
    result = result.merge(name_map, on="ma_band_id", how="left")

    # Select and order output columns
    result = result[[
        "ma_band_id", "ma_name", "mb_release_group_id",
        "mb_release_group_name", "n_recordings",
    ]].sort_values(["ma_band_id", "mb_release_group_name"])

    return result, rg_recording


# ===================================================================
# STEP 5: Optional fuzzy matching with MA discography
# ===================================================================

def load_ma_discography() -> pd.DataFrame | None:
    """Load the MA discography CSV for cross-reference stats."""
    if not DISCOGRAPHY_CSV.exists():
        print(f"\n[DISCO] {DISCOGRAPHY_CSV} not found, skipping discography cross-reference.")
        return None

    print(f"\n[DISCO] Loading {DISCOGRAPHY_CSV} ...")
    df = pd.read_csv(DISCOGRAPHY_CSV, low_memory=False)
    print(f"  Total releases: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    return df


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 70)
    print("heavyML Pipeline Step 02b: Album-Level MusicBrainz Linkage")
    print("=" * 70)

    # --- 1. Ensure MB tables ---
    print("\n--- Step 1: Ensure MusicBrainz album tables ---")
    ensure_mb_tables()

    # --- 2. Load step-01 linkage ---
    print("\n--- Step 2: Load MA -> MB artist linkage ---")
    linkage = load_linkage()

    # --- 3. Build album linkage chain ---
    print("\n--- Step 3: Build album-level linkage chain ---")
    result, rg_recording = build_album_linkage(linkage)

    if len(result) == 0:
        print("\n[ERROR] No album linkages found. Check MB table integrity.")
        sys.exit(1)

    # --- 4. Save output ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OUTPUT] Saved: {OUTPUT_CSV}")
    print(f"  Rows: {len(result):,}")
    print(f"  Columns: {list(result.columns)}")

    # --- 5. Save rg -> recording mapping for downstream use (02c) ---
    rg_rec_path = PROCESSED_DIR / "mb_rg_recording_map.csv"
    rg_recording.to_csv(rg_rec_path, index=False)
    print(f"  RG-recording map: {rg_rec_path} ({len(rg_recording):,} rows)")

    # --- 6. Coverage report ---
    print()
    print("=" * 70)
    print("COVERAGE REPORT")
    print("=" * 70)

    total_bands = linkage["ma_band_id"].nunique()
    bands_with_albums = result["ma_band_id"].nunique()
    total_rgs = result["mb_release_group_id"].nunique()
    total_recordings = rg_recording["recording_id"].nunique()

    print(f"  MA bands with MBID (step 01):      {total_bands:,}")
    print(f"  -> with MB release groups:          {bands_with_albums:,}  "
          f"({bands_with_albums / total_bands * 100:.1f}%)")
    print(f"  Total MB release groups linked:     {total_rgs:,}")
    print(f"  Total unique recording IDs:         {total_recordings:,}")
    print()

    # Per-band stats
    albums_per_band = result.groupby("ma_band_id").size()
    recs_per_rg = result["n_recordings"]
    print(f"  Release groups per band:")
    print(f"    mean:   {albums_per_band.mean():.1f}")
    print(f"    median: {albums_per_band.median():.1f}")
    print(f"    min:    {albums_per_band.min()}")
    print(f"    max:    {albums_per_band.max()}")
    print()
    print(f"  Recordings per release group:")
    print(f"    mean:   {recs_per_rg.mean():.1f}")
    print(f"    median: {recs_per_rg.median():.1f}")
    print(f"    min:    {recs_per_rg.min()}")
    print(f"    max:    {recs_per_rg.max()}")

    # Cross-reference with MA discography
    ma_disco = load_ma_discography()
    if ma_disco is not None:
        # Count MA releases for linked bands
        linked_band_ids = set(result["ma_band_id"].unique())
        ma_albums = ma_disco[ma_disco["Band ID"].isin(linked_band_ids)]
        print(f"\n  MA discography entries for linked bands: {len(ma_albums):,}")
        print(f"  MB release groups for same bands:        {len(result):,}")

    # Spot check
    print()
    print("SPOT CHECK -- Top bands by release group count:")
    top_bands = (
        result.groupby(["ma_band_id", "ma_name"])
        .size()
        .rename("n_release_groups")
        .reset_index()
        .sort_values("n_release_groups", ascending=False)
        .head(10)
    )
    for _, row in top_bands.iterrows():
        print(f"  {row['ma_name']:30s}  {row['n_release_groups']:4d} release groups")

    print()
    print("=" * 70)
    print("Step 02b complete. Proceed to step 02c (album feature aggregation).")
    print("=" * 70)


if __name__ == "__main__":
    main()
