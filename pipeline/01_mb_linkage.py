#!/usr/bin/env python3
"""01_mb_linkage.py — Link Metal Archives band IDs to MusicBrainz artist MBIDs.

Reads 3 MusicBrainz dump TSV files (no headers) and the Kaggle MA CSV
to produce a linkage table: ma_band_id → mbid.

MusicBrainz TSV schema (from CreateTables.sql):
  url:          id(0), gid(1), url(2), edits_pending(3), last_updated(4)
  l_artist_url: id(0), link(1), entity0(2), entity1(3), edits_pending(4), ...
  artist:       id(0), gid(1), name(2), sort_name(3), ...
"""

import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MB_DIR = DATA_DIR / "musicbrainz"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"

# MA URL pattern — extract band ID from the last path segment
# e.g. https://www.metal-archives.com/bands/Metallica/125 → 125
MA_URL_RE = re.compile(r"metal-archives\.com/bands?/.+/(\d+)")


def load_url_table() -> pd.DataFrame:
    """Load the MusicBrainz `url` table, filtered to metal-archives.com URLs."""
    path = MB_DIR / "url"
    print(f"Loading {path} ...")

    # Read only the columns we need: id (0) and url (2)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 2],
        names=["url_id", "url"],
        dtype={"url_id": "int64", "url": "str"},
        na_filter=False,
        quoting=3,  # QUOTE_NONE — MB dumps have no quoting
    )
    print(f"  Total URLs: {len(df):,}")

    # Filter for metal-archives.com band URLs
    mask = df["url"].str.contains("metal-archives.com/band", case=False, na=False)
    df = df[mask].copy()
    print(f"  Metal Archives URLs: {len(df):,}")

    # Extract MA band ID from URL
    df["ma_band_id"] = df["url"].apply(_extract_ma_id)
    df = df.dropna(subset=["ma_band_id"])
    df["ma_band_id"] = df["ma_band_id"].astype("int64")
    print(f"  With parseable band ID: {len(df):,}")

    return df[["url_id", "ma_band_id"]]


def _extract_ma_id(url: str) -> int | None:
    """Extract the numeric band ID from a Metal Archives URL."""
    m = MA_URL_RE.search(url)
    return int(m.group(1)) if m else None


def load_l_artist_url() -> pd.DataFrame:
    """Load the MusicBrainz `l_artist_url` table (relationship links)."""
    path = MB_DIR / "l_artist_url"
    print(f"Loading {path} ...")

    # We need: entity0 (col 2, artist.id) and entity1 (col 3, url.id)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[2, 3],
        names=["artist_id", "url_id"],
        dtype={"artist_id": "int64", "url_id": "int64"},
        na_filter=False,
        quoting=3,
    )
    print(f"  Total artist-URL relationships: {len(df):,}")
    return df


def load_artist_table() -> pd.DataFrame:
    """Load the MusicBrainz `artist` table (id, gid/MBID, name)."""
    path = MB_DIR / "artist"
    print(f"Loading {path} ...")

    # We need: id (0), gid (1), name (2)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["artist_id", "mbid", "mb_name"],
        dtype={"artist_id": "int64", "mbid": "str", "mb_name": "str"},
        na_filter=False,
        quoting=3,
    )
    print(f"  Total artists: {len(df):,}")
    return df


def load_ma_kaggle() -> pd.DataFrame:
    """Load the Metal Archives Kaggle CSV."""
    # Try common filenames
    candidates = [
        RAW_DIR / "metal_bands.csv",
        RAW_DIR / "bands.csv",
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break

    if path is None:
        # Fall back to any CSV in raw dir
        csvs = list(RAW_DIR.glob("*.csv"))
        if not csvs:
            print("ERROR: No CSV files found in data/raw/. Run 00_download_data.sh first.")
            sys.exit(1)
        # Pick the largest (likely the main bands file)
        path = max(csvs, key=lambda p: p.stat().st_size)

    print(f"Loading MA Kaggle CSV: {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df):,}")

    # Detect the band ID column — could be "Band ID", "id", "band_id", etc.
    id_col = _find_column(df, ["band id", "id", "band_id", "bandid"])
    name_col = _find_column(df, ["name", "band_name", "band", "bandname"])

    if id_col is None:
        print(f"ERROR: Could not find band ID column. Available: {list(df.columns)}")
        sys.exit(1)
    if name_col is None:
        print(f"WARNING: Could not find name column. Available: {list(df.columns)}")
        name_col = id_col  # fallback

    result = df[[id_col, name_col]].copy()
    result.columns = ["ma_band_id", "ma_name"]
    result["ma_band_id"] = pd.to_numeric(result["ma_band_id"], errors="coerce")
    result = result.dropna(subset=["ma_band_id"])
    result["ma_band_id"] = result["ma_band_id"].astype("int64")
    print(f"  Bands with valid ID: {len(result):,}")

    return result


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column name from a list of candidates (case-insensitive)."""
    col_map = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        if name.lower() in col_map:
            return col_map[name.lower()]
    return None


def main():
    print("=" * 60)
    print("heavyML Pipeline Step 01: MusicBrainz Linkage")
    print("=" * 60)
    print()

    # 1. Load MB url table, filtered to MA URLs
    url_df = load_url_table()
    print()

    # 2. Load l_artist_url (relationship table)
    rel_df = load_l_artist_url()
    print()

    # 3. Join: url → l_artist_url on url_id = entity1(url_id)
    print("Joining URL → artist relationships ...")
    joined = url_df.merge(rel_df, on="url_id", how="inner")
    print(f"  MA URLs with artist links: {len(joined):,}")
    # Drop duplicate ma_band_id → artist_id mappings (keep first)
    joined = joined.drop_duplicates(subset=["ma_band_id"], keep="first")
    print(f"  Unique MA band IDs linked: {len(joined):,}")
    print()

    # 4. Load artist table, join to get MBID
    artist_df = load_artist_table()
    print()

    print("Joining artist IDs → MBIDs ...")
    linked = joined.merge(artist_df, on="artist_id", how="inner")
    print(f"  Bands with MBID: {len(linked):,}")
    print()

    # 5. Load MA Kaggle CSV, merge on band ID
    ma_df = load_ma_kaggle()
    print()

    print("Merging with MA Kaggle dataset ...")
    result = linked.merge(ma_df, on="ma_band_id", how="inner")
    print(f"  Merged rows (before dedup): {len(result):,}")

    # De-duplicate: Kaggle CSV has ~470 duplicate Band IDs, causing
    # one MBID to map to multiple ma_name values.  Prefer the row
    # where ma_name matches mb_name (the MusicBrainz name is ground
    # truth since it was linked via the MA URL).
    result["_name_match"] = result["ma_name"].str.lower() == result["mb_name"].str.lower()
    result = result.sort_values("_name_match", ascending=False)
    result = result.drop_duplicates(subset=["ma_band_id"], keep="first")
    result = result.drop(columns=["_name_match"])
    print(f"  After dedup (prefer name match): {len(result):,}")
    print()

    # 6. Output
    out_path = OUT_DIR / "ma_mb_linkage.csv"
    out_df = result[["ma_band_id", "ma_name", "mbid", "mb_name"]].sort_values("ma_band_id")
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"  Rows: {len(out_df):,}")
    print()

    # 7. Coverage stats
    total_ma = len(ma_df)
    matched = len(out_df)
    pct = (matched / total_ma * 100) if total_ma > 0 else 0

    print("=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)
    print(f"  Total MA bands (Kaggle):  {total_ma:,}")
    print(f"  Matched with MBID:        {matched:,}")
    print(f"  Coverage:                  {pct:.1f}%")
    print()

    if matched >= 30_000:
        print("VERDICT: Coverage is sufficient (>30k). Proceed with pipeline.")
    elif matched >= 15_000:
        print("VERDICT: Coverage is marginal (15k-30k). Usable but may limit model quality.")
    else:
        print("VERDICT: Coverage is low (<15k). Consider alternative feature sources.")

    # 8. Spot-check known bands
    print()
    print("SPOT CHECK — Known bands:")
    spot_check = ["Metallica", "Iron Maiden", "Megadeth", "Black Sabbath", "Slayer"]
    for name in spot_check:
        row = out_df[out_df["ma_name"].str.lower() == name.lower()]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"  {r['ma_name']:20s}  MA={r['ma_band_id']:<8}  MBID={r['mbid']}")
        else:
            # Try MB name as fallback
            row = out_df[out_df["mb_name"].str.lower() == name.lower()]
            if len(row) > 0:
                r = row.iloc[0]
                print(f"  {r['mb_name']:20s}  MA={r['ma_band_id']:<8}  MBID={r['mbid']}  (matched via MB name)")
            else:
                print(f"  {name:20s}  NOT FOUND")


if __name__ == "__main__":
    main()
