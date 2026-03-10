#!/usr/bin/env python3
"""03b_lastfm_labels.py — Fetch similar artists from Last.fm for MA bands.

This is a HELD-OUT evaluation dataset. It is NOT used in model training.
Last.fm similarity is derived from listening behaviour (collaborative filtering),
providing an independent signal to measure whether the model's audio-feature-based
similarity predictions generalise beyond the Metal Archives ground truth.

Usage:
    python pipeline/03b_lastfm_labels.py              # full run (resumable)
    python pipeline/03b_lastfm_labels.py --limit 10   # test run on first 10 bands
    python pipeline/03b_lastfm_labels.py --reset       # clear checkpoint, start fresh

Requires:
    LASTFM_API_KEY env var (or in .env file at project root)
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"

OUTPUT_CSV = OUT_DIR / "lastfm_similar_artists.csv"
CHECKPOINT_FILE = OUT_DIR / "lastfm_checkpoint.json"

# ---------------------------------------------------------------------------
# Last.fm API config
# ---------------------------------------------------------------------------
LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"
REQUEST_DELAY = 0.34  # ~3 req/sec (stay under 5/sec limit)
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds, doubles on each retry
REQUEST_TIMEOUT = 15  # seconds


def load_api_key() -> str:
    """Load LASTFM_API_KEY from environment or .env file."""
    # Try .env first
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    key = os.environ.get("LASTFM_API_KEY", "").strip()
    if not key:
        print("=" * 60)
        print("ERROR: LASTFM_API_KEY not found.")
        print("=" * 60)
        print()
        print("To fix this:")
        print("  1. Go to https://www.last.fm/api/account/create")
        print("  2. Sign up / log in, create an API application")
        print("  3. Copy your API Key")
        print("  4. Either:")
        print(f"     a) Create {PROJECT_DIR / '.env'} with:")
        print("        LASTFM_API_KEY=your_api_key_here")
        print("     b) Or export it:")
        print("        export LASTFM_API_KEY=your_api_key_here")
        print()
        sys.exit(1)

    return key


def load_ma_bands() -> pd.DataFrame:
    """Load Metal Archives Kaggle CSV — band IDs and names."""
    path = RAW_DIR / "metal_bands.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run 00_download_data.sh first.")
        sys.exit(1)

    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)

    # Normalise column names
    id_col = _find_column(df, ["band id", "id", "band_id"])
    name_col = _find_column(df, ["name", "band_name", "band"])

    if id_col is None or name_col is None:
        print(f"ERROR: Could not find required columns. Available: {list(df.columns)}")
        sys.exit(1)

    result = df[[id_col, name_col]].copy()
    result.columns = ["band_id", "band_name"]
    result["band_id"] = pd.to_numeric(result["band_id"], errors="coerce")
    result = result.dropna(subset=["band_id"])
    result["band_id"] = result["band_id"].astype("int64")
    # Drop rows with empty/missing names
    result = result.dropna(subset=["band_name"])
    result = result[result["band_name"].str.strip() != ""]
    print(f"  Loaded {len(result):,} bands")
    return result


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column name from a list of candidates (case-insensitive)."""
    col_map = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        if name.lower() in col_map:
            return col_map[name.lower()]
    return None


def build_name_index(bands_df: pd.DataFrame) -> dict[str, int]:
    """Build a case-insensitive name → band_id index for cross-referencing.

    When multiple bands share the same name (case-insensitive), keeps the first.
    This is acceptable for evaluation — exact duplicates are rare and the match
    score threshold will filter out noise.
    """
    index: dict[str, int] = {}
    for _, row in bands_df.iterrows():
        key = row["band_name"].strip().lower()
        if key not in index:
            index[key] = row["band_id"]
    return index


def load_checkpoint() -> set[int]:
    """Load the set of already-queried band IDs from checkpoint."""
    if not CHECKPOINT_FILE.exists():
        return set()

    with open(CHECKPOINT_FILE, "r") as f:
        data = json.load(f)

    queried = set(data.get("queried_band_ids", []))
    print(f"  Resuming from checkpoint: {len(queried):,} bands already queried")
    return queried


def save_checkpoint(queried_ids: set[int]) -> None:
    """Save the set of queried band IDs to checkpoint."""
    data = {"queried_band_ids": sorted(queried_ids)}
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)


def fetch_similar_artists(
    artist_name: str, api_key: str, session: requests.Session
) -> list[dict] | None:
    """Call Last.fm artist.getSimilar for the given artist.

    Returns a list of dicts with keys: name, match, mbid
    Returns None if artist not found or on permanent error.
    Raises on transient errors (caller should retry).
    """
    params = {
        "method": "artist.getSimilar",
        "artist": artist_name,
        "api_key": api_key,
        "format": "json",
        "limit": 100,  # get up to 100 similar artists
    }

    resp = session.get(
        LASTFM_BASE_URL, params=params, timeout=REQUEST_TIMEOUT
    )

    # Rate limit exceeded — Last.fm returns 429
    if resp.status_code == 429:
        raise requests.exceptions.ConnectionError("Rate limited (429)")

    # Server errors — transient, should retry
    if resp.status_code >= 500:
        raise requests.exceptions.ConnectionError(f"Server error ({resp.status_code})")

    # Client errors other than "not found"
    if resp.status_code != 200:
        # Last.fm returns 200 even for "artist not found" with error in body
        # But just in case:
        return None

    data = resp.json()

    # Last.fm error response (e.g. artist not found)
    if "error" in data:
        error_code = data.get("error", 0)
        # Error 6 = "Artist not found"
        if error_code == 6:
            return None
        # Error 29 = "Rate limit exceeded"
        if error_code == 29:
            raise requests.exceptions.ConnectionError("Rate limited (error 29)")
        # Other errors — treat as not found
        return None

    # Parse response
    similar_data = data.get("similarartists", {})
    artists_raw = similar_data.get("artist", [])

    # Last.fm returns an empty string instead of empty list when no results
    if not artists_raw or isinstance(artists_raw, str):
        return []

    results = []
    for a in artists_raw:
        if not isinstance(a, dict):
            continue
        results.append({
            "name": a.get("name", "").strip(),
            "match": float(a.get("match", 0.0)),
            "mbid": a.get("mbid", "").strip(),
        })

    return results


def query_band(
    band_id: int,
    band_name: str,
    api_key: str,
    session: requests.Session,
    name_index: dict[str, int],
    writer: csv.writer,
    out_file,
) -> int:
    """Query Last.fm for a single band and write results.

    Returns the number of similar artists found.
    """
    similar = None
    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            similar = fetch_similar_artists(band_name, api_key, session)
            break
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                tqdm.write(f"  WARN: Failed after {MAX_RETRIES} retries for '{band_name}': {e}")
                return 0
        except Exception as e:
            tqdm.write(f"  WARN: Unexpected error for '{band_name}': {e}")
            return 0

    if similar is None:
        # Artist not found on Last.fm — not an error, just skip
        return 0

    count = 0
    for s in similar:
        if not s["name"]:
            continue
        # Cross-reference similar artist name to MA band ID
        similar_band_id = name_index.get(s["name"].strip().lower(), "")
        writer.writerow([
            band_id,
            band_name,
            s["name"],
            similar_band_id,
            f"{s['match']:.6f}",
        ])
        count += 1

    # Flush periodically to avoid data loss
    out_file.flush()

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Last.fm similar artists for MA bands (held-out eval)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N bands (0 = all, default: 0)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear checkpoint and start fresh",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("heavyML Pipeline Step 03b: Last.fm Similar Artists")
    print("=" * 60)
    print("  Purpose: Held-out evaluation signal (NOT used in training)")
    print()

    # 1. Load API key
    api_key = load_api_key()
    print("  API key loaded successfully.")
    print()

    # 2. Load MA bands
    bands_df = load_ma_bands()
    print()

    # 3. Build name → band_id index for cross-referencing
    print("Building name index for cross-referencing ...")
    name_index = build_name_index(bands_df)
    print(f"  Indexed {len(name_index):,} unique band names")
    print()

    # 4. Handle checkpoint / resume
    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("  Checkpoint cleared.")
        if OUTPUT_CSV.exists():
            OUTPUT_CSV.unlink()
            print("  Previous output cleared.")
        print()

    queried_ids = load_checkpoint()

    # 5. Determine which bands to query
    all_band_ids = bands_df["band_id"].tolist()
    all_band_names = bands_df["band_name"].tolist()

    if args.limit > 0:
        all_band_ids = all_band_ids[: args.limit]
        all_band_names = all_band_names[: args.limit]
        print(f"  Limited to first {args.limit} bands")

    # Filter out already-queried bands
    to_query = [
        (bid, bname)
        for bid, bname in zip(all_band_ids, all_band_names)
        if bid not in queried_ids
    ]
    print(f"  Bands to query: {len(to_query):,} (skipping {len(all_band_ids) - len(to_query):,} already done)")
    print()

    if not to_query:
        print("All bands already queried. Use --reset to start fresh.")
        _print_stats()
        return

    # 6. Ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 7. Open output CSV (append mode if resuming)
    file_exists = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0
    mode = "a" if file_exists and not args.reset else "w"

    session = requests.Session()
    session.headers.update({"User-Agent": "heavyML/0.1 (research project)"})

    bands_with_results = 0
    total_similar = 0
    bands_queried = 0

    # Checkpoint save interval (every N bands)
    CHECKPOINT_INTERVAL = 100

    with open(OUTPUT_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header if new file
        if mode == "w":
            writer.writerow([
                "band_id",
                "band_name",
                "similar_name",
                "similar_band_id",
                "match_score",
            ])

        pbar = tqdm(to_query, desc="Querying Last.fm", unit="band")
        for band_id, band_name in pbar:
            count = query_band(
                band_id, band_name, api_key, session, name_index, writer, f
            )

            queried_ids.add(band_id)
            bands_queried += 1
            if count > 0:
                bands_with_results += 1
                total_similar += count

            # Update progress bar
            hit_rate = (
                f"{bands_with_results / bands_queried * 100:.0f}%"
                if bands_queried > 0
                else "0%"
            )
            pbar.set_postfix(
                hits=bands_with_results,
                hit_rate=hit_rate,
                avg_sim=(
                    f"{total_similar / bands_with_results:.1f}"
                    if bands_with_results > 0
                    else "0"
                ),
            )

            # Save checkpoint periodically
            if bands_queried % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(queried_ids)

            # Rate limit
            time.sleep(REQUEST_DELAY)

    # Final checkpoint save
    save_checkpoint(queried_ids)

    print()
    _print_stats()


def _print_stats():
    """Print summary statistics from the output CSV."""
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    if not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0:
        print("  No results yet.")
        return

    df = pd.read_csv(OUTPUT_CSV)
    total_pairs = len(df)
    unique_bands = df["band_id"].nunique()
    bands_with_ma_match = df[df["similar_band_id"] != ""]["similar_band_id"].count()

    # Handle case where similar_band_id might be mixed types
    try:
        ma_matched = df[
            pd.to_numeric(df["similar_band_id"], errors="coerce").notna()
        ].shape[0]
    except Exception:
        ma_matched = 0

    avg_similar = total_pairs / unique_bands if unique_bands > 0 else 0

    print(f"  Output file:              {OUTPUT_CSV}")
    print(f"  Total similarity pairs:   {total_pairs:,}")
    print(f"  Unique bands queried:     {unique_bands:,}")
    print(f"  Avg similar artists/band: {avg_similar:.1f}")
    print(f"  Pairs matched to MA ID:   {ma_matched:,} ({ma_matched / total_pairs * 100:.1f}%)" if total_pairs > 0 else "  Pairs matched to MA ID:   0")

    # Checkpoint stats
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        total_queried = len(data.get("queried_band_ids", []))
        print(f"  Total bands queried:      {total_queried:,}")
        remaining = 183_397 - total_queried
        if remaining > 0:
            hours = remaining / 3 / 3600
            print(f"  Remaining:                {remaining:,} (~{hours:.1f} hours at 3 req/sec)")

    print()


if __name__ == "__main__":
    main()
