#!/usr/bin/env python3
"""03_ma_similar_scraper.py — Scrape Metal Archives "Similar Artists" for ground-truth labels.

Scrapes the MA ajax-recommendations endpoint for bands in our MB linkage set
(46k bands with MBIDs). Prioritizes the most popular bands (by release count)
to maximize useful training pairs.

Uses cloudscraper to bypass Cloudflare protection.

Endpoint:
    GET https://www.metal-archives.com/band/ajax-recommendations/id/{band_id}?showMoreSimilar=1

Returns HTML with a <table id="artist_list"> containing rows:
    [band_name_link, country, genre, score]

Usage:
    python pipeline/03_ma_similar_scraper.py                  # scrape top 10k linked bands
    python pipeline/03_ma_similar_scraper.py --limit 10       # scrape first 10 unscraped bands
    python pipeline/03_ma_similar_scraper.py --sample 5000    # scrape top 5000 instead of 10k
    python pipeline/03_ma_similar_scraper.py --test           # test mode: scrape 5 known bands
    python pipeline/03_ma_similar_scraper.py --stats          # print progress stats and exit

Output:
    data/processed/ma_similar_artists.csv
        Columns: band_id, similar_band_id, similar_band_name, score

Progress is saved incrementally. The script is fully resumable — on restart it
skips bands that already have results (or confirmed "no similar artists").
"""

import argparse
import csv
import os
import random
import re
import sys
import time
from pathlib import Path

import json

import cloudscraper
import pandas as pd
import requests as stdlib_requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / "ma_similar_artists.csv"
# Track which band_ids we have already scraped (including those with 0 similar artists).
# This file stores one band_id per line — simple and fast to read/write.
PROGRESS_FILE = OUT_DIR / "ma_similar_scraper_progress.txt"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://www.metal-archives.com"
RECOMMENDATIONS_URL = BASE_URL + "/band/ajax-recommendations/id/{band_id}"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Headers mimicking a real browser — ported from Metaloreian Go scraper
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": BASE_URL,
    "Connection": "keep-alive",
}

# Rate limiting: randomised delay between requests (seconds)
DELAY_MIN = 2.0
DELAY_MAX = 5.0

# Exponential backoff on errors
BACKOFF_BASE = 10.0  # seconds
BACKOFF_MAX = 600.0  # 10 minutes
MAX_CONSECUTIVE_ERRORS = 20  # give up after this many consecutive errors

# FlareSolverr — headless browser Cloudflare bypass (Docker service)
# Run: docker run -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest
FLARESOLVERR_URL = os.environ.get("FLARESOLVERR_URL", "http://localhost:8191/v1")

# How often to flush output CSV (number of bands)
FLUSH_INTERVAL = 50

# Regex to extract band ID from a Metal Archives URL
BAND_ID_RE = re.compile(r"/bands?/[^/]+/(\d+)")

# ---------------------------------------------------------------------------
# Session setup
# ---------------------------------------------------------------------------

def _check_flaresolverr() -> bool:
    """Check if FlareSolverr is reachable."""
    try:
        resp = stdlib_requests.get(FLARESOLVERR_URL.replace("/v1", "/health"), timeout=5)
        return resp.status_code == 200
    except Exception:
        try:
            # Some versions don't have /health, try a dummy request
            resp = stdlib_requests.post(FLARESOLVERR_URL, json={"cmd": "sessions.list"}, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


def _fetch_via_flaresolverr(url: str) -> str | None:
    """Fetch a URL via FlareSolverr (headless browser bypass)."""
    payload = {
        "cmd": "request.get",
        "url": url,
        "maxTimeout": 60000,
    }
    try:
        resp = stdlib_requests.post(FLARESOLVERR_URL, json=payload, timeout=90)
        data = resp.json()
        if data.get("status") == "ok":
            solution = data.get("solution", {})
            return solution.get("response", "")
        else:
            return None
    except Exception as e:
        print(f"\n  FlareSolverr error: {e}")
        return None


# Global flag set during init
_use_flaresolverr = False


def create_session() -> cloudscraper.CloudScraper:
    """Create a scraper session. Uses FlareSolverr if available, else cloudscraper."""
    global _use_flaresolverr

    # Check FlareSolverr first
    if _check_flaresolverr():
        print("FlareSolverr detected — using headless browser bypass.")
        _use_flaresolverr = True
        # Still create a session for non-CF requests / fallback
        session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True},
        )
        session.headers.update(REQUEST_HEADERS)
        return session

    print("FlareSolverr not found — using cloudscraper (may get 403s).")
    print("  For best results, run: docker run -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest")
    _use_flaresolverr = False

    session = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "desktop": True},
    )
    session.headers.update(REQUEST_HEADERS)

    # Warm up
    print("Warming up session (hitting MA homepage via cloudscraper)...")
    try:
        resp = session.get(BASE_URL, timeout=30)
        print(f"  Homepage status: {resp.status_code}, cookies: {len(session.cookies)}")
    except Exception as e:
        print(f"  WARNING: Could not reach MA homepage: {e}")

    return session


# ---------------------------------------------------------------------------
# Scraping logic
# ---------------------------------------------------------------------------

def scrape_similar_artists(
    session: cloudscraper.CloudScraper,
    band_id: int,
) -> list[dict] | None:
    """Scrape similar artists for a single band.

    Returns:
        list of dicts with keys: similar_band_id, similar_band_name, score
        Empty list if the band has no similar artists.
        None if the request failed (should retry later).
    """
    url = RECOMMENDATIONS_URL.format(band_id=band_id) + "?showMoreSimilar=1"

    # Try FlareSolverr first if available
    if _use_flaresolverr:
        html = _fetch_via_flaresolverr(url)
        if html is None:
            print(f"\n  FlareSolverr failed for band {band_id}")
            return None
        return _parse_recommendations_html(html, band_id)

    # Fall back to cloudscraper
    try:
        resp = session.get(url, timeout=30)
    except Exception as e:
        print(f"\n  Network error for band {band_id}: {e}")
        return None

    # Check for Cloudflare challenge
    if "Just a moment" in resp.text or "challenge-platform" in resp.text:
        print(f"\n  Cloudflare challenge for band {band_id}")
        return None

    if resp.status_code == 520:
        print(f"\n  HTTP 520 (Cloudflare) for band {band_id}")
        return None

    if resp.status_code == 429:
        print(f"\n  HTTP 429 (rate limited) for band {band_id}")
        return None

    if resp.status_code != 200:
        print(f"\n  HTTP {resp.status_code} for band {band_id}")
        return None

    # Parse the HTML response
    return _parse_recommendations_html(resp.text, band_id)


def _parse_recommendations_html(html: str, band_id: int) -> list[dict]:
    """Parse the ajax-recommendations HTML response.

    The response contains a table with id="artist_list":
        <table id="artist_list">
          <tbody>
            <tr>
              <td><a href="/bands/SomeBand/12345">Some Band</a></td>
              <td>Country</td>
              <td>Genre</td>
              <td>123</td>  <!-- score -->
            </tr>
            ...
            <tr>
              <td id="no_artists" ...> or <td id="show_more" ...>
            </tr>
          </tbody>
        </table>
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"id": "artist_list"})

    if table is None:
        # Some bands have no similar artists table at all — the response
        # might just say "No similar artist has been recommended yet."
        if "no similar" in html.lower() or "no artists" in html.lower():
            return []
        # Could also be an empty/malformed response
        if len(html.strip()) < 50:
            return []
        # Unexpected format — log but don't crash
        print(f"\n  WARNING: No artist_list table for band {band_id}, "
              f"response length={len(html)}")
        return []

    recommendations = []
    tbody = table.find("tbody")
    if tbody is None:
        # Some pages put rows directly in the table without tbody
        rows = table.find_all("tr")
    else:
        rows = tbody.find_all("tr")

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 4:
            # End-of-table marker rows (show_more, no_artists)
            first_cell = cells[0] if cells else None
            if first_cell and first_cell.get("id") in ("show_more", "no_artists"):
                break
            continue

        # Cell 0: band name with link
        link = cells[0].find("a")
        if link is None:
            continue

        similar_band_name = link.get_text(strip=True)
        href = link.get("href", "")
        similar_band_id = _extract_band_id(href)
        if similar_band_id is None:
            continue

        # Cell 3: score (integer, user votes)
        score_text = cells[3].get_text(strip=True)
        try:
            score = int(score_text)
        except ValueError:
            # Sometimes the score cell might have non-numeric content
            score = 0

        recommendations.append({
            "similar_band_id": similar_band_id,
            "similar_band_name": similar_band_name,
            "score": score,
        })

    return recommendations


def _extract_band_id(url: str) -> int | None:
    """Extract numeric band ID from an MA URL like /bands/Metallica/125."""
    m = BAND_ID_RE.search(url)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress() -> set[int]:
    """Load the set of band IDs already scraped."""
    if not PROGRESS_FILE.exists():
        return set()
    done = set()
    with open(PROGRESS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(int(line))
                except ValueError:
                    pass
    return done


def save_progress_batch(band_ids: list[int]) -> None:
    """Append a batch of band IDs to the progress file."""
    with open(PROGRESS_FILE, "a") as f:
        for bid in band_ids:
            f.write(f"{bid}\n")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def init_output_csv() -> None:
    """Create the output CSV with headers if it doesn't exist."""
    if not OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["band_id", "similar_band_id", "similar_band_name", "score"])


def append_results(rows: list[dict]) -> None:
    """Append result rows to the output CSV."""
    if not rows:
        return
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow([
                row["band_id"],
                row["similar_band_id"],
                row["similar_band_name"],
                row["score"],
            ])


# ---------------------------------------------------------------------------
# Main scraping loop
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_SIZE = 10_000


def load_band_ids(sample_size: int | None = None) -> list[int]:
    """Load band IDs from the MB linkage CSV, prioritized by popularity.

    Only scrapes bands that have MBIDs (from Step 01 linkage).
    Ranks by number of releases in the discography CSV as a popularity proxy.
    Returns the top `sample_size` bands (default 10k).
    """
    linkage_path = OUT_DIR / "ma_mb_linkage.csv"
    if not linkage_path.exists():
        print(f"ERROR: {linkage_path} not found. Run 01_mb_linkage.py first.")
        sys.exit(1)

    linkage = pd.read_csv(linkage_path, usecols=["ma_band_id"])
    linked_ids = set(linkage["ma_band_id"].dropna().astype(int))
    print(f"Linked bands (with MBIDs): {len(linked_ids):,}")

    # Rank by number of releases (popularity proxy)
    disco_path = RAW_DIR / "all_bands_discography.csv"
    if disco_path.exists():
        disco = pd.read_csv(disco_path, usecols=["Band ID"], low_memory=False)
        release_counts = disco["Band ID"].value_counts()
        # Filter to linked bands, sort by release count descending
        ranked = release_counts[release_counts.index.isin(linked_ids)]
        band_ids = ranked.index.tolist()
        # Add any linked bands missing from discography at the end
        remaining = [bid for bid in linked_ids if bid not in set(band_ids)]
        band_ids.extend(remaining)
        print(f"Ranked by release count (top has {ranked.iloc[0] if len(ranked) > 0 else 0} releases)")
    else:
        band_ids = list(linked_ids)
        print("WARNING: No discography CSV found, using arbitrary order")

    if sample_size is not None:
        band_ids = band_ids[:sample_size]
        print(f"Sampling top {sample_size:,} bands")

    print(f"Band IDs to consider: {len(band_ids):,}")
    return band_ids


def run_scrape(limit: int | None = None, sample_size: int | None = DEFAULT_SAMPLE_SIZE) -> None:
    """Main scraping loop."""
    all_band_ids = load_band_ids(sample_size=sample_size)
    done = load_progress()
    print(f"Already scraped: {len(done):,} bands")

    # Filter to only unscraped bands
    todo = [bid for bid in all_band_ids if bid not in done]
    if limit is not None:
        todo = todo[:limit]

    if not todo:
        print("Nothing to scrape — all bands are already done!")
        return

    print(f"Bands to scrape this run: {len(todo):,}")
    print()

    init_output_csv()
    session = create_session()
    print()

    consecutive_errors = 0
    total_pairs = 0
    bands_with_similar = 0
    progress_batch: list[int] = []
    results_batch: list[dict] = []

    pbar = tqdm(todo, desc="Scraping", unit="band")
    for band_id in pbar:
        # Rate limiting — randomised delay
        delay = random.uniform(DELAY_MIN, DELAY_MAX)
        time.sleep(delay)

        result = scrape_similar_artists(session, band_id)

        if result is None:
            # Request failed — exponential backoff
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\n\nABORTING: {MAX_CONSECUTIVE_ERRORS} consecutive errors. "
                      f"Likely blocked or MA is down.")
                print("Progress has been saved — re-run to resume.")
                break

            backoff = min(BACKOFF_BASE * (2 ** (consecutive_errors - 1)), BACKOFF_MAX)
            # Add jitter
            backoff *= random.uniform(0.5, 1.5)
            pbar.write(f"  Backing off {backoff:.0f}s (error #{consecutive_errors})")
            time.sleep(backoff)

            # Re-create session in case cookies expired
            if consecutive_errors % 5 == 0:
                pbar.write("  Recreating session...")
                session = create_session()
            continue

        # Success — reset error counter
        consecutive_errors = 0

        # Record results
        if result:
            bands_with_similar += 1
            for rec in result:
                rec["band_id"] = band_id
            results_batch.extend(result)
            total_pairs += len(result)

        progress_batch.append(band_id)

        # Flush periodically
        if len(progress_batch) >= FLUSH_INTERVAL:
            append_results(results_batch)
            save_progress_batch(progress_batch)
            results_batch = []
            progress_batch = []

        # Update progress bar
        pbar.set_postfix(
            pairs=total_pairs,
            with_sim=bands_with_similar,
            errs=consecutive_errors,
        )

    # Final flush
    if results_batch:
        append_results(results_batch)
    if progress_batch:
        save_progress_batch(progress_batch)

    print()
    print("=" * 60)
    print("SCRAPE SUMMARY")
    print("=" * 60)
    total_done = len(load_progress())
    print(f"  Bands scraped (this run):  {len(todo):,}")
    print(f"  Bands scraped (total):     {total_done:,}")
    print(f"  Bands with similar artists:{bands_with_similar:,}")
    print(f"  Total similarity pairs:    {total_pairs:,}")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"  Progress: {PROGRESS_FILE}")


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

TEST_BANDS = [
    (125, "Metallica"),
    (72, "Iron Maiden"),
    (138, "Megadeth"),
    (99, "Black Sabbath"),
    (77, "Slayer"),
]


def run_test() -> None:
    """Test the scraper on a handful of well-known bands."""
    print("=" * 60)
    print("TEST MODE — scraping 5 known bands")
    print("=" * 60)
    print()

    session = create_session()
    print()

    for band_id, band_name in TEST_BANDS:
        print(f"--- {band_name} (ID={band_id}) ---")
        delay = random.uniform(DELAY_MIN, DELAY_MAX)
        print(f"  Waiting {delay:.1f}s...")
        time.sleep(delay)

        result = scrape_similar_artists(session, band_id)
        if result is None:
            print("  FAILED (request error)")
            continue
        if not result:
            print("  No similar artists found")
            continue

        print(f"  Found {len(result)} similar artists:")
        # Show top 5 by score
        sorted_result = sorted(result, key=lambda x: x["score"], reverse=True)
        for rec in sorted_result[:5]:
            print(f"    {rec['similar_band_name']:30s}  "
                  f"ID={rec['similar_band_id']:<10}  "
                  f"score={rec['score']}")
        if len(sorted_result) > 5:
            print(f"    ... and {len(sorted_result) - 5} more")
        print()


# ---------------------------------------------------------------------------
# Stats mode
# ---------------------------------------------------------------------------

def run_stats() -> None:
    """Print progress statistics."""
    print("=" * 60)
    print("SCRAPE PROGRESS STATS")
    print("=" * 60)
    print()

    all_band_ids = load_band_ids(sample_size=None)
    done = load_progress()
    remaining = len(all_band_ids) - len(done)

    print(f"  Total linked bands:         {len(all_band_ids):,}")
    print(f"  Bands scraped:              {len(done):,}")
    print(f"  Remaining:                  {remaining:,}")
    pct = (len(done) / len(all_band_ids) * 100) if all_band_ids else 0
    print(f"  Progress:                   {pct:.1f}%")
    print()

    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        print(f"  Total similarity pairs:     {len(df):,}")
        print(f"  Unique source bands:        {df['band_id'].nunique():,}")
        print(f"  Unique target bands:        {df['similar_band_id'].nunique():,}")
        if len(df) > 0:
            print(f"  Avg pairs per source band:  {len(df) / df['band_id'].nunique():.1f}")
            print(f"  Score range:                {df['score'].min()} — {df['score'].max()}")
            print()
            print("  Top 10 most-recommended bands:")
            top = df.groupby(["similar_band_id", "similar_band_name"]).size() \
                     .reset_index(name="count") \
                     .sort_values("count", ascending=False) \
                     .head(10)
            for _, row in top.iterrows():
                print(f"    {row['similar_band_name']:30s}  "
                      f"recommended by {row['count']} bands")
    else:
        print("  No output file yet.")

    # Time estimate
    if remaining > 0:
        avg_delay = (DELAY_MIN + DELAY_MAX) / 2
        est_hours = remaining * avg_delay / 3600
        print()
        print(f"  Estimated time remaining:   {est_hours:.0f} hours "
              f"({est_hours / 24:.1f} days) at ~{avg_delay:.1f}s/band")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape Metal Archives Similar Artists for ground-truth labels."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of bands to scrape (for testing)."
    )
    parser.add_argument(
        "--sample", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of top bands to sample (default {DEFAULT_SAMPLE_SIZE:,})."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: scrape 5 known bands and print results."
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print progress statistics and exit."
    )
    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.stats:
        run_stats()
    else:
        print("=" * 60)
        print("heavyML Pipeline Step 03: MA Similar Artists Scraper")
        print("=" * 60)
        print()
        run_scrape(limit=args.limit, sample_size=args.sample)


if __name__ == "__main__":
    main()
