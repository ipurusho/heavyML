#!/usr/bin/env python3
"""02c_album_ids.py — Extract MA album IDs from band discography pages.

The raw discography CSV (all_bands_discography.csv) has album names, types, years,
and band IDs, but NO album IDs. Album IDs are embedded in the HTML of MA's
discography AJAX endpoint as href attributes on album links.

This script scrapes the discography tab for each band and extracts album IDs.

Endpoint:
    GET https://www.metal-archives.com/band/discography/id/{band_id}/tab/all

Returns an HTML table where each row has:
    <a href="https://www.metal-archives.com/albums/Band/Album/12345">Album Name</a>

Usage:
    python pipeline/02c_album_ids.py                  # scrape all bands
    python pipeline/02c_album_ids.py --limit 10       # scrape first 10 unscraped bands
    python pipeline/02c_album_ids.py --test           # test mode: scrape 5 known bands
    python pipeline/02c_album_ids.py --stats          # print progress stats and exit

Output:
    data/processed/ma_album_ids.csv
        Columns: band_id, album_id, album_name, album_type, year

Progress is saved incrementally. The script is fully resumable — on restart it
skips bands that already have results (or confirmed "no albums").
"""

import argparse
import csv
import os
import random
import re
import sys
import time
from pathlib import Path

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

OUTPUT_CSV = OUT_DIR / "ma_album_ids.csv"
PROGRESS_FILE = OUT_DIR / "ma_album_ids_progress.txt"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://www.metal-archives.com"
DISCOGRAPHY_URL = BASE_URL + "/band/discography/id/{band_id}/tab/all"

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
MAX_CONSECUTIVE_ERRORS = 20

# FlareSolverr — headless browser Cloudflare bypass
FLARESOLVERR_URL = os.environ.get("FLARESOLVERR_URL", "http://localhost:8191/v1")

# How often to flush output CSV (number of bands)
FLUSH_INTERVAL = 50

# Regex to extract album ID from MA album URL: /albums/Band/Album/12345
ALBUM_ID_RE = re.compile(r"/albums/[^/]+/[^/]+/(\d+)")

# ---------------------------------------------------------------------------
# Session setup
# ---------------------------------------------------------------------------

def _check_flaresolverr() -> bool:
    """Check if FlareSolverr is reachable."""
    try:
        resp = stdlib_requests.get(
            FLARESOLVERR_URL.replace("/v1", "/health"), timeout=5
        )
        return resp.status_code == 200
    except Exception:
        try:
            resp = stdlib_requests.post(
                FLARESOLVERR_URL, json={"cmd": "sessions.list"}, timeout=5
            )
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

    if _check_flaresolverr():
        print("FlareSolverr detected — using headless browser bypass.")
        _use_flaresolverr = True
        session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True},
        )
        session.headers.update(REQUEST_HEADERS)
        return session

    print("FlareSolverr not found — using cloudscraper (may get 403s).")
    print(
        "  For best results, run: "
        "docker run -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest"
    )
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

def scrape_discography(
    session: cloudscraper.CloudScraper,
    band_id: int,
) -> list[dict] | None:
    """Scrape album IDs from a band's discography page.

    Returns:
        list of dicts with keys: album_id, album_name, album_type, year
        Empty list if the band has no albums.
        None if the request failed (should retry later).
    """
    url = DISCOGRAPHY_URL.format(band_id=band_id)

    if _use_flaresolverr:
        html = _fetch_via_flaresolverr(url)
        if html is None:
            print(f"\n  FlareSolverr failed for band {band_id}")
            return None
        return _parse_discography_html(html, band_id)

    try:
        resp = session.get(url, timeout=30)
    except Exception as e:
        print(f"\n  Network error for band {band_id}: {e}")
        return None

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

    return _parse_discography_html(resp.text, band_id)


def _parse_discography_html(html: str, band_id: int) -> list[dict]:
    """Parse the discography AJAX HTML response.

    The response contains:
        <table class="display discog" ...>
          <tbody>
            <tr>
              <td><a href="/albums/Band/Album/12345">Album Name</a></td>
              <td>Full-length</td>
              <td>2003</td>
              <td>3 (85%)</td>  <!-- reviews -->
            </tr>
            ...
          </tbody>
        </table>
    """
    soup = BeautifulSoup(html, "html.parser")

    # The discography table has class "display discog"
    table = soup.find("table", class_="discog")
    if table is None:
        # Some bands might have no discography at all
        if "nothing" in html.lower() or len(html.strip()) < 100:
            return []
        # Try a more general search
        table = soup.find("table")
        if table is None:
            print(
                f"\n  WARNING: No discography table for band {band_id}, "
                f"response length={len(html)}"
            )
            return []

    albums = []
    tbody = table.find("tbody")
    rows = (tbody or table).find_all("tr")

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        # Cell 0: album name with link containing album ID
        link = cells[0].find("a")
        if link is None:
            continue

        album_name = link.get_text(strip=True)
        href = link.get("href", "")

        m = ALBUM_ID_RE.search(href)
        if m is None:
            continue

        try:
            album_id = int(m.group(1))
        except ValueError:
            continue

        # Cell 1: type (Full-length, EP, Demo, etc.)
        album_type = cells[1].get_text(strip=True)

        # Cell 2: year
        year = cells[2].get_text(strip=True)

        albums.append({
            "album_id": album_id,
            "album_name": album_name,
            "album_type": album_type,
            "year": year,
        })

    return albums


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
            writer.writerow(["band_id", "album_id", "album_name", "album_type", "year"])


def append_results(rows: list[dict]) -> None:
    """Append result rows to the output CSV."""
    if not rows:
        return
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow([
                row["band_id"],
                row["album_id"],
                row["album_name"],
                row["album_type"],
                row["year"],
            ])


# ---------------------------------------------------------------------------
# Main scraping loop
# ---------------------------------------------------------------------------

def load_band_ids() -> list[int]:
    """Load all band IDs from the raw metal_bands CSV.

    Reads band IDs from data/raw/metal_bands.csv (the Band ID column).
    Returns all band IDs in file order.
    """
    bands_path = RAW_DIR / "metal_bands.csv"
    if not bands_path.exists():
        print(f"ERROR: {bands_path} not found.")
        sys.exit(1)

    bands = pd.read_csv(bands_path, usecols=["Band ID"])
    band_ids = bands["Band ID"].dropna().astype(int).tolist()
    print(f"Total bands in metal_bands.csv: {len(band_ids):,}")
    return band_ids


def run_scrape(limit: int | None = None) -> None:
    """Main scraping loop."""
    all_band_ids = load_band_ids()
    done = load_progress()
    print(f"Already scraped: {len(done):,} bands")

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
    total_albums = 0
    bands_with_albums = 0
    progress_batch: list[int] = []
    results_batch: list[dict] = []

    pbar = tqdm(todo, desc="Scraping discographies", unit="band")
    for band_id in pbar:
        # Rate limiting — randomised delay
        delay = random.uniform(DELAY_MIN, DELAY_MAX)
        time.sleep(delay)

        result = scrape_discography(session, band_id)

        if result is None:
            # Request failed — exponential backoff
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(
                    f"\n\nABORTING: {MAX_CONSECUTIVE_ERRORS} consecutive errors. "
                    f"Likely blocked or MA is down."
                )
                print("Progress has been saved — re-run to resume.")
                break

            backoff = min(
                BACKOFF_BASE * (2 ** (consecutive_errors - 1)), BACKOFF_MAX
            )
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

        if result:
            bands_with_albums += 1
            for album in result:
                album["band_id"] = band_id
            results_batch.extend(result)
            total_albums += len(result)

        progress_batch.append(band_id)

        # Flush periodically
        if len(progress_batch) >= FLUSH_INTERVAL:
            append_results(results_batch)
            save_progress_batch(progress_batch)
            results_batch = []
            progress_batch = []

        pbar.set_postfix(
            albums=total_albums,
            with_disco=bands_with_albums,
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
    print(f"  Bands scraped (this run):    {len(todo):,}")
    print(f"  Bands scraped (total):       {total_done:,}")
    print(f"  Bands with albums:           {bands_with_albums:,}")
    print(f"  Total album IDs extracted:   {total_albums:,}")
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
    print("TEST MODE — scraping discographies for 5 known bands")
    print("=" * 60)
    print()

    session = create_session()
    print()

    for band_id, band_name in TEST_BANDS:
        print(f"--- {band_name} (ID={band_id}) ---")
        delay = random.uniform(DELAY_MIN, DELAY_MAX)
        print(f"  Waiting {delay:.1f}s...")
        time.sleep(delay)

        result = scrape_discography(session, band_id)
        if result is None:
            print("  FAILED (request error)")
            continue
        if not result:
            print("  No albums found")
            continue

        print(f"  Found {len(result)} albums:")
        for album in result[:10]:
            print(
                f"    {album['album_name']:40s}  "
                f"ID={album['album_id']:<10}  "
                f"{album['album_type']:15s}  "
                f"{album['year']}"
            )
        if len(result) > 10:
            print(f"    ... and {len(result) - 10} more")
        print()


# ---------------------------------------------------------------------------
# Stats mode
# ---------------------------------------------------------------------------

def run_stats() -> None:
    """Print progress statistics."""
    print("=" * 60)
    print("ALBUM ID SCRAPE PROGRESS STATS")
    print("=" * 60)
    print()

    all_band_ids = load_band_ids()
    done = load_progress()
    remaining = len(all_band_ids) - len(done)

    print(f"  Total bands:                {len(all_band_ids):,}")
    print(f"  Bands scraped:              {len(done):,}")
    print(f"  Remaining:                  {remaining:,}")
    pct = (len(done) / len(all_band_ids) * 100) if all_band_ids else 0
    print(f"  Progress:                   {pct:.1f}%")
    print()

    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        print(f"  Total album IDs:            {len(df):,}")
        print(f"  Unique bands with albums:   {df['band_id'].nunique():,}")
        if len(df) > 0:
            print(
                f"  Avg albums per band:        "
                f"{len(df) / df['band_id'].nunique():.1f}"
            )
            type_counts = df["album_type"].value_counts()
            print()
            print("  Album type breakdown:")
            for atype, count in type_counts.items():
                print(f"    {atype:20s}  {count:,}")
    else:
        print("  No output file yet.")

    # Time estimate
    if remaining > 0:
        avg_delay = (DELAY_MIN + DELAY_MAX) / 2
        est_hours = remaining * avg_delay / 3600
        print()
        print(
            f"  Estimated time remaining:   {est_hours:.0f} hours "
            f"({est_hours / 24:.1f} days) at ~{avg_delay:.1f}s/band"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract MA album IDs from band discography pages."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of bands to scrape (for testing).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: scrape 5 known bands and print results.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print progress statistics and exit.",
    )
    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.stats:
        run_stats()
    else:
        print("=" * 60)
        print("heavyML Pipeline Step 02c: MA Album ID Extraction")
        print("=" * 60)
        print()
        run_scrape(limit=args.limit)


if __name__ == "__main__":
    main()
