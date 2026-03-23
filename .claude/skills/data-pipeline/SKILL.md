---
name: data-pipeline
description: >
  Run the heavyML data pipeline: download raw data, link MA→MusicBrainz→AcousticBrainz,
  scrape similarity labels, build feature matrices. Use this skill when the user says
  "run the pipeline", "refresh data", "rebuild features", "download data", "scrape similar
  artists", "link bands", or wants to re-run any data processing step for heavyML.
---

# Data Pipeline

You are orchestrating the heavyML data pipeline, which transforms raw metal band data into
training-ready feature matrices and similarity pairs. The pipeline is sequential — each step
depends on the previous one's output.

## Prerequisites

- Working directory: `/home/imman/projects/heavyML`
- Python venv activated: `source .venv/bin/activate`
- `.env` file with `LASTFM_API_KEY` (only needed for step 03b)
- Kaggle CLI configured (only needed for step 00)

## Pipeline Steps

The steps are numbered 00–06. Each is idempotent — it checks for existing output before
re-processing. Run them in order unless the user asks for a specific step.

### Step 00: Download Raw Data
```bash
bash pipeline/00_download_data.sh
```
Downloads Metal Archives Kaggle CSV and MusicBrainz dump TSVs. Skips files that already exist.

**Outputs:** `data/raw/metal_bands.csv`, `data/musicbrainz/*.tsv`

### Step 01: MusicBrainz Linkage
```bash
python pipeline/01_mb_linkage.py
```
Links MA Band IDs → MusicBrainz Artist MBIDs via URL relationships in the MB dump.

**Input:** `data/raw/metal_bands.csv`, `data/musicbrainz/` TSVs
**Output:** `data/processed/ma_mb_linkage.csv` (~46k rows)

### Step 02: AcousticBrainz Feature Extraction
```bash
python pipeline/02_ab_features.py
```
Bridges MA bands → MB artists → AB recordings. Aggregates track-level audio features
(BPM, energy, loudness, key, onset rate, MFCC, danceability) to band-level means.

**Input:** `data/processed/ma_mb_linkage.csv`, `data/acousticbrainz/*.csv`
**Output:** `data/processed/ma_ab_features.csv` (~16-20k rows × 13 features)

### Step 03: Scrape MA Similar Artists (Primary Ground Truth)
```bash
python pipeline/03_ma_similar_scraper.py
```
Scrapes Metal Archives "Similar Artists" section via AJAX. Resumable with progress tracking.
Rate-limited (2–5s delays). Uses cloudscraper + optional FlareSolverr for Cloudflare.

**Input:** `data/raw/metal_bands.csv` (for band URLs)
**Output:** `data/processed/ma_similar_artists.csv` (~184k pairs)

This step is slow (hours to days). Check progress with:
```bash
wc -l data/processed/ma_similar_artists.csv
```

### Step 03b: Last.fm Similar Artists (Held-Out Eval)
```bash
python pipeline/03b_lastfm_labels.py
```
Fetches Last.fm `artist.getSimilar` for cross-source evaluation. NOT used in training —
provides independent generalization signal.

**Input:** `data/processed/ma_mb_linkage.csv`, `.env` (LASTFM_API_KEY)
**Output:** `data/processed/lastfm_similar_artists.csv` (~3.37M pairs)

### Step 04: Build Feature Matrix
```bash
python pipeline/04_feature_matrix.py
```
Normalizes numeric features (StandardScaler), one-hot encodes keys, binary-encodes scale.
Filters similarity pairs to bands with complete features.

**Input:** `data/processed/ma_ab_features.csv`, `data/processed/ma_similar_artists.csv`
**Outputs:**
- `data/processed/feature_matrix.npy` (N × 21 dims)
- `data/processed/feature_band_ids.csv`
- `data/processed/valid_pairs.csv` (~115k pairs)
- `data/processed/scaler_params.npz`

## Running the Full Pipeline

```bash
cd /home/imman/projects/heavyML
source .venv/bin/activate
bash pipeline/00_download_data.sh
python pipeline/01_mb_linkage.py
python pipeline/02_ab_features.py
python pipeline/03_ma_similar_scraper.py
python pipeline/03b_lastfm_labels.py
python pipeline/04_feature_matrix.py
```

## Validation Checks

After each step, verify outputs exist and have reasonable row counts:
```bash
wc -l data/processed/*.csv
ls -lh data/processed/*.npy
```

Expected counts:
- `ma_mb_linkage.csv`: ~46k rows
- `ma_ab_features.csv`: ~16-20k rows
- `ma_similar_artists.csv`: ~184k pairs
- `valid_pairs.csv`: ~115k pairs

If counts are significantly off, investigate before proceeding.

## Error Handling

- If step 00 fails: check Kaggle CLI auth (`kaggle competitions list`)
- If step 01 produces 0 rows: MB dump format may have changed — check TSV column order
- If step 02 is slow: AB CSVs are ~30M rows each, expect 10-20 min
- If step 03 gets blocked: Cloudflare is active — ensure FlareSolverr is running
- If step 03b rate-limits: Last.fm free tier is 5 req/sec, script handles this
