# heavyML — Metaloreian ML

## Project Overview
A companion ML project to [Metaloreian](https://github.com/ipurusho/metaloreian) (production Go/React/PostgreSQL app integrating Metal Archives + Spotify). This repo adds a model layer that powers a **"Sonic Similar Bands"** feature: given a band, return the 10 most sonically similar bands using a neural retrieval model trained on real audio features.

---

## Architecture

### Product Feature
User lands on a band page in Metaloreian → clicks "Find Similar Bands" → Go backend queries pgvector directly → returns 10 sonically similar bands with Spotify playback.

**Zero forward passes at query time.** All embeddings are pre-computed offline. Inference is a pgvector ANN query from the existing Go backend — no separate Python serving layer.

### Data Sources
1. **Metal Archives dataset** (`data/raw/`) — 183k bands, genre tags, country, discography, status. Already downloaded from Kaggle (October 2024 dump).
2. **MusicBrainz DB dump** — links MA Band IDs to MusicBrainz Artist IDs via `metal-archives.com` URL relationships.
3. **AcousticBrainz CSV dump** — audio features (BPM, energy, danceability, key, loudness, onset rate, MFCC) keyed by MusicBrainz Recording ID. 29.4M tracks.
4. **Metal Archives scraper** — scrape `Similar Artists` section from MA band pages (URLs already in dataset) for ground-truth similarity labels. Cloudflare bypass ported from Metaloreian.
5. **Last.fm API** — `artist.getSimilar` as secondary ground-truth source. Used for cross-source eval: model trains on MA labels, agreement with Last.fm is a held-out generalisation signal.

### Linkage Chain
```
MA Band ID → (MusicBrainz URL relationships) → MB Artist MBID
                                                      ↓
                                          MB Recording MBIDs
                                                      ↓
                                     AcousticBrainz audio features
                                     (BPM, energy, key, loudness...)
```

### Model Architecture
Siamese MLP with contrastive learning:
```
Band audio fingerprint (20 dims)  →  MLP (20→32→16)  →  L2-normalized 16-dim embedding
```
- **Parameters:** 1,264
- **Loss:** InfoNCE (noise-contrastive estimation)
- **Positives:** Metal Archives similar artist pairs (scraped)
- **Negatives:** In-batch random
- **Eval:** Recall@10, MRR, genre purity, Last.fm agreement

### v2 Improvements (after v1 ships)
- Richer features: Spotify audio features, Essentia, genre/tag features, graph-based signal
- Fine-tune a small LLM (Phi-3 mini or Llama 3.2 3B) with QLoRA on Metal Archives data to generate natural-language band descriptions explaining sonic similarity.

---

## Repo Structure

```
heavyML/
├── CLAUDE.md                        ← you are here
├── README.md
├── data/
│   ├── raw/                         ← MA Kaggle dump CSVs (already present)
│   │   ├── metal_bands.csv
│   │   ├── all_bands_discography.csv
│   │   ├── complete_roster.csv
│   │   ├── labels_roster.csv
│   │   └── metal_bands_roster.csv
│   ├── musicbrainz/                 ← MB dump TSVs (artist, url, l_artist_url, artist_credit_name, recording)
│   ├── acousticbrainz/              ← AB CSV dumps (to be downloaded)
│   └── processed/                   ← feature matrices, train/val/test splits
├── pipeline/
│   ├── 00_download_data.sh          ← idempotent download: Kaggle + MB dump
│   ├── 01_mb_linkage.py             ← MA Band ID → MB Artist MBID (TSV join, no PostgreSQL)
│   ├── 02_ab_features.py            ← MB Recording IDs → AcousticBrainz features
│   ├── 03_ma_similar_scraper.py     ← scrape MA similar artists (primary ground truth)
│   ├── 03b_lastfm_labels.py         ← Last.fm similar artists (secondary / held-out eval)
│   ├── 04_feature_matrix.py         ← aggregate track features → band fingerprints
│   ├── 05_train_val_split.py        ← stratified split
│   └── 06_export_embeddings.py      ← pre-compute embeddings + generate pgvector SQL
├── model/
│   ├── tower.py                     ← MLP tower definition (raw PyTorch)
│   ├── loss.py                      ← InfoNCE implementation
│   ├── dataset.py                   ← contrastive pair sampling
│   ├── train.py                     ← training loop
│   ├── evaluate.py                  ← Recall@k, MRR, genre purity, Last.fm agreement
│   └── baseline_cosine.py           ← cosine similarity baseline
├── notebooks/                       ← (empty, for future EDA)
├── requirements.txt
├── LICENSE
└── .env.example
```

---

## Current Status

### Completed (v1 pipeline)
- [x] **Step 00** — MA Kaggle dataset downloaded (183k bands, 636k releases)
- [x] **Step 01** — MusicBrainz linkage: 46,229 / 183,397 MA bands linked to MBIDs (25.2%)
  - Fixed: Kaggle CSV has 471 duplicate Band IDs — dedup prefers rows where ma_name == mb_name
- [x] **Step 02** — AcousticBrainz feature extraction: 7 audio features (BPM, energy, loudness, key, onset rate, MFCC, danceability) + key/scale encoding → 20-dim feature vector per band
- [x] **Step 03** — MA similar artists scraped: 184,853 pairs from 7,420 bands
- [x] **Step 03b** — Last.fm similar artists fetched: 3.37M pairs from 46,229 bands
- [x] **Step 04** — Contrastive training pairs: 115,846 valid pairs (both bands have features)
- [x] **Step 05** — Train/val/test split: 81,551 / 16,739 / 17,556 (zero leakage)
- [x] **Training** — Siamese MLP, early stopping epoch 50 (best epoch 40), val_loss=5.2109
- [x] **Step 06** — Embeddings exported: 16,079 bands × 16-dim, pgvector SQL (5.9MB)
- [x] **Security scan** — passed (torch.load weights_only, HTTPS, SQL escaping, no secrets)
- [x] **GitHub repo** — initial commit pushed

### Model Evaluation (v1)
| Metric | Siamese MLP | Random | Lift |
|--------|------------|--------|------|
| Recall@10 | 0.0168 | 0.0014 | 12.4x |
| MRR | 0.0537 | 0.0034 | 15.9x |
| Genre Purity@10 | 0.1580 | 0.0772 | 2.0x |
| Last.fm Agreement@10 | 0.0029 | 0.0005 | 5.5x |

Hit rate (≥1 correct in top 10): 11.8% vs MA, 2.8% vs Last.fm.
Main limitation: 20-dim AcousticBrainz features too narrow, embeddings tightly clustered.

### Pending
- [ ] Metaloreian integration (pgvector image swap, Go handler, migration, deploy workflow)
- [ ] GitHub Releases workflow for deployment artifacts (load_embeddings.sql + best_model.pt)
- [ ] v2: richer features (Spotify audio, genre tags, graph signal) for better recommendation quality

---

## Key Decisions & Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| Audio features source | AcousticBrainz (free dump) | Spotify deprecated; AB has 29.4M tracks, metal well-covered |
| Linkage method | MusicBrainz TSV extraction (pandas) | Bulk, clean ID match via MA URLs in MB; no PostgreSQL or fuzzy matching needed |
| Ground truth labels | MA similar artists (primary) + Last.fm (held-out eval) | MA is metal-specific and human-curated; Last.fm provides independent cross-source generalisation signal |
| Training loss | InfoNCE contrastive | Powers CLIP, SimCLR; well-understood, explainable |
| Inference architecture | Pre-computed embeddings + pgvector via Go backend | Zero GPU, no Python in prod; Go queries pgvector directly |
| Phase 2 model | Phi-3 mini + QLoRA | Fits Colab T4 free GPU; strong base model for text generation |

---

## Data Notes

### MA Dataset Columns
- `metal_bands.csv`: Band ID, Name, URL, Country, Genre, Status, Photo_URL (183k rows)
- `all_bands_discography.csv`: Album Name, Type, Year, Reviews, Band ID (636k rows)
- Reviews format: `"5 (84%)"` — parse to count + score
- Genre format: slash-separated multi-label e.g. `"Progressive Death Metal/Folk Metal"`

### Expected Audio Feature Columns (AcousticBrainz CSV dumps)
- `acousticbrainz_rhythm.csv`: mbid, bpm, danceability, onset_rate
- `acousticbrainz_tonal.csv`: mbid, key_key, key_scale, tuning_frequency
- `acousticbrainz_lowlevel.csv`: mbid, average_loudness, dynamic_complexity, mfcc_zero_mean

### MusicBrainz TSV Schema (headerless, tab-separated)
- `artist`: id(0), gid(1), name(2), sort_name(3), ... (19 cols)
- `url`: id(0), gid(1), url(2), edits_pending(3), last_updated(4)
- `l_artist_url`: id(0), link(1), entity0(2)→artist.id, entity1(3)→url.id, ... (9 cols)
- `artist_credit_name`: artist_credit(0), position(1), artist(2), name(3), join_phrase(4)
- `artist_credit`: id(0), name(1), artist_count(2), ref_count(3), created(4), edits_pending(5), gid(6)
- `recording`: id(0), gid(1), name(2), artist_credit(3), length(4), comment(5), edits_pending(6), last_updated(7), video(8)
- Step 01 join: `url` (filter metal-archives.com) → `l_artist_url.entity1` → `artist.id` via `entity0`
- Step 02 join: `artist.id` → `artist_credit_name.artist` → `recording.artist_credit` → `recording.gid` (recording MBID)
- MA band ID parsed from URL path: `.../bands/Name/12345` → `12345`

### MA Similar Artists (scraper)
- Source: MA band pages — `Similar Artists` section, user-voted
- Input: `URL` column already in `metal_bands.csv`
- Cloudflare bypass: port from Metaloreian (already battle-tested)
- Rate limit: 1–2 req/sec with randomised delays to avoid bans
- Store: `ma_similar_artists` table (band_id, similar_band_id, similarity_rank)

### Last.fm API (held-out eval)
- Endpoint: `https://ws.audioscrobbler.com/2.0/?method=artist.getSimilar&artist=<name>&api_key=<key>&format=json`
- Returns: similar artists with match score (0–1), derived from listening behaviour
- Rate limit: 5 req/sec on free tier
- Store: `lastfm_similar_artists` table (band_id, similar_band_id, match_score)
- Usage: **not used in training** — held out to measure cross-source generalisation

---

## Infrastructure

### Training
- **GPU:** CPU-only (model is 1,264 params, trains in ~2 min on CPU)
- **Framework:** Raw PyTorch (no HuggingFace)

### Serving (production)
- **Embeddings:** pgvector on existing Metaloreian PostgreSQL instance
- **API:** Go backend queries pgvector directly (no separate Python service)
- **Docker:** Swap `postgres:16-alpine` → `pgvector/pgvector:pg16` in Metaloreian docker-compose
- **Data load:** `psql $DATABASE_URL < data/processed/load_embeddings.sql`
- **Runtime compute:** zero — inference is a pgvector ANN query
- **Deployment bridge:** GitHub Releases — heavyML publishes load_embeddings.sql, Metaloreian deploy workflow downloads it
- **LLM (v2):** pre-generated descriptions cached in Postgres

---

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Required env vars (copy .env.example to .env)
LASTFM_API_KEY=...
POSTGRES_URL=...
```

