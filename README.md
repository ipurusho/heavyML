# heavyML

Machine learning pipeline for **[Metaloreian](https://github.com/ipurusho/metaloreian)** — powers the "Sonic Similar Bands" feature. Given any metal band, find the 10 most sonically similar bands using neural embeddings trained on real audio features.

## How It Works

1. **Link** 183k Metal Archives bands to MusicBrainz artist IDs
2. **Extract** audio features (BPM, energy, loudness, key, onset rate, MFCC) from AcousticBrainz
3. **Scrape** ground-truth similarity labels from Metal Archives + Last.fm
4. **Train** a siamese MLP with InfoNCE contrastive loss on audio fingerprints
5. **Export** pre-computed embeddings as pgvector-ready SQL

**Zero forward passes at query time.** All embeddings are pre-computed offline. The Go backend queries pgvector directly — no Python in production.

## Architecture

```
Band audio fingerprint (20 dims)  →  MLP (20→32→16)  →  L2-normalized 16-dim embedding
```

- **Loss:** InfoNCE (noise-contrastive estimation)
- **Positives:** Metal Archives similar artist pairs (scraped)
- **Negatives:** In-batch random
- **Eval:** Recall@10, MRR, genre purity, Last.fm agreement (held-out)
- **Inference:** pgvector ANN query from Go backend

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 00 | `pipeline/00_download_data.sh` | Download Kaggle MA dataset + MusicBrainz dump |
| 01 | `pipeline/01_mb_linkage.py` | Link MA Band IDs → MusicBrainz MBIDs |
| 02 | `pipeline/02_ab_features.py` | Extract AcousticBrainz audio features per band |
| 03 | `pipeline/03_ma_similar_scraper.py` | Scrape MA similar artists (primary ground truth) |
| 03b | `pipeline/03b_lastfm_labels.py` | Fetch Last.fm similar artists (held-out eval) |
| 04 | `pipeline/04_feature_matrix.py` | Build contrastive training pairs |
| 05 | `pipeline/05_train_val_split.py` | Stratified train/val/test split |
| — | `python -m model.train` | Train siamese MLP |
| 06 | `pipeline/06_export_embeddings.py` | Export embeddings + pgvector SQL |

## Data Sources

- **[Metal Archives](https://www.kaggle.com/datasets/guimacrlh/every-metal-archives-band-october-2024)** — 183k bands (Kaggle dump, Oct 2024)
- **[MusicBrainz](https://metabrainz.org/datasets/postgres-dumps)** — artist/URL relationship tables for ID linkage
- **[AcousticBrainz](https://acousticbrainz.org/download)** — 29.4M tracks of audio features (BPM, energy, key, loudness, MFCC)

## Deployment

The pipeline outputs `data/processed/load_embeddings.sql` — load directly into Metaloreian's PostgreSQL:

```bash
# Swap postgres image in docker-compose
# postgres:16-alpine → pgvector/pgvector:pg16

psql $DATABASE_URL < data/processed/load_embeddings.sql
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy .env.example to .env and fill in API keys
cp .env.example .env
```

## License

MIT
