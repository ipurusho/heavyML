---
name: evaluate
description: >
  Evaluate heavyML model performance: compute metrics, compare models, analyze errors, generate
  reports. Use this skill when the user says "evaluate the model", "how good is it", "compare
  models", "check metrics", "run eval", "is v2 better", or wants to understand model quality.
---

# Evaluate

You are evaluating the heavyML band similarity model. Evaluation answers one question:
"Are the recommendations good, and are they getting better?"

## Prerequisites

- Working directory: `/home/imman/projects/heavyML`
- Python venv: `source .venv/bin/activate`
- Trained model: `model/checkpoints/best_model.pt`
- Test pairs: `data/processed/test_pairs.csv`
- Ground truth: `data/processed/ma_similar_artists.csv`
- Last.fm eval data: `data/processed/lastfm_similar_artists.csv` (optional)
- Band genres: `data/raw/metal_bands.csv`

## Metrics

| Metric | What it measures | Good target |
|--------|-----------------|-------------|
| **Recall@10** | % of true similar bands in top 10 results | >10% (v2 target) |
| **MRR** | Mean reciprocal rank of first correct result | >0.1 |
| **Genre Purity@10** | % of top 10 sharing primary genre with query | >50% |
| **Last.fm Agreement@10** | % of top 10 that Last.fm also considers similar | >5% |
| **Hit Rate@10** | % of queries with at least 1 correct in top 10 | >30% |

## Running Evaluation

### Full Evaluation (uses evaluate.py harness)

The training script (`model/train.py`) runs evaluation automatically at the end. To re-run
evaluation independently on existing embeddings:

```python
# In Python or a notebook:
from model.evaluate import evaluate_model, print_comparison_table

results = evaluate_model(
    embeddings_csv='data/processed/siamese_embeddings.csv',
    test_pairs_csv='data/processed/test_pairs.csv',
    metal_bands_csv='data/raw/metal_bands.csv',
    lastfm_csv='data/processed/lastfm_similar_artists.csv',
    k=10
)
print_comparison_table(results)
```

### Baseline Comparison

Always compare against the cosine baseline:
```bash
python model/baseline_cosine.py
```

### Quick Spot Check

For a fast sanity check on a specific band:
```bash
python -c "
import pandas as pd
import numpy as np

emb = pd.read_csv('data/processed/siamese_embeddings.csv')
# or deployment_embeddings.csv

band_name = 'Blind Guardian'
row = emb[emb['band_name'] == band_name]
if row.empty:
    print(f'{band_name} not in embeddings')
else:
    band_vec = row.iloc[0, 2:].values.astype(float)
    all_vecs = emb.iloc[:, 2:].values.astype(float)
    sims = all_vecs @ band_vec  # cosine sim (already L2-normed)
    top_idx = np.argsort(-sims)[1:11]  # skip self
    for i in top_idx:
        print(f'{emb.iloc[i][\"band_name\"]:30s} {sims[i]:.4f}')
"
```

## Evaluation Report Format

When presenting results, use this format:

```
# Evaluation Report — heavyML [version]

Date: [date]
Model: [architecture summary]
Features: [input dimensions and types]
Training: [epochs, best epoch, val_loss]

## Metrics (Test Set)
| Metric           | Model  | Baseline | Random | Lift vs Random |
|-----------------|--------|----------|--------|----------------|
| Recall@10       | x.xxxx | x.xxxx   | x.xxxx | xx.x x         |
| MRR             | x.xxxx | x.xxxx   | x.xxxx | xx.x x         |
| Genre Purity@10 | x.xxxx | x.xxxx   | x.xxxx | xx.x x         |
| Last.fm Agree   | x.xxxx | x.xxxx   | x.xxxx | xx.x x         |
| Hit Rate@10     | xx.x%  | xx.x%    | xx.x%  | -              |

## Sample Predictions
[Show top 10 similar bands for 3-5 well-known bands as spot checks]

## Error Analysis
[Identify patterns in failures — e.g., cross-genre confusion, rare subgenre problems]

## Recommendations
[What to try next to improve metrics]
```

## Error Analysis Checklist

When metrics are poor, investigate these in order:

1. **Feature coverage** — How many test bands have embeddings? Low coverage = selection bias.
2. **Genre confusion** — Are recommendations crossing genre boundaries? Check genre purity.
3. **Embedding clustering** — Are embeddings too tightly clustered? Check variance of distances.
4. **Training data quality** — Are MA similarity labels noisy? Spot-check random pairs.
5. **Feature informativeness** — Which features have highest variance? Low-variance features add noise.
6. **Leakage check** — Any query bands appearing in training set? Should be zero.

## Comparing Two Models

When comparing v1 vs v2 (or any two checkpoints):

1. Load both embedding sets
2. Run `evaluate_model()` on each with the same test set
3. Present side-by-side metrics table
4. Show spot-check comparisons for 3-5 bands
5. Statistical significance: if metric differences are <1%, they may be noise
