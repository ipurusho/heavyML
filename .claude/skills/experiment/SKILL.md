---
name: experiment
description: >
  Run and manage heavyML training experiments with tracked hyperparameters, metrics, and
  model checkpoints. Use this skill when the user says "train the model", "run an experiment",
  "try different hyperparameters", "retrain", "tune", or wants to compare training runs.
---

# Experiment

You are running a training experiment for the heavyML Siamese MLP model. Each experiment
produces a model checkpoint, training history, and evaluation metrics that can be compared
against previous runs.

## Prerequisites

- Working directory: `/home/imman/projects/heavyML`
- Python venv: `source .venv/bin/activate`
- Feature matrix must exist: `data/processed/feature_matrix.npy`
- Pairs must exist: `data/processed/valid_pairs.csv`
- If not, run the `data-pipeline` skill first

## Running a Training Experiment

### Step 1: Split Data (if not done)
```bash
python pipeline/05_train_val_split.py
```
Splits pairs 70/15/15 by source band with genre stratification. No query leakage across splits.

**Outputs:** `data/processed/train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv`

### Step 2: Run Baseline (first time only)
```bash
python model/baseline_cosine.py
```
Cosine similarity on raw features — the floor the Siamese model must beat.

### Step 3: Train the Model
```bash
python model/train.py
```

**Default hyperparameters:**
| Param | Default | What it controls |
|-------|---------|-----------------|
| HIDDEN_DIM | 32 | MLP hidden layer width |
| EMBED_DIM | 16 | Output embedding dimensionality |
| DROPOUT | 0.2 | Regularization |
| LEARNING_RATE | 1e-3 | Adam optimizer LR |
| BATCH_SIZE | 256 | Training batch size |
| MAX_EPOCHS | 100 | Max training epochs |
| TEMPERATURE | 0.07 | InfoNCE temperature (lower = sharper) |
| NOISE_STD | 0.01 | Gaussian noise augmentation on features |
| MIN_SCORE | 0 | Minimum similarity score to include pair |

To modify hyperparameters, edit the constants at the top of `model/train.py` before running.

**Outputs:**
- `model/checkpoints/best_model.pt` (weights + metadata)
- `model/checkpoints/training_history.json` (per-epoch loss)
- `data/processed/siamese_embeddings.csv`

### Step 4: Track the Experiment

After training completes, the script prints metrics. Capture them by saving a run log:

```bash
python model/train.py 2>&1 | tee experiments/run_$(date +%Y%m%d_%H%M%S).log
```

Create `experiments/` directory if it doesn't exist. Each log captures:
- Hyperparameters used
- Per-epoch train/val loss
- Best epoch and early stopping info
- Test set metrics (Recall@10, MRR, Genre Purity, Last.fm Agreement)
- Comparison against baseline

## Comparing Experiments

When the user wants to compare runs, read the experiment logs and present a table:

```
Run       | Embed | Hidden | Temp | Recall@10 | MRR    | Genre Purity
----------|-------|--------|------|-----------|--------|-------------
baseline  | -     | -      | -    | 0.0014    | 0.0034 | 0.0772
v1 (curr) | 16    | 32     | 0.07 | 0.0168    | 0.0537 | 0.1580
v2 trial  | 32    | 64     | 0.05 | ...       | ...    | ...
```

## Hyperparameter Tuning Strategy

When the user asks to tune, try these in order (highest expected impact first):

1. **Feature input** — Adding genre tags, graph embeddings (requires data-pipeline changes)
2. **EMBED_DIM** — Try 32, 64 (currently 16 may be too compressed)
3. **HIDDEN_DIM** — Try 64, 128 (more capacity for richer features)
4. **TEMPERATURE** — Try 0.05, 0.1 (affects how sharp the similarity distribution is)
5. **BATCH_SIZE** — Larger batches = more in-batch negatives = better InfoNCE signal
6. **NOISE_STD** — Try 0.02, 0.05 (more augmentation may help with small data)

Always run the baseline comparison after training to confirm the model still beats it.

## Security Check (per experiment)

After every training run that changes code, run a security check before committing:

```bash
cd /home/imman/projects/heavyML && source .venv/bin/activate

# 1. Dependency vulnerabilities (ignore build tools: pip, setuptools, filelock)
python -m pip_audit

# 2. torch.load safety — must use weights_only=True
grep -rn 'torch.load' model/ pipeline/

# 3. No secrets in source
grep -rn -E '(password=|secret=|api_key=|AKIA|token=)' model/ pipeline/

# 4. subprocess calls use hardcoded paths only (no user input)
grep -rn 'subprocess\.\(call\|run\|Popen\)\|os\.system' model/ pipeline/

# 5. SQL escaping in generated SQL
grep -n 'escape_sql_string' pipeline/06_export_embeddings.py
```

This is lighter than the deploy-model security gate (which also checks artifacts). The goal is to catch issues before they get committed.

## Architecture Notes

- **BandEncoder:** Linear(input→32) → BatchNorm → ReLU → Dropout → Linear(32→16) → L2-norm
- **Loss:** InfoNCE with in-batch negatives (same as CLIP/SimCLR)
- **Optimizer:** Adam with CosineAnnealingLR
- **Early stopping:** Patience 10 epochs on validation loss
- **Model size:** 1,264 parameters (trains in ~2 min on CPU)
