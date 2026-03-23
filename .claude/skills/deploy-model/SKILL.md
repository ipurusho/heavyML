---
name: deploy-model
description: >
  Deploy heavyML model artifacts to production: export embeddings to pgvector SQL, publish
  GitHub Release, update Metaloreian. Use this skill when the user says "deploy the model",
  "publish embeddings", "create a release", "update production", "export for pgvector",
  or wants to ship a new model version to Metaloreian.
---

# Deploy Model

You are deploying a trained heavyML model to production in Metaloreian. Deployment means
pre-computing embeddings and getting them into the Metaloreian PostgreSQL instance via
pgvector. There is no Python serving layer — inference is a pgvector ANN query.

## Prerequisites

- Working directory: `/home/imman/projects/heavyML`
- Python venv: `source .venv/bin/activate`
- Trained model: `model/checkpoints/best_model.pt`
- Feature matrix: `data/processed/feature_matrix.npy`
- `gh` CLI authenticated (for GitHub Releases)
- Metaloreian repo at `/home/imman/projects/metaloreian`

## Deployment Pipeline

### Step 1: Export Embeddings

```bash
python pipeline/06_export_embeddings.py
```

Loads the best checkpoint, runs inference on all bands, generates:
- `data/processed/deployment_embeddings.csv` (band_id, band_name, emb_0…emb_15)
- `data/processed/load_embeddings.sql` (pgvector INSERT…ON CONFLICT + IVFFlat index)

Verify the output:
```bash
wc -l data/processed/deployment_embeddings.csv  # should be ~16k+
head -5 data/processed/load_embeddings.sql       # should have CREATE TABLE + INSERT
```

### Step 2: Run Final Evaluation

Before deploying, confirm the model meets quality bar. Use the `evaluate` skill or:
```bash
python model/train.py  # re-runs eval at the end
```

**Minimum bar for deployment:**
- Recall@10 > baseline cosine
- Genre Purity@10 > 2x random
- No regression vs previous deployed version

### Step 3: Security Audit

Before publishing, run a security check on all artifacts and dependencies.

```bash
cd /home/imman/projects/heavyML
source .venv/bin/activate

# 1. Dependency vulnerabilities
pip audit 2>/dev/null || pip install pip-audit && pip audit

# 2. Model loading safety — verify weights_only=True in all torch.load calls
grep -rn 'torch.load' model/ pipeline/

# 3. SQL injection in generated SQL — check for unescaped user input
# load_embeddings.sql should only contain numeric band IDs and float vectors
grep -v "^INSERT\|^CREATE\|^BEGIN\|^COMMIT\|^--\|^$" data/processed/load_embeddings.sql | head -5
# Should return nothing — only INSERTs, DDL, and comments

# 4. Secrets in artifacts — no API keys, passwords, or tokens baked in
grep -rn -E '(password|secret|api_key|AKIA|token=)' data/processed/load_embeddings.sql model/checkpoints/ 2>/dev/null

# 5. Model file integrity — verify .pt file loads cleanly with weights_only
python -c "
import torch
ckpt = torch.load('model/checkpoints/best_model.pt', weights_only=True)
print(f'Checkpoint keys: {list(ckpt.keys())}')
print('Model loads safely with weights_only=True')
"
```

**Gate check:**
- pip audit: 0 high/critical vulnerabilities
- All torch.load calls use `weights_only=True`
- Generated SQL contains only parameterized INSERTs (no dynamic strings)
- No secrets in artifacts
- Model file loads cleanly

If any check fails, fix before proceeding.

### Step 4: Publish GitHub Release

Determine the next version by checking existing releases:
```bash
gh release list --repo ipurusho/heavyML
```

Create the release with both artifacts:
```bash
gh release create v<X.Y.Z> \
  --repo ipurusho/heavyML \
  --title "v<X.Y.Z> — <brief description of what changed>" \
  --notes "<release notes with metrics table>" \
  data/processed/load_embeddings.sql \
  model/checkpoints/best_model.pt
```

Release notes should include:
- What changed (new features, hyperparameters, data)
- Metrics table (Recall@10, MRR, Genre Purity, Last.fm Agreement)
- Comparison vs previous version
- Embedding count and dimensions
- Usage instructions for Metaloreian

### Step 5: Load into Metaloreian (Local)

For local testing:
```bash
# Ensure pgvector postgres is running
cd /home/imman/projects/metaloreian
docker compose up -d postgres

# Load embeddings (drops and recreates table)
cat /home/imman/projects/heavyML/data/processed/load_embeddings.sql | \
  docker exec -i metaloreian-postgres-1 psql -U metaloreian -d metaloreian

# Verify
docker exec metaloreian-postgres-1 psql -U metaloreian -d metaloreian \
  -c "SELECT count(*) FROM band_embeddings;"
```

### Step 6: Load into Metaloreian (Production)

For production deployment, the embeddings SQL needs to be loaded on the remote VM.
This can be done via the CI/CD pipeline or manually:

```bash
# Download from GitHub Release
gh release download v<X.Y.Z> --repo ipurusho/heavyML --pattern "load_embeddings.sql" --dir /tmp

# SCP to VM and load
scp -i ~/.ssh/metaloreian-aws-deploy /tmp/load_embeddings.sql metaloreian@<VM_HOST>:~/app/
ssh -i ~/.ssh/metaloreian-aws-deploy metaloreian@<VM_HOST> \
  "cd ~/app && docker exec -i metaloreian-postgres-1 psql -U metaloreian -d metaloreian < load_embeddings.sql"
```

### Step 7: Verify Production

```bash
# Test the endpoint
curl -s https://<domain>/api/bands/3/similar | python3 -m json.tool | head -20
```

Should return 10 similar bands with scores for Blind Guardian (band ID 3).

## Versioning Convention

- **Major (v2.0.0):** New model architecture or fundamentally different features
- **Minor (v1.1.0):** New feature sources added, significant quality improvement
- **Patch (v1.0.1):** Bug fixes, re-exports with same model, data corrections

## Rollback

If the new model is worse in production:
1. Download the previous release's `load_embeddings.sql`
2. Re-load it into postgres (the SQL uses ON CONFLICT DO UPDATE, so it overwrites)
3. Verify with the API endpoint

## Checklist

Before publishing a release:
- [ ] Evaluation metrics computed and compared to previous version
- [ ] Embedding count matches expected band count
- [ ] load_embeddings.sql is syntactically valid (spot check)
- [ ] Local Metaloreian test confirms endpoint returns results
- [ ] Release notes include metrics and comparison
