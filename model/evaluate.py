"""Evaluation harness for sonic similarity models.

Shared between the cosine baseline and the Siamese MLP.  All functions take
numpy arrays so they are framework-agnostic.

Metrics
-------
- Recall@K: fraction of true similar bands in the top K predictions.
- Genre purity@K: fraction of top K that share primary genre with query.
- MRR (Mean Reciprocal Rank): average 1/rank of first true similar band.
- Last.fm agreement: overlap between model top 10 and Last.fm top 10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    query_ids: np.ndarray,
    predicted_ids: np.ndarray,
    true_similar: dict[int, set[int]],
    k: int = 10,
) -> float:
    """Compute mean Recall@K across all query bands.

    Parameters
    ----------
    query_ids : ndarray of shape (Q,)
        Band IDs used as queries.
    predicted_ids : ndarray of shape (Q, M) where M >= K
        For each query, the top-M predicted similar band IDs (most similar
        first).
    true_similar : dict mapping band_id -> set of true similar band_ids
        Ground truth.
    k : int
        Cutoff. Default 10.

    Returns
    -------
    float : mean recall@K across queries that have at least one ground-truth
        similar band.
    """
    recalls = []
    for i, qid in enumerate(query_ids):
        qid = int(qid)
        if qid not in true_similar or len(true_similar[qid]) == 0:
            continue
        top_k = set(int(x) for x in predicted_ids[i, :k])
        n_relevant = len(true_similar[qid])
        n_found = len(top_k & true_similar[qid])
        recalls.append(n_found / min(n_relevant, k))

    return float(np.mean(recalls)) if recalls else 0.0


def mrr(
    query_ids: np.ndarray,
    predicted_ids: np.ndarray,
    true_similar: dict[int, set[int]],
) -> float:
    """Compute Mean Reciprocal Rank.

    For each query, find the rank (1-indexed) of the first true similar band
    in the predicted list, then average 1/rank.

    Parameters
    ----------
    query_ids : ndarray of shape (Q,)
    predicted_ids : ndarray of shape (Q, M)
    true_similar : dict mapping band_id -> set of true similar band_ids

    Returns
    -------
    float : MRR across queries with ground truth.
    """
    reciprocal_ranks = []
    for i, qid in enumerate(query_ids):
        qid = int(qid)
        if qid not in true_similar or len(true_similar[qid]) == 0:
            continue
        rr = 0.0
        for rank, pid in enumerate(predicted_ids[i], start=1):
            if int(pid) in true_similar[qid]:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def genre_purity_at_k(
    query_ids: np.ndarray,
    predicted_ids: np.ndarray,
    band_genres: dict[int, str],
    k: int = 10,
) -> float:
    """Compute mean genre purity@K.

    Genre purity = fraction of top-K predicted bands that share the primary
    genre with the query band.  Primary genre is the first slash-separated
    token (e.g. "Progressive Death Metal/Folk Metal" -> "Death Metal"
    after stripping adjectives, or just the first genre token).

    Parameters
    ----------
    query_ids : ndarray of shape (Q,)
    predicted_ids : ndarray of shape (Q, M) where M >= K
    band_genres : dict mapping band_id -> raw genre string from MA
    k : int

    Returns
    -------
    float : mean genre purity@K.
    """
    purities = []
    for i, qid in enumerate(query_ids):
        qid = int(qid)
        q_genre = _primary_genre(band_genres.get(qid, ""))
        if not q_genre:
            continue
        matches = 0
        for pid in predicted_ids[i, :k]:
            pid = int(pid)
            p_genre = _primary_genre(band_genres.get(pid, ""))
            if p_genre and p_genre == q_genre:
                matches += 1
        purities.append(matches / k)

    return float(np.mean(purities)) if purities else 0.0


def _primary_genre(genre_str: str) -> str:
    """Extract the primary (base) genre from an MA genre string.

    Strategy: take the first slash-separated token, then strip common
    adjective prefixes to get the core genre.

    Examples:
      "Progressive Death Metal/Folk Metal" -> "Death Metal"
      "Thrash Metal (early); Groove Metal (later)" -> "Thrash Metal"
      "Black/Death Metal" -> "Black Metal"
    """
    if not genre_str or genre_str != genre_str:  # NaN check
        return ""
    # Handle semicolons (temporal genre changes)
    genre_str = genre_str.split(";")[0].strip()
    # Handle parenthetical qualifiers
    if "(" in genre_str:
        genre_str = genre_str.split("(")[0].strip()
    # Take first slash-separated token
    first = genre_str.split("/")[0].strip()
    # Strip common adjective prefixes
    prefixes = [
        "Progressive ", "Symphonic ", "Melodic ", "Atmospheric ",
        "Technical ", "Brutal ", "Raw ", "Epic ", "Avant-garde ",
        "Experimental ", "Post-", "Neo-", "Old School ",
    ]
    cleaned = first
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    return cleaned.strip()


def lastfm_agreement(
    query_ids: np.ndarray,
    predicted_ids: np.ndarray,
    lastfm_similar: dict[int, set[int]],
    k: int = 10,
) -> float:
    """Compute overlap between model's top K and Last.fm's top K.

    For each query band that exists in both model predictions and Last.fm data,
    compute |model_top_k intersect lastfm_top_k| / k, then average.

    Parameters
    ----------
    query_ids : ndarray of shape (Q,)
    predicted_ids : ndarray of shape (Q, M) where M >= K
    lastfm_similar : dict mapping band_id -> set of similar band_ids from Last.fm
    k : int

    Returns
    -------
    float : mean overlap fraction.
    """
    overlaps = []
    for i, qid in enumerate(query_ids):
        qid = int(qid)
        if qid not in lastfm_similar or len(lastfm_similar[qid]) == 0:
            continue
        model_top_k = set(int(x) for x in predicted_ids[i, :k])
        lfm_top_k = lastfm_similar[qid]
        # Use min(k, len(lfm_top_k)) as denominator to be fair when
        # Last.fm has fewer than k similar artists
        denom = min(k, len(lfm_top_k))
        if denom == 0:
            continue
        overlaps.append(len(model_top_k & lfm_top_k) / denom)

    return float(np.mean(overlaps)) if overlaps else 0.0


# ---------------------------------------------------------------------------
# Helpers: load ground truth data
# ---------------------------------------------------------------------------

def load_true_similar(
    pairs_csv: str | Path,
    min_score: int = 0,
) -> dict[int, set[int]]:
    """Load MA similar artist pairs into a dict for evaluation.

    Parameters
    ----------
    pairs_csv : path to CSV with columns band_id, similar_band_id, score
    min_score : minimum score threshold

    Returns
    -------
    dict mapping band_id -> set of similar band_ids
    """
    df = pd.read_csv(pairs_csv)
    # Handle both column naming conventions
    if "band_id_a" in df.columns:
        df = df.rename(columns={"band_id_a": "band_id", "band_id_b": "similar_band_id"})
    if "score" in df.columns and min_score > 0:
        df = df[df["score"] >= min_score]

    result: dict[int, set[int]] = {}
    for _, row in df.iterrows():
        bid = int(row["band_id"])
        sid = int(row["similar_band_id"])
        result.setdefault(bid, set()).add(sid)
    return result


def load_lastfm_similar(
    lastfm_csv: str | Path,
    top_k: int = 10,
) -> dict[int, set[int]]:
    """Load Last.fm similar artist data.

    Parameters
    ----------
    lastfm_csv : path to CSV with columns band_id, similar_band_id, match_score
        (or band_id, band_name, similar_name, similar_band_id, match_score)
    top_k : keep only the top K most similar per query band

    Returns
    -------
    dict mapping band_id -> set of up to top_k similar band_ids
    """
    df = pd.read_csv(lastfm_csv)

    # Drop rows where similar_band_id is NaN (Last.fm artist not matched to MA)
    df = df.dropna(subset=["similar_band_id"])
    df["similar_band_id"] = df["similar_band_id"].astype(int)
    df["band_id"] = df["band_id"].astype(int)

    # Sort by match_score descending and keep top_k per band
    if "match_score" in df.columns:
        df = df.sort_values("match_score", ascending=False)
    result: dict[int, set[int]] = {}
    for bid, group in df.groupby("band_id"):
        result[int(bid)] = set(group["similar_band_id"].head(top_k).astype(int))
    return result


def load_band_genres(
    metal_bands_csv: str | Path,
) -> dict[int, str]:
    """Load band_id -> genre string mapping from metal_bands.csv.

    Parameters
    ----------
    metal_bands_csv : path to the MA Kaggle dataset CSV

    Returns
    -------
    dict mapping band_id -> genre string
    """
    df = pd.read_csv(metal_bands_csv, low_memory=False)
    # Column names may vary; handle "Band ID" vs "band_id"
    id_col = "Band ID" if "Band ID" in df.columns else "band_id"
    genre_col = "Genre" if "Genre" in df.columns else "genre"
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    df = df.dropna(subset=[id_col])
    return dict(zip(df[id_col].astype(int), df[genre_col].fillna("")))


# ---------------------------------------------------------------------------
# Predict top-K from embeddings
# ---------------------------------------------------------------------------

def predict_top_k_from_embeddings(
    query_ids: np.ndarray,
    all_ids: np.ndarray,
    all_embeddings: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """For each query band, find the K nearest neighbours by cosine similarity.

    Parameters
    ----------
    query_ids : ndarray of shape (Q,)
        Band IDs to query.
    all_ids : ndarray of shape (N,)
        All band IDs in the embedding matrix.
    all_embeddings : ndarray of shape (N, D)
        L2-normalized embeddings.
    k : int
        Number of neighbours.

    Returns
    -------
    ndarray of shape (Q, K) — predicted similar band IDs.
    """
    # Build id -> index map
    id_to_idx = {int(bid): i for i, bid in enumerate(all_ids)}

    # Normalise embeddings (should already be, but safety)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = all_embeddings / norms

    predicted = np.zeros((len(query_ids), k), dtype=np.int64)

    for qi, qid in enumerate(query_ids):
        qid = int(qid)
        if qid not in id_to_idx:
            continue
        q_emb = normed[id_to_idx[qid]]  # (D,)
        sims = normed @ q_emb  # (N,)
        # Exclude self
        sims[id_to_idx[qid]] = -np.inf
        top_indices = np.argsort(sims)[::-1][:k]
        predicted[qi] = all_ids[top_indices]

    return predicted


# ---------------------------------------------------------------------------
# Full evaluation + comparison table
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    query_ids: np.ndarray,
    predicted_ids: np.ndarray,
    true_similar: dict[int, set[int]],
    band_genres: dict[int, str],
    lastfm_similar: Optional[dict[int, set[int]]] = None,
    k: int = 10,
) -> dict[str, float]:
    """Run all metrics and return results dict.

    Parameters
    ----------
    model_name : str, for display purposes
    query_ids : ndarray of shape (Q,)
    predicted_ids : ndarray of shape (Q, M) where M >= K
    true_similar : ground truth from MA
    band_genres : band_id -> genre string
    lastfm_similar : optional Last.fm ground truth
    k : int

    Returns
    -------
    dict with keys: recall@k, mrr, genre_purity@k, lastfm_agreement@k
    """
    results = {
        f"recall@{k}": recall_at_k(query_ids, predicted_ids, true_similar, k),
        "mrr": mrr(query_ids, predicted_ids, true_similar),
        f"genre_purity@{k}": genre_purity_at_k(
            query_ids, predicted_ids, band_genres, k
        ),
    }

    if lastfm_similar is not None:
        results[f"lastfm_agreement@{k}"] = lastfm_agreement(
            query_ids, predicted_ids, lastfm_similar, k
        )

    return results


def print_comparison_table(
    results: dict[str, dict[str, float]],
) -> None:
    """Print a clean comparison table of multiple models.

    Parameters
    ----------
    results : dict mapping model_name -> metrics_dict
        Each metrics_dict has the same keys (metric names).
    """
    if not results:
        print("No results to compare.")
        return

    # Gather all metric names
    all_metrics: list[str] = []
    for metrics in results.values():
        for m in metrics:
            if m not in all_metrics:
                all_metrics.append(m)

    # Column widths
    name_width = max(len(n) for n in results) + 2
    metric_width = max(max(len(m) for m in all_metrics) + 2, 12)

    # Header
    header = f"{'Model':<{name_width}}"
    for m in all_metrics:
        header += f"{m:>{metric_width}}"
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for name, metrics in results.items():
        row = f"{name:<{name_width}}"
        for m in all_metrics:
            val = metrics.get(m, float("nan"))
            row += f"{val:>{metric_width}.4f}"
        print(row)

    print("=" * len(header))
    print()
