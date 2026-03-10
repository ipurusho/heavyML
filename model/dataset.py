"""PyTorch Dataset for contrastive pair sampling.

Loads the preprocessed feature matrix (numpy array keyed by band_id) and pair
CSV (band_id_a, band_id_b, score).  Each __getitem__ returns
(anchor_features, positive_features) as float32 tensors.

Negatives are handled at the loss level (in-batch negatives) so the Dataset
only needs to provide positive pairs.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Feature preprocessing helpers
# ---------------------------------------------------------------------------

# The 12 chromatic pitch classes used in AcousticBrainz tonal features
KEY_CLASSES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

# Numeric columns in the feature CSV (order matters for reproducibility)
NUMERIC_COLS = [
    "average_loudness",
    "bpm",
    "danceability",
    "dynamic_complexity",
    "mfcc_zero_mean",
    "onset_rate",
    "tuning_frequency",
]


def preprocess_features(
    features_csv: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and preprocess the feature CSV into a numeric matrix.

    Preprocessing steps:
      1. Load the CSV, index by ma_band_id.
      2. Z-score normalise the 7 numeric columns.
      3. One-hot encode key_key (12 classes).
      4. Binary-encode key_scale (major=1, minor=0).

    Returns
    -------
    band_ids : ndarray of shape (N,)
        MA band IDs in row order.
    feature_matrix : ndarray of shape (N, D), float32
        Preprocessed feature vectors.  D ~ 7 + 12 + 1 = 20.
    feature_names : list[str]
        Human-readable column names for each dimension.
    """
    df = pd.read_csv(features_csv)

    # Drop rows with missing feature values
    required = NUMERIC_COLS + ["key_key", "key_scale"]
    available = [c for c in required if c in df.columns]
    df = df.dropna(subset=available)

    band_ids = df["ma_band_id"].values.copy()

    # --- Numeric features: z-score normalise ---
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    numeric_vals = df[numeric_present].values.astype(np.float32)
    means = numeric_vals.mean(axis=0)
    stds = numeric_vals.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero
    numeric_vals = (numeric_vals - means) / stds

    feature_names: list[str] = list(numeric_present)

    # --- One-hot encode key_key ---
    if "key_key" in df.columns:
        key_vals = df["key_key"].values
        key_onehot = np.zeros((len(df), len(KEY_CLASSES)), dtype=np.float32)
        for i, k in enumerate(KEY_CLASSES):
            key_onehot[key_vals == k, i] = 1.0
        feature_names.extend([f"key_{k}" for k in KEY_CLASSES])
    else:
        key_onehot = np.zeros((len(df), 0), dtype=np.float32)

    # --- Binary encode key_scale ---
    if "key_scale" in df.columns:
        scale_vals = (df["key_scale"].str.lower() == "major").astype(np.float32).values.reshape(-1, 1)
        feature_names.append("scale_major")
    else:
        scale_vals = np.zeros((len(df), 0), dtype=np.float32)

    feature_matrix = np.hstack([numeric_vals, key_onehot, scale_vals])

    return band_ids, feature_matrix, feature_names


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SimilarBandDataset(Dataset):
    """PyTorch Dataset that yields (anchor_features, positive_features) pairs.

    Parameters
    ----------
    feature_matrix : ndarray of shape (N, D)
        Preprocessed feature vectors for all bands.
    band_ids : ndarray of shape (N,)
        Band IDs in the same row order as feature_matrix.
    pairs_csv : str or Path
        CSV with columns: band_id, similar_band_id, score.
        Only pairs where *both* bands appear in band_ids are kept.
    noise_std : float, optional
        If > 0, add Gaussian noise with this std to features as
        regularisation.  Applied independently to anchor and positive.
        Default 0.0 (no noise).
    min_score : int, optional
        Minimum similarity score to include a pair. Pairs below this
        threshold are dropped. Default 0 (keep all).
    """

    def __init__(
        self,
        feature_matrix: np.ndarray,
        band_ids: np.ndarray,
        pairs_csv: str | Path,
        noise_std: float = 0.0,
        min_score: int = 0,
    ):
        super().__init__()
        self.feature_matrix = torch.from_numpy(feature_matrix).float()
        self.noise_std = noise_std

        # Build band_id -> row index mapping
        self.id_to_idx: dict[int, int] = {
            int(bid): i for i, bid in enumerate(band_ids)
        }

        # Load pairs and filter to bands we have features for
        pairs_df = pd.read_csv(pairs_csv)
        # Handle both column naming conventions
        if "band_id_a" in pairs_df.columns:
            pairs_df = pairs_df.rename(columns={"band_id_a": "band_id", "band_id_b": "similar_band_id"})
        valid_ids = set(self.id_to_idx.keys())

        mask = (
            pairs_df["band_id"].isin(valid_ids)
            & pairs_df["similar_band_id"].isin(valid_ids)
        )
        if "score" in pairs_df.columns and min_score > 0:
            mask = mask & (pairs_df["score"] >= min_score)

        pairs_df = pairs_df[mask].reset_index(drop=True)

        if len(pairs_df) == 0:
            warnings.warn(
                "No valid pairs found after filtering! "
                "Check that pairs_csv band IDs overlap with feature matrix band IDs.",
                stacklevel=2,
            )

        self.anchor_ids = pairs_df["band_id"].values
        self.positive_ids = pairs_df["similar_band_id"].values

        n_total = mask.sum() if hasattr(mask, "sum") else len(pairs_df)
        n_dropped = len(pairs_df) if not hasattr(mask, "sum") else (~mask).sum()

    def __len__(self) -> int:
        return len(self.anchor_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_row = self.id_to_idx[int(self.anchor_ids[idx])]
        pos_row = self.id_to_idx[int(self.positive_ids[idx])]

        anchor_feat = self.feature_matrix[anchor_row].clone()
        pos_feat = self.feature_matrix[pos_row].clone()

        # Optional Gaussian noise for regularisation
        if self.noise_std > 0 and self.training_mode:
            anchor_feat += torch.randn_like(anchor_feat) * self.noise_std
            pos_feat += torch.randn_like(pos_feat) * self.noise_std

        return anchor_feat, pos_feat

    @property
    def training_mode(self) -> bool:
        """Hacky but simple: set by the training loop."""
        return getattr(self, "_training", True)

    @training_mode.setter
    def training_mode(self, value: bool) -> None:
        self._training = value

    @property
    def input_dim(self) -> int:
        """Feature dimensionality."""
        return self.feature_matrix.shape[1]
