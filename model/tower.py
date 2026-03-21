"""Shared MLP encoder for the Siamese network.

v1: Single MLP tower, input -> hidden -> embed.
v2: Two-branch architecture — audio and genre features get separate hidden
    layers before being concatenated and projected to the embedding space.
    This lets the model learn genre-specific and audio-specific representations
    independently, then fuse them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandEncoder(nn.Module):
    """Two-branch MLP encoder for band similarity.

    Branch 1 (audio): audio features (20 dims) -> 32 -> ReLU
    Branch 2 (genre): genre multi-hot (50 dims) -> 32 -> ReLU
    Fusion: concat(32+32=64) -> 64 -> ReLU -> embed_dim -> L2-normalize

    Falls back to single-branch if n_genre_dims=0 (backward compatible).

    Parameters
    ----------
    input_dim : int
        Total input dimensionality (audio + genre).
    hidden_dim : int
        Hidden layer width for each branch and fusion layer.
    embed_dim : int
        Output embedding dimensionality.
    dropout : float
        Dropout probability.
    n_audio_dims : int
        Number of audio feature dimensions (numeric + key + scale).
        Genre dims = input_dim - n_audio_dims.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        dropout: float = 0.2,
        n_audio_dims: int = 20,
    ):
        super().__init__()
        self.n_audio_dims = n_audio_dims
        n_genre_dims = input_dim - n_audio_dims

        if n_genre_dims > 0:
            # Two-branch architecture
            self.audio_branch = nn.Sequential(
                nn.Linear(n_audio_dims, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.genre_branch = nn.Sequential(
                nn.Linear(n_genre_dims, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
            )
            self._two_branch = True
        else:
            # Fallback: single branch (v1 compatible)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
            )
            self._two_branch = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns L2-normalized embeddings."""
        if self._two_branch:
            audio_feat = x[:, :self.n_audio_dims]
            genre_feat = x[:, self.n_audio_dims:]
            audio_out = self.audio_branch(audio_feat)
            genre_out = self.genre_branch(genre_feat)
            fused = torch.cat([audio_out, genre_out], dim=1)
            raw = self.fusion(fused)
        else:
            raw = self.net(x)
        return F.normalize(raw, p=2, dim=1)
