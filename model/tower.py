"""Shared MLP encoder for the Siamese network.

Architecture: input (~25 dims) -> 32 (ReLU, BN, Dropout) -> 16-dim L2-normalized embedding.
Both anchors and positives use the same encoder (shared weights).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandEncoder(nn.Module):
    """MLP encoder that maps a band's audio feature vector to an embedding.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector (determined at runtime
        after preprocessing — expected ~25 after one-hot encoding key).
    hidden_dim : int
        Hidden layer width. Default 32.
    embed_dim : int
        Output embedding dimensionality. Default 16.
    dropout : float
        Dropout probability. Default 0.2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        embed_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns L2-normalized embeddings.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        Tensor of shape (batch, embed_dim), L2-normalized along dim=1.
        """
        raw = self.net(x)
        return F.normalize(raw, p=2, dim=1)
