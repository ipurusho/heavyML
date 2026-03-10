"""InfoNCE contrastive loss with in-batch negatives.

Given a batch of (anchor, positive) embedding pairs, every other positive in
the batch serves as a negative for each anchor.  This is the same loss used in
CLIP and SimCLR.

Numerical stability is handled by the log-sum-exp trick (subtracting the
row-wise max from logits before exponentiation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE loss with in-batch negatives.

    Parameters
    ----------
    temperature : float
        Scaling temperature. Smaller values make the distribution sharper.
        Default 0.07 (same as CLIP).
    learnable_temperature : bool
        If True, temperature is a learnable log-parameter. Default False.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
    ):
        super().__init__()
        if learnable_temperature:
            # Store as log(temperature) for unconstrained optimisation
            self.log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                "log_temperature", torch.tensor(temperature).log()
            )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(
        self,
        anchor_embeds: torch.Tensor,
        positive_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Parameters
        ----------
        anchor_embeds : Tensor of shape (B, D)
            L2-normalized anchor embeddings.
        positive_embeds : Tensor of shape (B, D)
            L2-normalized positive embeddings.  Row i is the positive pair
            for anchor i.

        Returns
        -------
        Scalar loss tensor.

        Notes
        -----
        The similarity matrix is (B, B) where entry (i, j) = cosine similarity
        between anchor_i and positive_j, scaled by 1/temperature.  The target
        is that the diagonal entry (i, i) should be the highest in each row.
        This is equivalent to a B-way classification problem with cross-entropy.
        """
        # Cosine similarity matrix scaled by temperature
        # Both inputs are already L2-normalised, so dot product = cosine sim
        logits = anchor_embeds @ positive_embeds.T  # (B, B)
        logits = logits / self.temperature  # scale

        # Labels: the diagonal is the correct positive for each anchor
        labels = torch.arange(logits.size(0), device=logits.device)

        # Cross-entropy with log-sum-exp trick (handled by PyTorch internally)
        loss = F.cross_entropy(logits, labels)

        return loss
