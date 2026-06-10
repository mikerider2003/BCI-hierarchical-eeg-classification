"""EEGNet-8,2 (Lawhern et al., 2018) baseline for the flat 4-class task.

Architecture follows the original paper: a temporal conv, a depthwise spatial
conv (one filter per channel-map, depth multiplier D), then a separable conv,
each block ending in BatchNorm -> ELU -> AvgPool -> Dropout. Max-norm constraints
on the depthwise and classifier weights are applied after each optimiser step
(see ``apply_max_norm``).

Input tensor shape: (batch, 1, n_channels, n_samples).
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_channels: int = 64,
        n_samples: int = 500,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kern_length: int = 125,   # ~half the sampling rate (250 Hz)
        dropout: float = 0.5,     # 0.5 for within-/cross-subject per the paper
    ):
        super().__init__()
        F2 = F2 if F2 is not None else F1 * D

        # Block 1: temporal conv -> depthwise spatial conv
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            # collapses the channel dim (kernel spans all n_channels), groups=F1
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: separable conv (depthwise temporal + pointwise)
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Flatten dim: n_samples downsampled by 4 then 8 = //32
        flat = F2 * (n_samples // 32)
        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x):
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @torch.no_grad()
    def apply_max_norm(self, depthwise_max: float = 1.0, dense_max: float = 0.25):
        """Clamp depthwise spatial-conv and classifier weights (original EEGNet)."""
        _renorm(self.depthwise[0].weight, depthwise_max, dim=2)
        _renorm(self.classifier.weight, dense_max, dim=0)


def _renorm(weight: torch.Tensor, max_norm: float, dim: int):
    norm = weight.norm(2, dim=dim, keepdim=True).clamp(min=1e-8)
    desired = norm.clamp(max=max_norm)
    weight.mul_(desired / norm)
