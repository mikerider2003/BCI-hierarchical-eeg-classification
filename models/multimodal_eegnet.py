"""Multimodal EEGNet: EEG branch (EEGNet feature extractor) + GSR/PPG branch.

The EEG path reuses the EEGNet-8,2 convolutional stack (temporal -> depthwise
spatial -> separable) up to its flattened features, dropping EEGNet's own linear
head. The physiological path is a small 1-D CNN over the 2 aux channels
(GSR, PPG). The two embeddings are concatenated and a single linear layer maps
them to the class logits.

The same class covers the ablations via ``use_eeg`` / ``use_physio``:
    fused  : use_eeg=True,  use_physio=True   (EEG + physio)
    eeg    : use_eeg=True,  use_physio=False  (reproduces the flat EEGNet head)
    physio : use_eeg=False, use_physio=True   (GSR/PPG only)

Input shapes:
    x_eeg : (batch, 1, n_eeg_channels, n_samples)   e.g. (N, 1, 64, 500)
    x_phys: (batch, n_physio_channels, n_samples)    e.g. (N, 2, 500)
"""

import torch
import torch.nn as nn

from models.eegnet import EEGNet, _renorm


class PhysioBranch(nn.Module):
    """Compact 1-D CNN over the GSR/PPG channels."""

    def __init__(self, n_channels: int = 2, n_samples: int = 500, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=25, padding=12, bias=False),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = self.net(torch.zeros(1, n_channels, n_samples))
            self.out_dim = dummy.flatten(1).shape[1]

    def forward(self, x):  # x: (B, n_channels, n_samples)
        return self.net(x).flatten(1)


class MultimodalEEGNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_eeg_channels: int = 64,
        n_samples: int = 500,
        n_physio_channels: int = 2,
        dropout: float = 0.5,
        use_eeg: bool = True,
        use_physio: bool = True,
    ):
        super().__init__()
        if not (use_eeg or use_physio):
            raise ValueError("at least one of use_eeg / use_physio must be True")
        self.use_eeg = use_eeg
        self.use_physio = use_physio

        fusion_dim = 0
        if use_eeg:
            # EEGNet instance used purely as a feature extractor (its .classifier is unused).
            self.eeg = EEGNet(
                n_classes=n_classes, n_channels=n_eeg_channels,
                n_samples=n_samples, dropout=dropout,
            )
            fusion_dim += self.eeg.classifier.in_features
        if use_physio:
            self.physio = PhysioBranch(n_physio_channels, n_samples, dropout)
            fusion_dim += self.physio.out_dim

        self.classifier = nn.Linear(fusion_dim, n_classes)

    def forward(self, x_eeg, x_phys):
        feats = []
        if self.use_eeg:
            h = self.eeg.temporal(x_eeg)
            h = self.eeg.depthwise(h)
            h = self.eeg.separable(h)
            feats.append(torch.flatten(h, 1))
        if self.use_physio:
            feats.append(self.physio(x_phys))
        return self.classifier(torch.cat(feats, dim=1))

    @torch.no_grad()
    def apply_max_norm(self, depthwise_max: float = 1.0, dense_max: float = 0.25):
        """Max-norm on the EEG depthwise filters and the fusion classifier (as in EEGNet)."""
        if self.use_eeg:
            _renorm(self.eeg.depthwise[0].weight, depthwise_max, dim=2)
        _renorm(self.classifier.weight, dense_max, dim=0)
