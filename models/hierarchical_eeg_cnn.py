"""
HF-CNN-style hierarchical EEG CNN.

Stage 1:
    video vs smartphone

Stage 2:
    smartphone subtype classification

Input shape:
    (batch, channels, samples), e.g. (N, 64, 500)

This is closer to the HF-CNN paper than a shared-head model because:
    CNN I extracts feature maps
    CNN II receives the CNN I feature maps and further processes them
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = 2,
        dropout: float = 0.35,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=pool_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class HFCNNStyleEEG(nn.Module):
    """
    Hierarchical-flow CNN for EEG.

    CNN I:
        Learns features for broad classification:
            0 = video
            1 = smartphone

    CNN II:
        Takes feature maps from CNN I and further processes them for:
            0 = smartphone/gaming
            1 = smartphone/reading
            2 = smartphone/short_videos

    This is different from the previous shared-head model.
    Here, Stage 2 has its own CNN layers after Stage 1.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_smartphone_classes: int = 3,
        dropout: float = 0.35,
    ):
        super().__init__()

        # -------------------------
        # CNN I: broad classifier
        # -------------------------
        self.cnn1 = nn.Sequential(
            ConvBlock1D(
                in_channels=n_channels,
                out_channels=32,
                kernel_size=25,
                pool_size=2,
                dropout=dropout,
            ),
            ConvBlock1D(
                in_channels=32,
                out_channels=64,
                kernel_size=15,
                pool_size=2,
                dropout=dropout,
            ),
        )

        # Compute CNN I feature size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_samples)
            cnn1_feature_map = self.cnn1(dummy)
            cnn1_flat_dim = cnn1_feature_map.flatten(start_dim=1).shape[1]
            cnn1_out_channels = cnn1_feature_map.shape[1]

        self.stage1_fc = nn.Sequential(
            nn.Linear(cnn1_flat_dim, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.stage1_head = nn.Linear(128, 2)

        # -------------------------
        # CNN II: detailed classifier
        # -------------------------
        # CNN II receives feature maps from CNN I.
        self.cnn2 = nn.Sequential(
            ConvBlock1D(
                in_channels=cnn1_out_channels,
                out_channels=128,
                kernel_size=9,
                pool_size=2,
                dropout=dropout,
            ),
            ConvBlock1D(
                in_channels=128,
                out_channels=128,
                kernel_size=5,
                pool_size=2,
                dropout=dropout,
            ),
        )

        # Compute CNN II feature size automatically
        with torch.no_grad():
            cnn2_feature_map = self.cnn2(cnn1_feature_map)
            cnn2_flat_dim = cnn2_feature_map.flatten(start_dim=1).shape[1]

        self.stage2_fc = nn.Sequential(
            nn.Linear(cnn2_flat_dim, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.stage2_head = nn.Linear(128, n_smartphone_classes)

    def forward(self, x):
        """
        Returns:
            stage1_logits:
                shape (batch, 2)
                video vs smartphone

            stage2_logits:
                shape (batch, n_smartphone_classes)
                smartphone subtype
        """

        # CNN I feature map
        cnn1_feature_map = self.cnn1(x)

        # Stage 1 prediction
        stage1_features = cnn1_feature_map.flatten(start_dim=1)
        stage1_features = self.stage1_fc(stage1_features)
        stage1_logits = self.stage1_head(stage1_features)

        # CNN II uses CNN I feature map
        cnn2_feature_map = self.cnn2(cnn1_feature_map)

        # Stage 2 prediction
        stage2_features = cnn2_feature_map.flatten(start_dim=1)
        stage2_features = self.stage2_fc(stage2_features)
        stage2_logits = self.stage2_head(stage2_features)

        return stage1_logits, stage2_logits


def hierarchical_loss(
    stage1_logits,
    stage2_logits,
    binary_targets,
    smartphone_targets,
    stage1_weight: float = 0.5,
    stage2_weight: float = 0.5,
):
    """
    Loss for hierarchical training.

    binary_targets:
        shape (batch,)
        0 = video
        1 = smartphone

    smartphone_targets:
        shape (batch,)
        For smartphone trials:
            0 = gaming
            1 = reading
            2 = short_videos

        For video trials:
            can be any value, for example -1,
            because video trials are ignored for Stage 2.
    """

    # Stage 1 uses all samples
    loss_stage1 = F.cross_entropy(stage1_logits, binary_targets)

    # Stage 2 uses only smartphone samples
    smartphone_mask = binary_targets == 1

    if smartphone_mask.sum() > 0:
        loss_stage2 = F.cross_entropy(
            stage2_logits[smartphone_mask],
            smartphone_targets[smartphone_mask],
        )
    else:
        loss_stage2 = torch.tensor(
            0.0,
            device=stage1_logits.device,
            dtype=stage1_logits.dtype,
        )

    total_loss = stage1_weight * loss_stage1 + stage2_weight * loss_stage2

    return total_loss, loss_stage1, loss_stage2


@torch.no_grad()
def hierarchical_predict(
    stage1_logits,
    stage2_logits,
    smartphone_class_offset: int = 1,
):
    """
    Converts hierarchical outputs into final 4-class predictions.

    Final label convention:
        0 = video
        1 = smartphone/gaming
        2 = smartphone/reading
        3 = smartphone/short_videos

    smartphone_class_offset = 1 because smartphone classes start after video.
    """

    stage1_pred = stage1_logits.argmax(dim=1)
    stage2_pred = stage2_logits.argmax(dim=1)

    final_pred = torch.zeros_like(stage1_pred)

    smartphone_mask = stage1_pred == 1
    final_pred[smartphone_mask] = stage2_pred[smartphone_mask] + smartphone_class_offset

    return final_pred