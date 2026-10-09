from __future__ import annotations

import torch
import torch.nn as nn
import timm


NUM_SEQUENCES = 4
NUM_LABELS = 12


class AttentionPool(nn.Module):
    """
    Attention-weighted pooling across windows or MRI sequences.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:
            (B, N, D)

        mask:
            optional (B, N)
            1 = available
            0 = missing
        """

        scores = self.score(x).squeeze(-1)

        if mask is not None:
            mask = mask.bool()

            scores = scores.masked_fill(
                ~mask,
                -1e4,
            )

        weights = torch.softmax(
            scores,
            dim=1,
        )

        if mask is not None:
            weights = (
                weights
                * mask.float()
            )

            weights = (
                weights
                / weights.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-6)
            )

        pooled = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        return pooled


class KneeAbnormalityModel(nn.Module):
    """
    2.5D multi-sequence knee MRI classifier.

    Input:
        images:
            (B, 4, W, 3, H, W)

        sequence_mask:
            (B, 4)

    Output:
        logits:
            (B, 12)
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.backbone_name = backbone_name

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.feature_dim = int(
            self.backbone.num_features
        )

        # Pool the 8 windows belonging to one MRI series.
        self.window_pool = AttentionPool(
            feature_dim=self.feature_dim,
            hidden_dim=256,
        )

        # Tell the model that axial, sagittal etc.
        # represent different anatomical views.
        self.sequence_embeddings = nn.Parameter(
            torch.zeros(
                NUM_SEQUENCES,
                self.feature_dim,
            )
        )

        nn.init.normal_(
            self.sequence_embeddings,
            mean=0.0,
            std=0.02,
        )

        # Pool information across the four MRI sequences.
        self.sequence_pool = AttentionPool(
            feature_dim=self.feature_dim,
            hidden_dim=256,
        )

        self.norm = nn.LayerNorm(
            self.feature_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.head = nn.Linear(
            self.feature_dim,
            NUM_LABELS,
        )

        # ImageNet normalization expected by pretrained
        # RGB image backbones.
        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

    def encode_images(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode 2.5D MRI windows with the image backbone.

        x:
            (N, 3, H, W)

        returns:
            (N, feature_dim)
        """

        x = (
            x - self.image_mean
        ) / self.image_std

        return self.backbone(x)

    def forward(
        self,
        images: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = images.shape[0]
        num_sequences = images.shape[1]
        num_windows = images.shape[2]

        if num_sequences != NUM_SEQUENCES:
            raise ValueError(
                f"Expected {NUM_SEQUENCES} MRI sequences, "
                f"got {num_sequences}"
            )

        # B,S,W,3,H,W -> B*S*W,3,H,W
        x = images.reshape(
            batch_size
            * num_sequences
            * num_windows,
            *images.shape[3:],
        )

        features = self.encode_images(x)

        # ->
        # B,S,W,D
        features = features.reshape(
            batch_size,
            num_sequences,
            num_windows,
            self.feature_dim,
        )

        # Pool the 8 windows belonging to each series.
        sequence_features = features.reshape(
            batch_size * num_sequences,
            num_windows,
            self.feature_dim,
        )

        sequence_features = self.window_pool(
            sequence_features
        )

        sequence_features = sequence_features.reshape(
            batch_size,
            num_sequences,
            self.feature_dim,
        )

        # Add learned identity for each sequence.
        sequence_features = (
            sequence_features
            + self.sequence_embeddings.unsqueeze(0)
        )

        # Pool the four MRI sequence representations.
        study_features = self.sequence_pool(
            sequence_features,
            mask=sequence_mask,
        )

        study_features = self.norm(
            study_features
        )

        study_features = self.dropout(
            study_features
        )

        logits = self.head(
            study_features
        )

        return logits