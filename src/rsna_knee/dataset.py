from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import dicom


LABEL_COLS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


SEQUENCE_COLS = [
    "axial_fluid",
    "sagittal_nonfluid",
    "coronal_fluid",
    "sagittal_fluid",
]


class KneeDataset(Dataset):
    """
    One item represents one knee MRI study.

    Output image tensor:
        (4, num_windows, 3, image_size, image_size)

    Dimensions:
        4   = canonical MRI sequences
        W   = 2.5D windows per series
        3   = previous/current/next MRI slices

    Also returns:
        targets:
            (12,)

        sequence_mask:
            (4,)
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_root: str | Path,
        image_size: int = 224,
        num_windows: int = 8,
        include_targets: bool = True,
    ):
        self.manifest = (
            manifest
            .reset_index(drop=True)
            .copy()
        )

        self.image_root = Path(image_root)

        self.image_size = int(image_size)
        self.num_windows = int(num_windows)

        self.include_targets = include_targets

    def __len__(self) -> int:
        return len(self.manifest)

    def _resize_slice(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        """
        Resize one normalized grayscale image.

        Input:
            H x W numpy array

        Output:
            image_size x image_size torch tensor
        """
        tensor = torch.from_numpy(
            image.astype(
                np.float32,
                copy=False,
            )
        )

        tensor = tensor[
            None,
            None,
            ...,
        ]

        tensor = F.interpolate(
            tensor,
            size=(
                self.image_size,
                self.image_size,
            ),
            mode="bilinear",
            align_corners=False,
        )

        return tensor[
            0,
            0,
        ]

    def _load_series(
        self,
        study_uid: str,
        series_uid: str,
    ) -> torch.Tensor:
        """
        Returns:
            (num_windows, 3, H, W)
        """
        windows = dicom.load_2p5d_windows(
            image_root=self.image_root,
            study_uid=study_uid,
            series_uid=series_uid,
            num_windows=self.num_windows,
        )

        processed_windows = []

        for window in windows:

            channels = [
                self._resize_slice(image)
                for image in window
            ]

            window_tensor = torch.stack(
                channels,
                dim=0,
            )

            processed_windows.append(
                window_tensor
            )

        return torch.stack(
            processed_windows,
            dim=0,
        )

    def _empty_series(self) -> torch.Tensor:
        """
        Placeholder for a missing MRI sequence.
        """
        return torch.zeros(
            (
                self.num_windows,
                3,
                self.image_size,
                self.image_size,
            ),
            dtype=torch.float32,
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | str]:

        row = self.manifest.iloc[index]

        study_uid = str(
            row["StudyInstanceUID"]
        )

        sequence_tensors = []
        sequence_mask = []

        for sequence_name in SEQUENCE_COLS:

            series_uid = row[sequence_name]

            if pd.isna(series_uid):
                sequence_tensors.append(
                    self._empty_series()
                )

                sequence_mask.append(0.0)

                continue

            try:
                series_tensor = self._load_series(
                    study_uid=study_uid,
                    series_uid=str(series_uid),
                )

                sequence_tensors.append(
                    series_tensor
                )

                sequence_mask.append(1.0)

            except Exception as exc:
                raise RuntimeError(
                    f"Failed loading "
                    f"study={study_uid}, "
                    f"sequence={sequence_name}, "
                    f"series={series_uid}"
                ) from exc

        images = torch.stack(
            sequence_tensors,
            dim=0,
        )

        output = {
            "StudyInstanceUID": study_uid,
            "images": images,
            "sequence_mask": torch.tensor(
                sequence_mask,
                dtype=torch.float32,
            ),
        }

        if self.include_targets:

            targets = (
                row[LABEL_COLS]
                .astype(float)
                .to_numpy(
                    dtype=np.float32
                )
            )

            output["targets"] = (
                torch.from_numpy(targets)
            )

        return output