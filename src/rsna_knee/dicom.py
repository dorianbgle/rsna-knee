from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pydicom


CANONICAL_SEQUENCES = {
    "axial_fluid": {
        "Anatomical_Plane": "Axial",
        "Fluid_Sensitive": 1,
        "Fat_Suppression": 1,
    },
    "sagittal_nonfluid": {
        "Anatomical_Plane": "Sagittal",
        "Fluid_Sensitive": 0,
        "Fat_Suppression": 0,
    },
    "coronal_fluid": {
        "Anatomical_Plane": "Coronal",
        "Fluid_Sensitive": 1,
        "Fat_Suppression": 1,
    },
    "sagittal_fluid": {
        "Anatomical_Plane": "Sagittal",
        "Fluid_Sensitive": 1,
        "Fat_Suppression": 1,
    },
}


def get_series_directory(
    image_root: Path,
    study_uid: str,
    series_uid: str,
) -> Path:
    """
    Return the on-disk directory for one MRI series.
    """
    path = image_root / str(study_uid) / str(series_uid)

    if not path.exists():
        raise FileNotFoundError(
            f"Series directory not found: {path}"
        )

    return path


def list_dicom_files(series_dir: Path) -> list[Path]:
    """
    Return files in a series directory that can be read as DICOM.

    RSNA datasets may contain .dcm files or DICOM files without an
    extension, so we do not rely exclusively on the suffix.
    """
    files = [
        p
        for p in series_dir.iterdir()
        if p.is_file()
    ]

    return sorted(files)


def _read_header(path: Path):
    return pydicom.dcmread(
        path,
        stop_before_pixels=True,
        force=True,
    )


def _slice_coordinate(ds) -> float | None:
    """
    Calculate physical slice position using ImageOrientationPatient
    and ImagePositionPatient.

    This is more reliable than filename ordering and usually more
    reliable than InstanceNumber alone.
    """
    orientation = ds.get("ImageOrientationPatient")
    position = ds.get("ImagePositionPatient")

    if orientation is None or position is None:
        return None

    try:
        orientation = np.asarray(
            orientation,
            dtype=np.float64,
        )

        position = np.asarray(
            position,
            dtype=np.float64,
        )

        row_cosines = orientation[:3]
        col_cosines = orientation[3:]

        slice_normal = np.cross(
            row_cosines,
            col_cosines,
        )

        return float(
            np.dot(position, slice_normal)
        )

    except Exception:
        return None


def sort_dicom_files(paths: Iterable[Path]) -> list[Path]:
    """
    Sort slices anatomically where possible.

    Priority:
    1. physical slice coordinate
    2. InstanceNumber
    3. filename
    """
    records = []

    for path in paths:
        try:
            ds = _read_header(path)

            coordinate = _slice_coordinate(ds)

            instance_number = ds.get(
                "InstanceNumber"
            )

            if instance_number is not None:
                try:
                    instance_number = int(
                        instance_number
                    )
                except Exception:
                    instance_number = None

            records.append(
                (
                    path,
                    coordinate,
                    instance_number,
                )
            )

        except Exception:
            records.append(
                (
                    path,
                    None,
                    None,
                )
            )

    if not records:
        return []

    physical_positions = [
        r[1]
        for r in records
        if r[1] is not None
    ]

    if len(physical_positions) >= max(
        2,
        int(len(records) * 0.8),
    ):
        records.sort(
            key=lambda r: (
                r[1] is None,
                r[1]
                if r[1] is not None
                else float("inf"),
            )
        )

    else:
        records.sort(
            key=lambda r: (
                r[2] is None,
                r[2]
                if r[2] is not None
                else float("inf"),
                r[0].name,
            )
        )

    return [
        r[0]
        for r in records
    ]


def count_slices(
    image_root: Path,
    study_uid: str,
    series_uid: str,
) -> int:
    """
    Count files in a particular MRI series.
    """
    try:
        series_dir = get_series_directory(
            image_root=image_root,
            study_uid=study_uid,
            series_uid=series_uid,
        )

        return len(
            list_dicom_files(series_dir)
        )

    except FileNotFoundError:
        return 0


def select_canonical_series(
    study_uid: str,
    series_table: pd.DataFrame,
    image_root: Path,
) -> dict[str, str | None]:
    """
    Select one series for each of the four canonical MRI inputs.

    If multiple candidate series exist for the same sequence type,
    choose the series containing the largest number of slices.

    Returns:
        {
            "axial_fluid": SeriesInstanceUID | None,
            "sagittal_nonfluid": ...,
            "coronal_fluid": ...,
            "sagittal_fluid": ...,
        }
    """
    study_series = series_table.loc[
        series_table["StudyInstanceUID"].astype(str)
        == str(study_uid)
    ].copy()

    selections: dict[str, str | None] = {}

    for name, criteria in CANONICAL_SEQUENCES.items():

        candidates = study_series.copy()

        for column, value in criteria.items():
            candidates = candidates.loc[
                candidates[column] == value
            ]

        if candidates.empty:
            selections[name] = None
            continue

        candidates = candidates.copy()

        candidates["slice_count"] = candidates[
            "SeriesInstanceUID"
        ].map(
            lambda series_uid: count_slices(
                image_root=image_root,
                study_uid=study_uid,
                series_uid=str(series_uid),
            )
        )

        candidates = candidates.sort_values(
            [
                "slice_count",
                "SeriesInstanceUID",
            ],
            ascending=[
                False,
                True,
            ],
        )

        selections[name] = str(
            candidates.iloc[0]["SeriesInstanceUID"]
        )

    return selections


def normalize_image(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """
    Robust intensity normalization for MRI.

    Output range: [0, 1]
    """
    image = image.astype(
        np.float32,
        copy=False,
    )

    finite = np.isfinite(image)

    if not finite.any():
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    values = image[finite]

    low = np.percentile(
        values,
        lower_percentile,
    )

    high = np.percentile(
        values,
        upper_percentile,
    )

    if high <= low:
        low = float(values.min())
        high = float(values.max())

    if high <= low:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    image = np.clip(
        image,
        low,
        high,
    )

    image = (
        image - low
    ) / (
        high - low
    )

    return image.astype(
        np.float32,
        copy=False,
    )


def load_slice(
    path: Path,
) -> np.ndarray:
    """
    Decode and normalize one DICOM MRI slice.
    """
    ds = pydicom.dcmread(
        path,
        force=True,
    )

    image = ds.pixel_array.astype(
        np.float32
    )

    slope = float(
        ds.get(
            "RescaleSlope",
            1.0,
        )
    )

    intercept = float(
        ds.get(
            "RescaleIntercept",
            0.0,
        )
    )

    image = (
        image * slope
        + intercept
    )

    if ds.get(
        "PhotometricInterpretation"
    ) == "MONOCHROME1":
        image = image.max() - image

    return normalize_image(image)


def evenly_sample_indices(
    length: int,
    count: int,
) -> np.ndarray:
    """
    Select approximately evenly spaced slices throughout a series.
    """
    if length <= 0:
        raise ValueError(
            "Series contains no slices."
        )

    if count <= 0:
        raise ValueError(
            "count must be > 0."
        )

    if length == 1:
        return np.zeros(
            count,
            dtype=np.int64,
        )

    return np.linspace(
        0,
        length - 1,
        count,
    ).round().astype(
        np.int64
    )


def load_sampled_series(
    image_root: Path,
    study_uid: str,
    series_uid: str,
    num_slices: int = 16,
) -> np.ndarray:
    """
    Load a fixed number of slices from one MRI series.

    Returns:
        np.ndarray with shape:
            (num_slices, height, width)
    """
    series_dir = get_series_directory(
        image_root=image_root,
        study_uid=study_uid,
        series_uid=series_uid,
    )

    files = list_dicom_files(
        series_dir
    )

    files = sort_dicom_files(
        files
    )

    if not files:
        raise RuntimeError(
            f"No DICOM files found in {series_dir}"
        )

    indices = evenly_sample_indices(
        len(files),
        num_slices,
    )

    images = [
        load_slice(
            files[index]
        )
        for index in indices
    ]

    shapes = {
        image.shape
        for image in images
    }

    if len(shapes) != 1:
        raise ValueError(
            f"Inconsistent slice dimensions "
            f"in series {series_uid}: {shapes}"
        )

    return np.stack(
        images,
        axis=0,
    )