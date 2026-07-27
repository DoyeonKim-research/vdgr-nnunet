from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class ImageGeometry:
    size_xyz: tuple[int, ...]
    spacing_xyz: tuple[float, ...]
    origin_xyz: tuple[float, ...]
    direction: tuple[float, ...]


def image_geometry(image: sitk.Image) -> ImageGeometry:
    return ImageGeometry(
        size_xyz=tuple(int(value) for value in image.GetSize()),
        spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
        origin_xyz=tuple(float(value) for value in image.GetOrigin()),
        direction=tuple(float(value) for value in image.GetDirection()),
    )


def read_image(
    path: Path,
) -> tuple[np.ndarray, tuple[float, float, float], ImageGeometry]:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D image, found {image.GetDimension()}D: {path}")
    array = sitk.GetArrayFromImage(image)
    spacing_xyz = image.GetSpacing()
    spacing_zyx = tuple(float(value) for value in spacing_xyz[::-1])
    return array, spacing_zyx, image_geometry(image)


def geometry_differences(
    reference: ImageGeometry,
    candidate: ImageGeometry,
    *,
    absolute_tolerance: float = 1e-6,
) -> list[str]:
    differences: list[str] = []
    if candidate.size_xyz != reference.size_xyz:
        differences.append(
            f"size {candidate.size_xyz} != reference {reference.size_xyz}"
        )
    for name, candidate_values, reference_values in (
        ("spacing", candidate.spacing_xyz, reference.spacing_xyz),
        ("origin", candidate.origin_xyz, reference.origin_xyz),
        ("direction", candidate.direction, reference.direction),
    ):
        if len(candidate_values) != len(reference_values) or not np.allclose(
            candidate_values,
            reference_values,
            rtol=0.0,
            atol=absolute_tolerance,
        ):
            differences.append(
                f"{name} {candidate_values} != reference {reference_values}"
            )
    return differences


def require_matching_geometry(
    reference: ImageGeometry,
    candidate: ImageGeometry,
    *,
    context: str,
    absolute_tolerance: float = 1e-6,
) -> None:
    differences = geometry_differences(
        reference,
        candidate,
        absolute_tolerance=absolute_tolerance,
    )
    if differences:
        details = "; ".join(differences)
        raise ValueError(f"Image geometry mismatch for {context}: {details}")
