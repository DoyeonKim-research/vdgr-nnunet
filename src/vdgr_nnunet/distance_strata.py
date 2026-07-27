from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage


BACKGROUND = 0
DEEP_INTERIOR = 1
INTERMEDIATE = 2
BOUNDARY_PROXIMAL = 3


@dataclass(frozen=True)
class DistanceStratumStats:
    lower_threshold_mm: float
    upper_threshold_mm: float
    mean_distance_mm: float
    max_distance_mm: float
    vessel_voxels: int
    deep_interior_voxels: int
    intermediate_voxels: int
    boundary_proximal_voxels: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "lower_threshold_mm": self.lower_threshold_mm,
            "upper_threshold_mm": self.upper_threshold_mm,
            "mean_distance_mm": self.mean_distance_mm,
            "max_distance_mm": self.max_distance_mm,
            "vessel_voxels": self.vessel_voxels,
            "deep_interior_voxels": self.deep_interior_voxels,
            "intermediate_voxels": self.intermediate_voxels,
            "boundary_proximal_voxels": self.boundary_proximal_voxels,
        }


def binary_foreground(
    labels: np.ndarray,
    foreground_values: Iterable[int] | None = None,
) -> np.ndarray:
    """Return a binary vessel mask from binary or multi-label annotations."""
    if foreground_values is None:
        return np.asarray(labels) > 0
    values = tuple(int(value) for value in foreground_values)
    if not values:
        raise ValueError("foreground_values must not be empty")
    return np.isin(np.asarray(labels), values)


def _validate_spacing(spacing_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(spacing_xyz) != 3:
        raise ValueError(f"Expected three spacing values, received {spacing_xyz!r}")
    spacing = tuple(float(value) for value in spacing_xyz)
    if any(not np.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError(f"Spacing must contain positive finite values: {spacing!r}")
    return spacing


def distance_strata_from_mask(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    lower_percentile: float = 33.3333,
    upper_percentile: float = 66.6667,
) -> tuple[np.ndarray, DistanceStratumStats]:
    """Partition vessel voxels by physical distance to background.

    Arrays read through SimpleITK are ordered as ``(z, y, x)``, whereas image
    spacing is reported as ``(x, y, z)``. The sampling tuple is therefore
    reversed before the Euclidean distance transform. Ties at the upper
    threshold are assigned to deep interior; ties at the lower threshold are
    assigned to boundary proximal. This reproduces the manuscript experiment.
    """
    vessel = np.asarray(mask, dtype=bool)
    if vessel.ndim != 3:
        raise ValueError(f"Expected a 3-D mask, received shape {vessel.shape}")
    spacing = _validate_spacing(spacing_xyz)
    if not (0.0 < lower_percentile < upper_percentile < 100.0):
        raise ValueError("Percentiles must satisfy 0 < lower < upper < 100")

    strata = np.zeros(vessel.shape, dtype=np.uint8)
    vessel_voxels = int(vessel.sum())
    if vessel_voxels == 0:
        stats = DistanceStratumStats(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
        return strata, stats

    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    distances = ndimage.distance_transform_edt(vessel, sampling=spacing_zyx)
    vessel_distances = distances[vessel]
    lower = float(np.percentile(vessel_distances, lower_percentile))
    upper = float(np.percentile(vessel_distances, upper_percentile))

    deep = vessel & (distances >= upper)
    intermediate = vessel & (distances < upper) & (distances > lower)
    boundary = vessel & (distances <= lower)
    strata[deep] = DEEP_INTERIOR
    strata[intermediate] = INTERMEDIATE
    strata[boundary] = BOUNDARY_PROXIMAL

    if not np.array_equal(strata > 0, vessel):
        raise RuntimeError("Distance-stratum assignment did not cover the vessel mask")

    stats = DistanceStratumStats(
        lower_threshold_mm=lower,
        upper_threshold_mm=upper,
        mean_distance_mm=float(vessel_distances.mean()),
        max_distance_mm=float(vessel_distances.max()),
        vessel_voxels=vessel_voxels,
        deep_interior_voxels=int(deep.sum()),
        intermediate_voxels=int(intermediate.sum()),
        boundary_proximal_voxels=int(boundary.sum()),
    )
    return strata, stats
