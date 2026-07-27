import numpy as np

from vdgr_nnunet.distance_strata import (
    BOUNDARY_PROXIMAL,
    DEEP_INTERIOR,
    INTERMEDIATE,
    distance_strata_from_mask,
)


def test_distance_strata_cover_foreground_and_preserve_background() -> None:
    mask = np.zeros((13, 13, 13), dtype=bool)
    mask[2:11, 2:11, 2:11] = True
    # Nonuniform physical spacing avoids a deliberately valid tie case in
    # which discrete cube distances leave the open middle tertile empty.
    strata, stats = distance_strata_from_mask(mask, (0.7, 0.9, 1.1))

    assert np.array_equal(strata > 0, mask)
    assert set(np.unique(strata)) == {
        0,
        DEEP_INTERIOR,
        INTERMEDIATE,
        BOUNDARY_PROXIMAL,
    }
    assert strata[6, 6, 6] == DEEP_INTERIOR
    assert strata[2, 2, 2] == BOUNDARY_PROXIMAL
    assert stats.vessel_voxels == int(mask.sum())
    assert (
        stats.deep_interior_voxels
        + stats.intermediate_voxels
        + stats.boundary_proximal_voxels
        == stats.vessel_voxels
    )


def test_empty_mask_returns_empty_target() -> None:
    strata, stats = distance_strata_from_mask(
        np.zeros((4, 5, 6), dtype=bool),
        (0.7, 0.8, 1.2),
    )
    assert not strata.any()
    assert stats.vessel_voxels == 0


def test_anisotropic_spacing_changes_physical_distance() -> None:
    mask = np.ones((5, 5, 5), dtype=bool)
    _, isotropic = distance_strata_from_mask(mask, (1.0, 1.0, 1.0))
    _, anisotropic = distance_strata_from_mask(mask, (1.0, 1.0, 2.0))
    assert anisotropic.max_distance_mm > isotropic.max_distance_mm
