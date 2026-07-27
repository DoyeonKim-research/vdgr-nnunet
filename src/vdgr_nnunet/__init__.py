"""VDGR-nnU-Net reproducibility implementation."""

from .distance_strata import (
    BOUNDARY_PROXIMAL,
    DEEP_INTERIOR,
    INTERMEDIATE,
    distance_strata_from_mask,
)
from .refinement import (
    DistanceProbabilityGuidedRefinement,
    distance_stratum_gate,
)

__all__ = [
    "BOUNDARY_PROXIMAL",
    "DEEP_INTERIOR",
    "DistanceProbabilityGuidedRefinement",
    "INTERMEDIATE",
    "distance_stratum_gate",
    "distance_strata_from_mask",
]

__version__ = "0.1.0"
