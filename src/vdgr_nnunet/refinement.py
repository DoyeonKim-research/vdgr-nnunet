from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def distance_stratum_gate(
    distance_logits: torch.Tensor,
    intermediate_weight: float = 0.25,
) -> torch.Tensor:
    """Return the VDGR soft gate ``clip(P_B + alpha * P_I, 0, 1)``.

    Channel order is background, deep interior, intermediate, and
    boundary proximal. These are within-vessel distance strata, not
    anatomical branch-caliber classes.
    """
    if distance_logits.ndim != 5 or distance_logits.shape[1] != 4:
        raise ValueError(
            "distance_logits must have shape (N, 4, D, H, W); "
            f"received {tuple(distance_logits.shape)}"
        )
    probabilities = torch.softmax(distance_logits, dim=1)
    intermediate = probabilities[:, 2:3]
    boundary_proximal = probabilities[:, 3:4]
    return torch.clamp(
        boundary_proximal + float(intermediate_weight) * intermediate,
        0.0,
        1.0,
    )


class DistanceProbabilityGuidedRefinement(nn.Module):
    """Refine decoder features around likely boundary-proximal vessels."""

    def __init__(self, channels: int, intermediate_gate_weight: float = 0.25) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.intermediate_gate_weight = float(intermediate_gate_weight)
        self.context_proj = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * 3, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    @staticmethod
    def weighted_local_mean(
        feature: torch.Tensor,
        weight: torch.Tensor,
        kernel_size: int,
    ) -> torch.Tensor:
        if feature.ndim != 5 or weight.ndim != 5:
            raise ValueError("feature and weight must be 5-D tensors")
        min_size = min(feature.shape[2:])
        if min_size < kernel_size:
            kernel_size = max(1, min_size if min_size % 2 == 1 else min_size - 1)
        padding = kernel_size // 2
        numerator = F.avg_pool3d(
            feature * weight,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        denominator = F.avg_pool3d(
            weight,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        ).clamp_min(1e-4)
        return numerator / denominator

    def forward(
        self,
        feature: torch.Tensor,
        distance_logits: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(distance_logits, dim=1)
        deep_interior = probabilities[:, 1:2]
        intermediate = probabilities[:, 2:3]

        local_context = self.weighted_local_mean(
            feature,
            deep_interior + 0.5 * intermediate,
            kernel_size=5,
        )
        wide_context = self.weighted_local_mean(
            feature,
            deep_interior + intermediate,
            kernel_size=9,
        )
        context = self.context_proj(torch.cat([local_context, wide_context], dim=1))

        gate = distance_stratum_gate(
            distance_logits,
            intermediate_weight=self.intermediate_gate_weight,
        )
        return self.fuse(torch.cat([feature, context * gate, feature * gate], dim=1))
