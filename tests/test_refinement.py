import torch

from vdgr_nnunet.refinement import (
    DistanceProbabilityGuidedRefinement,
    distance_stratum_gate,
)


def test_distance_gate_matches_probability_equation() -> None:
    probabilities = torch.tensor([0.10, 0.20, 0.30, 0.40]).reshape(1, 4, 1, 1, 1)
    logits = probabilities.log()
    gate = distance_stratum_gate(logits, intermediate_weight=0.25)
    assert torch.allclose(gate, torch.tensor([[[[[0.475]]]]]), atol=1e-6)


def test_refinement_preserves_shape_and_gradient() -> None:
    module = DistanceProbabilityGuidedRefinement(4, intermediate_gate_weight=0.25)
    feature = torch.randn(2, 4, 11, 11, 11, requires_grad=True)
    logits = torch.randn(2, 4, 11, 11, 11, requires_grad=True)
    output = module(feature, logits)
    assert output.shape == feature.shape
    output.mean().backward()
    assert feature.grad is not None
    assert logits.grad is not None
