import pytest


nnunet = pytest.importorskip("nnunetv2")

from vdgr_nnunet.nnunet_trainer import (  # noqa: E402
    nnUNetTrainerDistanceSupervisionOnly100epochs,
    nnUNetTrainerRefinementOnly100epochs,
    nnUNetTrainerVDGR100epochs,
)


def test_direct_ablation_configuration() -> None:
    assert nnUNetTrainerVDGR100epochs.distance_loss_weight == 0.10
    assert nnUNetTrainerVDGR100epochs.enable_refinement is True
    assert nnUNetTrainerVDGR100epochs.intermediate_gate_weight == 0.25

    assert nnUNetTrainerDistanceSupervisionOnly100epochs.distance_loss_weight == 0.10
    assert nnUNetTrainerDistanceSupervisionOnly100epochs.enable_refinement is False

    assert nnUNetTrainerRefinementOnly100epochs.distance_loss_weight == 0.0
    assert nnUNetTrainerRefinementOnly100epochs.enable_refinement is True
