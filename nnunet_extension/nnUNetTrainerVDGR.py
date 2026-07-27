"""nnU-Net trainer discovery shim installed by scripts/install_nnunet_extension.py."""

from vdgr_nnunet.nnunet_trainer import (
    nnUNetTrainerDistanceSupervisionOnly100epochs as _DistanceOnly,
    nnUNetTrainerPlain100Checkpoint5 as _Plain,
    nnUNetTrainerRefinementOnly100epochs as _RefinementOnly,
    nnUNetTrainerVDGR100epochs as _VDGR,
)


class nnUNetTrainerVDGR100epochs(_VDGR):
    pass


class nnUNetTrainerDistanceSupervisionOnly100epochs(_DistanceOnly):
    pass


class nnUNetTrainerRefinementOnly100epochs(_RefinementOnly):
    pass


class nnUNetTrainerPlain100Checkpoint5(_Plain):
    pass
