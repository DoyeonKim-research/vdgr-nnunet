from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch import nn
import torch.nn.functional as F
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_100epochs import (
    nnUNetTrainer_100epochs,
)
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast

from vdgr_nnunet.refinement import DistanceProbabilityGuidedRefinement


def _main_target(target):
    if isinstance(target, list):
        return [item[:, 0:1] for item in target]
    return target[:, 0:1]


def _distance_target(target):
    if isinstance(target, list):
        return [item[:, 1:2] for item in target]
    return target[:, 1:2]


def _first(value):
    return value[0] if isinstance(value, list) else value


class DistanceStratumDatasetBlosc2(nnUNetDatasetBlosc2):
    """Append a distance-stratum target channel to an nnU-Net case."""

    _mapping_cache: tuple[Path | None, dict[str, str]] | None = None

    @classmethod
    def _strata_dir(cls) -> Path:
        value = os.environ.get("VDGR_STRATA_DIR")
        if not value:
            raise RuntimeError(
                "Set VDGR_STRATA_DIR to the directory produced by "
                "scripts/generate_distance_strata.py"
            )
        directory = Path(value)
        if not directory.is_dir():
            raise FileNotFoundError(f"VDGR_STRATA_DIR does not exist: {directory}")
        return directory

    @classmethod
    def _mapping(cls) -> dict[str, str]:
        mapping_value = os.environ.get("VDGR_CASE_MAPPING")
        mapping_path = Path(mapping_value) if mapping_value else None
        if cls._mapping_cache is not None and cls._mapping_cache[0] == mapping_path:
            return cls._mapping_cache[1]
        if mapping_path is None:
            mapping: dict[str, str] = {}
        else:
            if not mapping_path.is_file():
                raise FileNotFoundError(f"VDGR_CASE_MAPPING does not exist: {mapping_path}")
            with mapping_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or "case_id" not in reader.fieldnames:
                    raise ValueError("VDGR_CASE_MAPPING must contain a case_id column")
                target_column = next(
                    (
                        name
                        for name in ("stratum_id", "target_id", "uid")
                        if name in reader.fieldnames
                    ),
                    None,
                )
                if target_column is None:
                    raise ValueError(
                        "VDGR_CASE_MAPPING must contain stratum_id, target_id, or uid"
                    )
                mapping = {
                    str(row["case_id"]): str(row[target_column])
                    for row in reader
                    if row.get("case_id") and row.get(target_column)
                }
        cls._mapping_cache = (mapping_path, mapping)
        return mapping

    def load_case(self, identifier):
        data, segmentation, segmentation_previous, properties = super().load_case(
            identifier
        )
        target_id = self._mapping().get(str(identifier), str(identifier))
        target_path = self._strata_dir() / f"{target_id}.nii.gz"
        if not target_path.is_file():
            raise FileNotFoundError(
                f"Missing distance-stratum target for {identifier}: {target_path}"
            )
        strata = sitk.GetArrayFromImage(sitk.ReadImage(str(target_path))).astype(
            np.int16
        )
        if tuple(strata.shape) != tuple(segmentation.shape[1:]):
            raise RuntimeError(
                f"Distance-stratum shape {strata.shape} does not match "
                f"nnU-Net target shape {segmentation.shape[1:]} for {identifier}"
            )
        strata = np.where(np.asarray(segmentation[0]) > 0, strata, 0).astype(np.int16)
        segmentation = np.vstack((np.asarray(segmentation), strata[None]))
        return data, segmentation, segmentation_previous, properties


class DistanceStratumDecoder(nn.Module):
    """nnU-Net decoder plus distance head and optional VDGR refinement."""

    def __init__(
        self,
        decoder: nn.Module,
        intermediate_gate_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.return_distance = False
        self.use_refinement = False
        self.hierarchy_layers = nn.ModuleList(
            [
                nn.Conv3d(
                    layer.in_channels,
                    4,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                )
                for layer in decoder.seg_layers
            ]
        )
        # Preserve the experiment key name so original checkpoints remain readable.
        self.refinement_layers = nn.ModuleList(
            [
                DistanceProbabilityGuidedRefinement(
                    layer.in_channels,
                    intermediate_gate_weight=intermediate_gate_weight,
                )
                for layer in decoder.seg_layers
            ]
        )

    @property
    def deep_supervision(self):
        return self.decoder.deep_supervision

    @deep_supervision.setter
    def deep_supervision(self, value):
        self.decoder.deep_supervision = value

    def forward(self, skips):
        low_resolution = skips[-1]
        segmentation_outputs = []
        distance_outputs = []
        for stage_index in range(len(self.decoder.stages)):
            feature = self.decoder.transpconvs[stage_index](low_resolution)
            feature = torch.cat((feature, skips[-(stage_index + 2)]), dim=1)
            feature = self.decoder.stages[stage_index](feature)
            distance_logits = self.hierarchy_layers[stage_index](feature)
            segmentation_feature = (
                self.refinement_layers[stage_index](feature, distance_logits)
                if self.use_refinement
                else feature
            )
            if self.deep_supervision or stage_index == len(self.decoder.stages) - 1:
                segmentation_outputs.append(
                    self.decoder.seg_layers[stage_index](segmentation_feature)
                )
                distance_outputs.append(distance_logits)
            low_resolution = feature

        segmentation_outputs = segmentation_outputs[::-1]
        distance_outputs = distance_outputs[::-1]
        segmentation = (
            segmentation_outputs if self.deep_supervision else segmentation_outputs[0]
        )
        if not self.return_distance:
            return segmentation
        return {
            "seg": segmentation,
            "distance": distance_outputs if self.deep_supervision else distance_outputs[0],
        }

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.decoder, name)


def _set_distance_output(network: nn.Module, enabled: bool) -> None:
    if isinstance(network.decoder, DistanceStratumDecoder):
        network.decoder.return_distance = enabled


class VDGRTrainingMixin:
    distance_loss_weight = 0.10
    distance_class_weights = (0.1, 1.0, 1.5, 2.0)
    ignore_background_in_distance_loss = True
    enable_refinement = True
    intermediate_gate_weight = 0.25

    @classmethod
    def build_network_architecture(
        cls,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        network = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        network.decoder = DistanceStratumDecoder(
            network.decoder,
            intermediate_gate_weight=cls.intermediate_gate_weight,
        )
        network.decoder.use_refinement = cls.enable_refinement
        return network

    def get_tr_and_val_datasets(self):
        training_keys, validation_keys = self.do_split()
        training = DistanceStratumDatasetBlosc2(
            self.preprocessed_dataset_folder,
            training_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )
        validation = DistanceStratumDatasetBlosc2(
            self.preprocessed_dataset_folder,
            validation_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )
        return training, validation

    def initialize(self):
        super().initialize()
        if not isinstance(self.network.decoder, DistanceStratumDecoder):
            self.network.decoder = DistanceStratumDecoder(
                self.network.decoder,
                intermediate_gate_weight=self.intermediate_gate_weight,
            ).to(self.device)
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.network.decoder.use_refinement = self.enable_refinement

    def _distance_loss(self, distance_output, target) -> torch.Tensor:
        logits = _first(distance_output)
        distance_target = _first(_distance_target(target)).long()
        main_target = _first(_main_target(target)).long()
        ignore_mask = main_target == -1
        if self.ignore_background_in_distance_loss:
            ignore_mask = torch.logical_or(ignore_mask, main_target == 0)
        distance_target = torch.where(ignore_mask, -1, distance_target)
        weights = torch.tensor(
            self.distance_class_weights,
            dtype=logits.dtype,
            device=logits.device,
        )
        return F.cross_entropy(
            logits,
            distance_target[:, 0],
            weight=weights,
            ignore_index=-1,
        )

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        _set_distance_output(self.network, True)
        try:
            context = (
                autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else dummy_context()
            )
            with context:
                output = self.network(data)
                segmentation_loss = self.loss(output["seg"], _main_target(target))
                if self.distance_loss_weight > 0:
                    auxiliary_loss = self._distance_loss(output["distance"], target)
                    loss = segmentation_loss + self.distance_loss_weight * auxiliary_loss
                else:
                    loss = segmentation_loss
        finally:
            _set_distance_output(self.network, False)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        batch["target"] = _main_target(batch["target"])
        return super().validation_step(batch)


class RetainedEpochCheckpointMixin:
    retained_checkpoint_interval = 5

    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.save_every = self.retained_checkpoint_interval

    def _retain_epoch_checkpoint(self, epoch: int, source_name: str) -> None:
        if self.local_rank != 0:
            return
        source = Path(self.output_folder) / source_name
        if not source.is_file():
            return
        target = Path(self.output_folder) / f"checkpoint_epoch_{epoch:03d}.pth"
        if not target.is_file():
            shutil.copy2(source, target)
            self.print_to_log_file(f"Retained checkpoint copy: {target.name}")

    def on_epoch_end(self):
        super().on_epoch_end()
        completed_epoch = int(self.current_epoch)
        if (
            0 < completed_epoch < int(self.num_epochs)
            and completed_epoch % self.retained_checkpoint_interval == 0
        ):
            self._retain_epoch_checkpoint(completed_epoch, "checkpoint_latest.pth")

    def on_train_end(self):
        super().on_train_end()
        final_epoch = int(self.num_epochs)
        if final_epoch % self.retained_checkpoint_interval == 0:
            self._retain_epoch_checkpoint(final_epoch, "checkpoint_final.pth")


class nnUNetTrainerVDGR100epochs(
    RetainedEpochCheckpointMixin,
    VDGRTrainingMixin,
    nnUNetTrainer_100epochs,
):
    """Full proposed VDGR-nnU-Net (lambda_dist=0.10, alpha=0.25)."""


class nnUNetTrainerDistanceSupervisionOnly100epochs(
    RetainedEpochCheckpointMixin,
    VDGRTrainingMixin,
    nnUNetTrainer_100epochs,
):
    """Direct ablation: distance-stratum auxiliary supervision only."""

    enable_refinement = False


class nnUNetTrainerRefinementOnly100epochs(
    RetainedEpochCheckpointMixin,
    VDGRTrainingMixin,
    nnUNetTrainer_100epochs,
):
    """Direct ablation: probability-guided refinement without auxiliary loss."""

    distance_loss_weight = 0.0


class nnUNetTrainerPlain100Checkpoint5(
    RetainedEpochCheckpointMixin,
    nnUNetTrainer_100epochs,
):
    """nnU-Net reference with the manuscript checkpoint schedule."""
