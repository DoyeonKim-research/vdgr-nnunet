# VDGR-nnU-Net

Public reference implementation of vessel distance-guided refinement
(VDGR-nnU-Net) for binary pulmonary vessel segmentation on chest CT.

## Public release scope

This repository provides a reference implementation of the proposed core
method and scripts for its primary quantitative evaluation:

- physical distance-stratum target generation;
- the distance-stratum auxiliary head and loss;
- probability-guided feature refinement;
- the proposed trainer and direct ablation configurations;
- standard segmentation, centerline, endpoint, and branch-recovery evaluation;
- the fixed manuscript cohort split and public source-series mapping; and
- environment, configuration, and focused unit tests.

The repository intentionally does not include raw CT data, annotations, trained
weights, prediction volumes, external architecture-baseline implementations,
internal experiment-management utilities, exploratory development analyses, or
figure-generation code. The omitted architecture baselines use public framework
implementations and are described in the manuscript and Supplementary
Information.

## Method summary

VDGR-nnU-Net derives deep-interior, intermediate, and boundary-proximal targets
from each binary vessel annotation using a physical distance transform. An
auxiliary head predicts these distance strata. Its soft probabilities guide
feature refinement before the final binary vessel head:

```text
G = clip(P_boundary + 0.25 * P_intermediate, 0, 1)
L = L_seg + 0.10 * L_dist
```

The distance strata are voxelwise distance-to-background classes. They should
not be interpreted as anatomical branch generations or exclusive vessel-caliber
classes.

## Installation

The reported environment used Python 3.12.7, PyTorch 2.5.1 with CUDA 12.1, and
nnU-Net v2.6.2.

```powershell
conda env create -f environment.yml
conda activate vdgr-nnunet
pip install -e .
python scripts/install_nnunet_extension.py
```

Install a CUDA-compatible PyTorch build separately when not using
`environment.yml`.

## Data

Image data are not redistributed. Obtain the AirRC annotations and corresponding
LUNA16 CT series from their public sources, then arrange the files according to
`docs/DATA_LAYOUT.md`.

The exact 178/38/38 train, validation, and test assignment is provided in
`splits/airrc_split_ids.csv`, using the split labels `train`, `val`, and `test`.
Public source-series identifiers are provided in
`splits/airrc_case_to_luna16_series_uid.csv`.

## Reproduction

See `docs/REPRODUCTION.md` for target generation, training, prediction, and
primary evaluation commands. The complete reported hyperparameters are in
`configs/paper.json` and summarized in `docs/METHOD_CONFIG.md`.

Run the focused test suite with:

```powershell
python -m pytest -q
```

## License

The code in this repository is released under the Apache License 2.0. Third-party
packages and datasets remain subject to their respective licences and terms.

## Citation

Citation metadata are provided in `CITATION.cff`. The software authors and the
five-author preferred citation for the associated manuscript are recorded
separately. Publication details and the article DOI will be added after
publication.
