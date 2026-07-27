# Primary reproduction workflow

This public package covers the proposed VDGR-nnU-Net method and the primary
quantitative evaluation. It assumes that the public source data have already
been downloaded and converted to the nnU-Net v2 layout described in
`DATA_LAYOUT.md`.

## 1. Install the package and trainer shim

```powershell
conda env create -f environment.yml
conda activate vdgr-nnunet
pip install -e .
python scripts/install_nnunet_extension.py
```

## 2. Generate distance-stratum targets

```powershell
python scripts/generate_distance_strata.py `
  --label-dir <binary_labels> `
  --output-dir <distance_strata>
```

For multi-label source masks, specify the foreground values:

```powershell
python scripts/generate_distance_strata.py `
  --label-dir <source_labels> `
  --output-dir <distance_strata> `
  --foreground-values 3,4
```

The command writes one stratum image per case, a case-level summary CSV, and
metadata describing the percentile thresholds.

## 3. Train VDGR-nnU-Net

```powershell
powershell -File scripts/train_vdgr.ps1 `
  -NnunetPreprocessed <nnUNet_preprocessed> `
  -NnunetResults <nnUNet_results> `
  -StrataDir <distance_strata>
```

If the stratum filenames use public source UIDs instead of manuscript case IDs,
also provide:

```text
-CaseMapping splits/airrc_case_to_luna16_series_uid.csv
```

The default trainer is `nnUNetTrainerVDGR100epochs`. Direct ablations can be
selected with:

- `nnUNetTrainerDistanceSupervisionOnly100epochs`
- `nnUNetTrainerRefinementOnly100epochs`
- `nnUNetTrainerPlain100Checkpoint5`

## 4. Generate predictions

```powershell
powershell -File scripts/predict_vdgr.ps1 `
  -InputDir <images> `
  -OutputDir <predictions> `
  -NnunetResults <nnUNet_results>
```

## 5. Compute primary metrics

```powershell
python scripts/evaluate_predictions.py `
  --case-list splits/airrc_split_ids.csv `
  --split test `
  --label-dir <binary_references> `
  --strata-dir <distance_strata> `
  --prediction "nnU-Net=<nnunet_predictions>" `
  --prediction "VDGR-nnU-Net=<vdgr_predictions>" `
  --reference-model "nnU-Net" `
  --out-dir <standard_metrics>
```

## 6. Compute endpoint and branch-recovery metrics

```powershell
python scripts/evaluate_branch_recovery.py `
  --case-list splits/airrc_split_ids.csv `
  --split test `
  --label-dir <binary_references> `
  --prediction "nnU-Net=<nnunet_predictions>" `
  --prediction "VDGR-nnU-Net=<vdgr_predictions>" `
  --out-dir <branch_metrics>
```

Use `--help` on each Python script for all accepted options. The evaluation
scripts check image shape, spacing, origin, and direction before computing
metrics.

The paired-analysis command adjusts P values over the models and metrics
supplied in that run. The manuscript used a larger, conservatively retained
family of completed screening configurations. Therefore, a primary-pair-only
run can reproduce the case metrics and raw paired comparisons but need not
produce the manuscript's exact FDR-adjusted values. The frozen adjustment
context and reported adjusted values are provided with the article as
Supplementary Data.

## Scope boundary

This repository does not reproduce external architecture-baseline training,
internal experiment scheduling, exploratory development analyses, or manuscript
figure rendering. Those components are not part of the public method
implementation.
