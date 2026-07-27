param(
    [Parameter(Mandatory = $true)][string]$NnunetPreprocessed,
    [Parameter(Mandatory = $true)][string]$NnunetResults,
    [Parameter(Mandatory = $true)][string]$StrataDir,
    [string]$CaseMapping = "",
    [int]$DatasetId = 102,
    [string]$Configuration = "3d_fullres",
    [int]$Fold = 0,
    [string]$Trainer = "nnUNetTrainerVDGR100epochs"
)

$ErrorActionPreference = "Stop"
$env:nnUNet_preprocessed = (Resolve-Path -LiteralPath $NnunetPreprocessed).Path
$env:nnUNet_results = (Resolve-Path -LiteralPath $NnunetResults).Path
$env:VDGR_STRATA_DIR = (Resolve-Path -LiteralPath $StrataDir).Path
if ($CaseMapping) {
    $env:VDGR_CASE_MAPPING = (Resolve-Path -LiteralPath $CaseMapping).Path
}

nnUNetv2_train $DatasetId $Configuration $Fold -tr $Trainer
if ($LASTEXITCODE -ne 0) {
    throw "nnUNetv2_train failed with exit code $LASTEXITCODE"
}
