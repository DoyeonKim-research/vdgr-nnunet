param(
    [Parameter(Mandatory = $true)][string]$InputDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [Parameter(Mandatory = $true)][string]$NnunetResults,
    [int]$DatasetId = 102,
    [string]$Configuration = "3d_fullres",
    [int]$Fold = 0,
    [string]$Trainer = "nnUNetTrainerVDGR100epochs",
    [string]$Checkpoint = "checkpoint_final.pth"
)

$ErrorActionPreference = "Stop"
$env:nnUNet_results = (Resolve-Path -LiteralPath $NnunetResults).Path
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

nnUNetv2_predict -i $InputDir -o $OutputDir -d $DatasetId -c $Configuration -f $Fold -tr $Trainer -chk $Checkpoint
if ($LASTEXITCODE -ne 0) {
    throw "nnUNetv2_predict failed with exit code $LASTEXITCODE"
}
