# Data layout

The repository does not contain image data. The commands assume nnU-Net v2
conventions plus one directory of distance-stratum targets.

```text
<nnUNet_raw>/
  Dataset102_AirRCFullBinaryVessel/
    dataset.json
    imagesTr/
      airrc_0001_0000.nii.gz
    labelsTr/
      airrc_0001.nii.gz

<derived>/distance_strata/
  airrc_0001.nii.gz
  distance_stratum_case_stats.csv
  distance_stratum_metadata.json
```

Each stratum file must have the same voxel array shape as the segmentation
loaded by the nnU-Net data loader. The manuscript cohort did not require an
additional target resampling step. If preprocessing changes the target grid,
resample the stratum labels with nearest-neighbor interpolation and preserve
class values 0-3.

## Case IDs and mappings

By default, `VDGR_STRATA_DIR/airrc_0001.nii.gz` is matched to nnU-Net case
`airrc_0001`. `splits/airrc_case_to_luna16_series_uid.csv` provides the
one-to-one mapping between the manuscript case IDs and the public LUNA16 series
UIDs. Together with `splits/airrc_split_ids.csv`, it makes the reported
178/38/38 split identifiable in the public source collection without exposing
local paths or private clinical identifiers.

If source labels use the public series UID rather than the manuscript case ID,
pass this CSV through `VDGR_CASE_MAPPING`. A minimal custom mapping has the same
form:

```csv
case_id,uid,split
airrc_0001,1.3.6.1.4.1.example,train
```

`stratum_id`, `target_id`, or `uid` is accepted as the target identifier
column. The released mapping contains only public collection identifiers.

## Binary masks

`generate_distance_strata.py` treats all values greater than zero as vessel by
default. For a multi-label source, pass selected foreground values, for example:

```powershell
python scripts/generate_distance_strata.py `
  --label-dir <labels> `
  --output-dir <distance_strata> `
  --foreground-values 3,4
```

Physical distances are computed in millimetres using image spacing.
