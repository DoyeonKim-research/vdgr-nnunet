# Cohort split IDs

`airrc_split_ids.csv` contains 254 nnU-Net case IDs:

- 178 training cases;
- 38 validation cases (`val`);
- 38 test cases (`test`).

`airrc_case_to_luna16_series_uid.csv` maps every case ID to its public LUNA16
series UID. No local image path, label path, scanner metadata, or private
patient identifier is included. The CSV uses the exact split labels `train`,
`val`, and `test`. The `test` label corresponds to `heldout_test` in frozen
internal result tables.
