# Cohort split IDs

`airrc_split_ids.csv` contains 254 nnU-Net case IDs:

- 178 training cases;
- 38 validation cases;
- 38 reserved held-out test cases.

`airrc_case_to_luna16_series_uid.csv` maps every case ID to its public LUNA16
series UID. No local image path, label path, scanner metadata, or private
patient identifier is included. `test` corresponds to `heldout_test` in frozen
result tables.
