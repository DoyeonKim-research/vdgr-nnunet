from scripts.evaluate_branch_recovery import analyze_paired


def test_branch_case_counts_follow_metric_direction() -> None:
    rows = [
        {
            "split": "heldout",
            "variant": "Plain",
            "case_id": "a",
            "gt_centerline_to_pred_mean_mm": 0.50,
            "gt_centerline_recall_2mm": 0.90,
        },
        {
            "split": "heldout",
            "variant": "Plain",
            "case_id": "b",
            "gt_centerline_to_pred_mean_mm": 0.70,
            "gt_centerline_recall_2mm": 0.80,
        },
        {
            "split": "heldout",
            "variant": "VDGR-nnU-Net",
            "case_id": "a",
            "gt_centerline_to_pred_mean_mm": 0.40,
            "gt_centerline_recall_2mm": 0.92,
        },
        {
            "split": "heldout",
            "variant": "VDGR-nnU-Net",
            "case_id": "b",
            "gt_centerline_to_pred_mean_mm": 0.60,
            "gt_centerline_recall_2mm": 0.85,
        },
    ]

    results = analyze_paired(
        rows,
        baseline="Plain",
        proposed="VDGR-nnU-Net",
        repetitions=100,
    )
    by_metric = {row["metric"]: row for row in results}

    distance = by_metric["gt_centerline_to_pred_mean_mm"]
    assert distance["direction"] == "lower"
    assert distance["mean_delta"] < 0
    assert distance["improved_cases"] == 2
    assert distance["worse_cases"] == 0

    recall = by_metric["gt_centerline_recall_2mm"]
    assert recall["direction"] == "higher"
    assert recall["mean_delta"] > 0
    assert recall["improved_cases"] == 2
    assert recall["worse_cases"] == 0
