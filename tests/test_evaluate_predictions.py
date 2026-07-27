from scripts.evaluate_predictions import paired_analysis


def test_paired_case_counts_follow_metric_direction() -> None:
    rows = [
        {"model": "nnU-Net", "case_id": "a", "split": "test", "dice": 0.8, "asd_mm": 0.5},
        {"model": "nnU-Net", "case_id": "b", "split": "test", "dice": 0.7, "asd_mm": 0.7},
        {"model": "VDGR", "case_id": "a", "split": "test", "dice": 0.9, "asd_mm": 0.4},
        {"model": "VDGR", "case_id": "b", "split": "test", "dice": 0.8, "asd_mm": 0.6},
    ]

    results = paired_analysis(rows, "nnU-Net", repetitions=100, seed=1)
    by_metric = {row["metric"]: row for row in results}

    assert by_metric["dice"]["direction"] == "higher"
    assert by_metric["dice"]["improved_cases"] == 2
    assert by_metric["asd_mm"]["direction"] == "lower"
    assert by_metric["asd_mm"]["mean_delta"] < 0
    assert by_metric["asd_mm"]["improved_cases"] == 2
