import numpy as np

from vdgr_nnunet.distance_strata import distance_strata_from_mask
from vdgr_nnunet.metrics import all_metrics, metric_direction


def test_identical_masks_have_perfect_overlap_and_topology() -> None:
    mask = np.zeros((15, 15, 15), dtype=bool)
    mask[3:12, 7, 7] = True
    mask[7, 4:11, 7] = True
    strata, _ = distance_strata_from_mask(mask, (1.0, 1.0, 1.0))
    metrics = all_metrics(mask, mask, strata, (1.0, 1.0, 1.0))

    assert metrics["dice"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["asd_mm"] == 0.0
    assert metrics["cldice"] == 1.0
    assert metrics["component_delta"] == 0.0
    assert metrics["gt_endpoint_recall_2mm"] == 1.0


def test_metric_directions_cover_reported_tradeoffs() -> None:
    assert metric_direction("dice") == "higher"
    assert metric_direction("gt_endpoint_recall_2mm") == "higher"
    assert metric_direction("asd_mm") == "lower"
    assert metric_direction("hd95_mm") == "lower"
    assert metric_direction("gt_centerline_to_pred_mean_mm") == "lower"
    assert metric_direction("far_fp_fraction_gt_5mm") == "lower"
    assert metric_direction("component_count_abs_error") == "lower"
    assert metric_direction("component_count_bias") == "descriptive"
