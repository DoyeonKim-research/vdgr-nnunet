from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


STRATUM_NAMES = {
    1: "deep_interior",
    2: "intermediate",
    3: "boundary_proximal",
}


LOWER_IS_BETTER_METRICS = {
    "asd_mm",
    "hd95_mm",
    "component_count_abs_error",
    "component_delta",
    "gt_centerline_to_pred_mean_mm",
    "pred_centerline_to_gt_mean_mm",
    "gt_endpoint_to_pred_mean_mm",
    "pred_endpoint_to_gt_mean_mm",
}

DESCRIPTIVE_METRICS = {
    "pred_components",
    "gt_components",
    "component_count_bias",
    "pred_endpoints",
    "gt_endpoints",
    "gt_largest_component_fraction",
}


def metric_direction(metric: str) -> str:
    """Return the performance direction used for paired case counts."""
    if metric in LOWER_IS_BETTER_METRICS or metric.startswith(
        ("far_fp_fraction_gt_", "far_fp_voxels_gt_")
    ):
        return "lower"
    if metric in DESCRIPTIVE_METRICS:
        return "descriptive"
    return "higher"


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    coordinates = np.argwhere(mask)
    minimum = np.maximum(coordinates.min(axis=0) - 2, 0)
    maximum = np.minimum(coordinates.max(axis=0) + 3, mask.shape)
    slices = tuple(slice(int(low), int(high)) for low, high in zip(minimum, maximum))
    result = np.zeros(mask.shape, dtype=bool)
    result[slices] = skeletonize(mask[slices]).astype(bool)
    return result


def component_stats(mask: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"components": 0.0, "largest_component_fraction": float("nan")}
    structure = ndimage.generate_binary_structure(3, 3)
    labels, number = ndimage.label(mask, structure=structure)
    counts = np.bincount(labels.ravel())
    largest = float(counts[1:].max()) if counts.size > 1 else 0.0
    return {
        "components": float(number),
        "largest_component_fraction": safe_div(largest, float(mask.sum())),
    }


def surface_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if not prediction.any() or not reference.any():
        return {"asd_mm": float("nan"), "hd95_mm": float("nan")}
    structure = ndimage.generate_binary_structure(3, 1)
    prediction_surface = np.logical_xor(
        prediction,
        ndimage.binary_erosion(prediction, structure=structure, border_value=0),
    )
    reference_surface = np.logical_xor(
        reference,
        ndimage.binary_erosion(reference, structure=structure, border_value=0),
    )
    prediction_to_reference = ndimage.distance_transform_edt(
        ~reference_surface,
        sampling=spacing_zyx,
    )[prediction_surface]
    reference_to_prediction = ndimage.distance_transform_edt(
        ~prediction_surface,
        sampling=spacing_zyx,
    )[reference_surface]
    distances = np.concatenate([prediction_to_reference, reference_to_prediction])
    if distances.size == 0:
        return {"asd_mm": float("nan"), "hd95_mm": float("nan")}
    return {
        "asd_mm": float(distances.mean()),
        "hd95_mm": float(np.percentile(distances, 95)),
    }


def overlap_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    strata: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    strata = np.asarray(strata)
    true_positive = float((prediction & reference).sum())
    false_positive = float((prediction & ~reference).sum())
    false_negative = float((~prediction & reference).sum())
    prediction_count = float(prediction.sum())
    reference_count = float(reference.sum())
    union = float((prediction | reference).sum())
    metrics = {
        "dice": safe_div(2.0 * true_positive, prediction_count + reference_count),
        "iou": safe_div(true_positive, union),
        "sensitivity": safe_div(true_positive, true_positive + false_negative),
        "precision": safe_div(true_positive, true_positive + false_positive),
    }
    for value, name in STRATUM_NAMES.items():
        target = reference & (strata == value)
        metrics[f"{name}_sensitivity"] = safe_div(
            float((prediction & target).sum()),
            float(target.sum()),
        )
    metrics.update(surface_metrics(prediction, reference, spacing_zyx))
    return metrics


def _degree(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    return ndimage.convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)


def _tree(mask: np.ndarray, spacing_zyx: tuple[float, float, float]) -> cKDTree | None:
    if not mask.any():
        return None
    points = np.argwhere(mask).astype(np.float32)
    points *= np.asarray(spacing_zyx, dtype=np.float32)
    return cKDTree(points)


def _distances(
    source: np.ndarray,
    target_tree: cKDTree | None,
    spacing_zyx: tuple[float, float, float],
) -> np.ndarray:
    if not source.any() or target_tree is None:
        return np.array([], dtype=np.float32)
    points = np.argwhere(source).astype(np.float32)
    points *= np.asarray(spacing_zyx, dtype=np.float32)
    distances, _ = target_tree.query(points, k=1)
    return distances.astype(np.float32)


def topology_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    strata: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    endpoint_thresholds_mm: tuple[float, ...] = (2.0, 5.0),
    far_fp_thresholds_mm: tuple[float, ...] = (2.0, 5.0),
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    strata = np.asarray(strata)
    prediction_skeleton = skeletonize_binary(prediction)
    reference_skeleton = skeletonize_binary(reference)

    centerline_precision = safe_div(
        float((prediction_skeleton & reference).sum()),
        float(prediction_skeleton.sum()),
    )
    centerline_recall = safe_div(
        float((reference_skeleton & prediction).sum()),
        float(reference_skeleton.sum()),
    )
    cldice = (
        safe_div(
            2.0 * centerline_precision * centerline_recall,
            centerline_precision + centerline_recall,
        )
        if np.isfinite(centerline_precision) and np.isfinite(centerline_recall)
        else float("nan")
    )

    prediction_components = component_stats(prediction)
    reference_components = component_stats(reference)
    component_bias = prediction_components["components"] - reference_components["components"]
    metrics: dict[str, float] = {
        "cldice": cldice,
        "centerline_precision": centerline_precision,
        "centerline_recall": centerline_recall,
        "pred_components": prediction_components["components"],
        "gt_components": reference_components["components"],
        "component_count_bias": component_bias,
        "component_count_abs_error": abs(component_bias),
        "component_delta": abs(component_bias),
        "pred_largest_component_fraction": prediction_components[
            "largest_component_fraction"
        ],
        "gt_largest_component_fraction": reference_components[
            "largest_component_fraction"
        ],
    }

    for value, name in STRATUM_NAMES.items():
        stratum_skeleton = reference_skeleton & (strata == value)
        metrics[f"{name}_skeleton_recall"] = safe_div(
            float((prediction & stratum_skeleton).sum()),
            float(stratum_skeleton.sum()),
        )

    prediction_degree = _degree(prediction_skeleton)
    reference_degree = _degree(reference_skeleton)
    prediction_endpoints = prediction_skeleton & (prediction_degree == 1)
    reference_endpoints = reference_skeleton & (reference_degree == 1)
    prediction_tree = _tree(prediction_skeleton, spacing_zyx)
    reference_tree = _tree(reference_skeleton, spacing_zyx)
    reference_to_prediction = _distances(reference_skeleton, prediction_tree, spacing_zyx)
    prediction_to_reference = _distances(prediction_skeleton, reference_tree, spacing_zyx)
    endpoint_reference_to_prediction = _distances(
        reference_endpoints,
        prediction_tree,
        spacing_zyx,
    )
    endpoint_prediction_to_reference = _distances(
        prediction_endpoints,
        reference_tree,
        spacing_zyx,
    )
    metrics.update(
        {
            "pred_endpoints": float(prediction_endpoints.sum()),
            "gt_endpoints": float(reference_endpoints.sum()),
            "gt_centerline_to_pred_mean_mm": (
                float(reference_to_prediction.mean())
                if reference_to_prediction.size
                else float("nan")
            ),
            "pred_centerline_to_gt_mean_mm": (
                float(prediction_to_reference.mean())
                if prediction_to_reference.size
                else float("nan")
            ),
            "gt_endpoint_to_pred_mean_mm": (
                float(endpoint_reference_to_prediction.mean())
                if endpoint_reference_to_prediction.size
                else float("nan")
            ),
            "pred_endpoint_to_gt_mean_mm": (
                float(endpoint_prediction_to_reference.mean())
                if endpoint_prediction_to_reference.size
                else float("nan")
            ),
        }
    )
    for threshold in endpoint_thresholds_mm:
        suffix = f"{threshold:g}mm".replace(".", "p")
        metrics[f"gt_centerline_recall_{suffix}"] = safe_div(
            float((reference_to_prediction <= threshold).sum()),
            float(reference_to_prediction.size),
        )
        metrics[f"gt_endpoint_recall_{suffix}"] = safe_div(
            float((endpoint_reference_to_prediction <= threshold).sum()),
            float(endpoint_reference_to_prediction.size),
        )
        metrics[f"pred_endpoint_precision_{suffix}"] = safe_div(
            float((endpoint_prediction_to_reference <= threshold).sum()),
            float(endpoint_prediction_to_reference.size),
        )

    false_positive = prediction & ~reference
    label_distance = ndimage.distance_transform_edt(
        ~reference,
        sampling=spacing_zyx,
    )
    false_positive_distances = label_distance[false_positive]
    for threshold in far_fp_thresholds_mm:
        suffix = f"{threshold:g}mm".replace(".", "p")
        if false_positive_distances.size:
            far_count = float((false_positive_distances > threshold).sum())
            metrics[f"far_fp_fraction_gt_{suffix}"] = safe_div(
                far_count,
                float(false_positive_distances.size),
            )
            metrics[f"far_fp_voxels_gt_{suffix}"] = far_count
        else:
            metrics[f"far_fp_fraction_gt_{suffix}"] = 0.0
            metrics[f"far_fp_voxels_gt_{suffix}"] = 0.0
    return metrics


def all_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    strata: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> dict[str, float]:
    metrics = overlap_metrics(prediction, reference, strata, spacing_zyx)
    metrics.update(topology_metrics(prediction, reference, strata, spacing_zyx))
    return metrics
