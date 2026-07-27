from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from vdgr_nnunet.image_geometry import read_image, require_matching_geometry
from vdgr_nnunet.metrics import metric_direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate branch recovery with GT-centerline caliber "
            "definitions and 25/50/75 percent coverage sensitivity."
        )
    )
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--case-limit", type=int, default=38)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Prediction source as name=directory. The first source is the baseline.",
    )
    parser.add_argument(
        "--topology-cache-dir",
        default=None,
        help="Optional cache containing <case_id>_label_topology.npz files.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
    )
    parser.add_argument("--distance-threshold-mm", type=float, default=2.0)
    parser.add_argument("--coverage-thresholds", default="0.25,0.50,0.75")
    parser.add_argument("--caliber-quantiles", default="0.20,0.333333")
    parser.add_argument("--branch-min-voxels", type=int, default=3)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    return parser.parse_args()


def parse_prediction_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs = values
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Prediction spec must be name=directory, got {spec!r}")
        name, path_text = spec.split("=", 1)
        name = name.strip()
        path = Path(path_text.strip())
        if not name:
            raise ValueError(f"Prediction name is empty in {spec!r}")
        if not path.exists():
            raise FileNotFoundError(path)
        parsed.append((name, path))
    if len(parsed) < 2:
        raise ValueError("At least two prediction sources are required.")
    return parsed


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one numeric value is required.")
    return values


def load_cases(case_list: Path, split: str, case_limit: int) -> pd.DataFrame:
    frame = pd.read_csv(case_list)
    if not {"case_id", "split"}.issubset(frame.columns):
        raise ValueError("Case list must contain case_id and split columns")
    frame = frame[frame["split"].astype(str).str.lower() == split.lower()].copy()
    frame = frame.sort_values("case_id")
    if case_limit > 0:
        frame = frame.head(case_limit)
    return frame


def crop_slices(mask: np.ndarray, margin: int = 4) -> tuple[slice, slice, slice]:
    if not mask.any():
        return tuple(slice(0, size) for size in mask.shape)  # type: ignore[return-value]
    coords = np.argwhere(mask)
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin + 1, mask.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))  # type: ignore[return-value]


def skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    coords = np.argwhere(mask)
    mins = np.maximum(coords.min(axis=0) - 2, 0)
    maxs = np.minimum(coords.max(axis=0) + 3, mask.shape)
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))
    result = np.zeros(mask.shape, dtype=bool)
    result[slices] = skeletonize(mask[slices]).astype(bool)
    return result


def load_cached_gt_skeleton(
    case_id: str,
    cache_dir: Path | None,
    label_fg: np.ndarray,
) -> np.ndarray:
    cache_path = cache_dir / f"{case_id}_label_topology.npz" if cache_dir else None
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as data:
            return data["gt_skeleton"].astype(bool)
    return skeletonize_binary(label_fg)


def skeleton_degree(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    return ndimage.convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)


def point_tree(mask: np.ndarray, spacing: tuple[float, float, float]) -> cKDTree | None:
    if not mask.any():
        return None
    points = np.argwhere(mask).astype(np.float32)
    points *= np.asarray(spacing, dtype=np.float32)
    return cKDTree(points)


def distances_to_tree(
    mask: np.ndarray,
    tree: cKDTree | None,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    if not mask.any() or tree is None:
        return np.array([], dtype=np.float32)
    points = np.argwhere(mask).astype(np.float32)
    points *= np.asarray(spacing, dtype=np.float32)
    distances, _ = tree.query(points, k=1)
    return distances.astype(np.float32)


def quantile_suffix(quantile: float) -> str:
    return f"q{int(round(quantile * 100)):02d}"


def coverage_suffix(threshold: float) -> str:
    return f"cov{int(round(threshold * 100)):02d}"


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def build_branch_definition(
    label_fg: np.ndarray,
    gt_skeleton: np.ndarray,
    spacing: tuple[float, float, float],
    caliber_quantiles: list[float],
    branch_min_voxels: int,
) -> dict[str, object]:
    radius_mm = ndimage.distance_transform_edt(label_fg, sampling=spacing).astype(np.float32)
    degree = skeleton_degree(gt_skeleton)
    endpoint_mask = gt_skeleton & (degree == 1)
    skeleton_without_junctions = gt_skeleton & (degree < 3)
    structure = ndimage.generate_binary_structure(3, 3)
    branch_labels, _ = ndimage.label(skeleton_without_junctions, structure=structure)
    counts = np.bincount(branch_labels.ravel())
    branch_ids = np.asarray(
        [index for index in range(1, len(counts)) if counts[index] >= branch_min_voxels],
        dtype=np.int32,
    )
    branch_sizes = counts[branch_ids].astype(np.float64) if branch_ids.size else np.array([], dtype=float)

    if branch_ids.size:
        branch_median_radius = np.asarray(
            ndimage.median(radius_mm, labels=branch_labels, index=branch_ids),
            dtype=np.float64,
        )
        endpoint_counts = np.asarray(
            ndimage.sum(endpoint_mask, labels=branch_labels, index=branch_ids),
            dtype=np.float64,
        )
        terminal = endpoint_counts > 0
    else:
        branch_median_radius = np.array([], dtype=float)
        terminal = np.array([], dtype=bool)

    groups: dict[str, np.ndarray] = {
        "all": np.ones(branch_ids.size, dtype=bool),
        "terminal": terminal,
    }
    caliber_thresholds: dict[str, float] = {}
    for quantile in caliber_quantiles:
        suffix = quantile_suffix(quantile)
        if branch_median_radius.size:
            threshold = float(np.quantile(branch_median_radius, quantile))
            small = branch_median_radius <= threshold
        else:
            threshold = float("nan")
            small = np.zeros(branch_ids.size, dtype=bool)
        caliber_thresholds[suffix] = threshold
        groups[f"small_{suffix}"] = small
        groups[f"terminal_small_{suffix}"] = terminal & small

    return {
        "radius_mm": radius_mm,
        "branch_labels": branch_labels,
        "branch_ids": branch_ids,
        "branch_sizes": branch_sizes,
        "branch_median_radius": branch_median_radius,
        "groups": groups,
        "caliber_thresholds": caliber_thresholds,
        "endpoint_count": int(endpoint_mask.sum()),
    }


def prediction_branch_metrics(
    pred_fg: np.ndarray,
    gt_skeleton: np.ndarray,
    branch_definition: dict[str, object],
    spacing: tuple[float, float, float],
    distance_threshold_mm: float,
    coverage_thresholds: list[float],
) -> dict[str, float]:
    branch_labels = branch_definition["branch_labels"]
    branch_ids = branch_definition["branch_ids"]
    branch_sizes = branch_definition["branch_sizes"]
    groups = branch_definition["groups"]
    assert isinstance(branch_labels, np.ndarray)
    assert isinstance(branch_ids, np.ndarray)
    assert isinstance(branch_sizes, np.ndarray)
    assert isinstance(groups, dict)

    pred_skeleton = skeletonize_binary(pred_fg)
    pred_tree = point_tree(pred_skeleton, spacing)
    gt_distances = distances_to_tree(gt_skeleton, pred_tree, spacing)
    distance_map = np.full(gt_skeleton.shape, np.inf, dtype=np.float32)
    if gt_distances.size:
        coords = np.argwhere(gt_skeleton)
        distance_map[coords[:, 0], coords[:, 1], coords[:, 2]] = gt_distances

    covered_mask = distance_map <= distance_threshold_mm
    if branch_ids.size:
        covered_counts = np.asarray(
            ndimage.sum(covered_mask, labels=branch_labels, index=branch_ids),
            dtype=np.float64,
        )
        branch_coverage = covered_counts / branch_sizes
    else:
        branch_coverage = np.array([], dtype=float)

    metrics: dict[str, float] = {
        "gt_centerline_to_pred_mean_mm": (
            float(np.mean(gt_distances)) if gt_distances.size else float("nan")
        ),
        "gt_centerline_recall_2mm": (
            float(np.mean(gt_distances <= distance_threshold_mm))
            if gt_distances.size
            else float("nan")
        ),
    }

    for group_name, selector_value in groups.items():
        selector = np.asarray(selector_value, dtype=bool)
        group_sizes = branch_sizes[selector]
        group_coverage = branch_coverage[selector]
        metrics[f"{group_name}_branch_count"] = float(selector.sum())
        metrics[f"{group_name}_branch_mean_coverage_2mm"] = (
            float(np.mean(group_coverage)) if group_coverage.size else float("nan")
        )
        for threshold in coverage_thresholds:
            suffix = coverage_suffix(threshold)
            detected = group_coverage >= threshold
            metrics[f"{group_name}_branch_recall_2mm_{suffix}"] = safe_div(
                float(detected.sum()),
                float(group_coverage.size),
            )
            metrics[f"{group_name}_branch_weighted_recall_2mm_{suffix}"] = safe_div(
                float(group_sizes[detected].sum()) if group_sizes.size else 0.0,
                float(group_sizes.sum()) if group_sizes.size else 0.0,
            )
    return metrics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna()
    if valid.empty:
        return result
    ordered = valid.sort_values()
    count = len(ordered)
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for reverse_index in range(count - 1, -1, -1):
        rank = reverse_index + 1
        candidate = float(ordered.iloc[reverse_index]) * count / rank
        running = min(running, candidate)
        adjusted[reverse_index] = min(running, 1.0)
    result.loc[ordered.index] = adjusted
    return result


def analyze_paired(
    rows: list[dict[str, object]],
    baseline: str,
    proposed: str,
    repetitions: int,
) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    excluded = {
        "split",
        "variant",
        "case_id",
        "uid",
        "prediction_path",
    }
    metrics = [
        column
        for column in frame.columns
        if column not in excluded
        and not column.endswith("_count")
        and not column.startswith("caliber_threshold_")
    ]
    baseline_frame = frame[frame["variant"] == baseline].set_index("case_id")
    proposed_frame = frame[frame["variant"] == proposed].set_index("case_id")
    common = baseline_frame.index.intersection(proposed_frame.index)

    stats_rows: list[dict[str, object]] = []
    for metric_index, metric in enumerate(metrics):
        baseline_values = pd.to_numeric(
            baseline_frame.loc[common, metric],
            errors="coerce",
        ).to_numpy(dtype=float)
        proposed_values = pd.to_numeric(
            proposed_frame.loc[common, metric],
            errors="coerce",
        ).to_numpy(dtype=float)
        valid = np.isfinite(baseline_values) & np.isfinite(proposed_values)
        baseline_values = baseline_values[valid]
        proposed_values = proposed_values[valid]
        if baseline_values.size == 0:
            continue

        difference = proposed_values - baseline_values
        direction = metric_direction(metric)
        if direction == "higher":
            oriented_difference = difference
        elif direction == "lower":
            oriented_difference = -difference
        else:
            oriented_difference = None
        try:
            p_value = float(
                stats.wilcoxon(
                    difference,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                ).pvalue
            )
        except ValueError:
            p_value = float("nan")
        ci_low, ci_high = bootstrap_mean_ci(
            difference,
            repetitions=repetitions,
            seed=260510 + metric_index,
        )
        baseline_mean = float(np.mean(baseline_values))
        proposed_mean = float(np.mean(proposed_values))
        mean_delta = float(np.mean(difference))
        residual_miss_reduction = (
            100.0 * mean_delta / (1.0 - baseline_mean)
            if "recall" in metric and baseline_mean < 1.0
            else float("nan")
        )
        stats_rows.append(
            {
                "metric": metric,
                "n": int(difference.size),
                "baseline_mean": baseline_mean,
                "proposed_mean": proposed_mean,
                "mean_delta": mean_delta,
                "direction": direction,
                "bootstrap95_low": ci_low,
                "bootstrap95_high": ci_high,
                "wilcoxon_p": p_value,
                "residual_miss_reduction_pct": residual_miss_reduction,
                "improved_cases": (
                    int((oriented_difference > 0).sum())
                    if oriented_difference is not None
                    else ""
                ),
                "worse_cases": (
                    int((oriented_difference < 0).sum())
                    if oriented_difference is not None
                    else ""
                ),
                "ties": (
                    int((oriented_difference == 0).sum())
                    if oriented_difference is not None
                    else ""
                ),
            }
        )

    stats_frame = pd.DataFrame(stats_rows)
    if not stats_frame.empty:
        stats_frame["fdr_bh"] = benjamini_hochberg(stats_frame["wilcoxon_p"])
    return stats_frame.to_dict(orient="records")


def summarize_variants(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    excluded = {"split", "variant", "case_id", "uid", "prediction_path"}
    metrics = [column for column in frame.columns if column not in excluded]
    summary: list[dict[str, object]] = []
    for variant in frame["variant"].drop_duplicates():
        variant_frame = frame[frame["variant"] == variant]
        for metric in metrics:
            values = pd.to_numeric(variant_frame[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            summary.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
                    "median": float(np.median(values)),
                }
            )
    return summary


def format_number(value: object, digits: int = 4) -> str:
    number = float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def write_findings(
    path: Path,
    split: str,
    paired_rows: list[dict[str, object]],
    baseline: str,
    proposed: str,
) -> None:
    selected_prefixes = (
        "terminal_small_q33_branch_recall_2mm_",
        "terminal_small_q20_branch_recall_2mm_",
        "terminal_branch_recall_2mm_",
        "small_q33_branch_recall_2mm_",
    )
    selected = [
        row
        for row in paired_rows
        if str(row["metric"]).startswith(selected_prefixes)
        or row["metric"]
        in {
            "terminal_small_q33_branch_mean_coverage_2mm",
            "terminal_small_q20_branch_mean_coverage_2mm",
            "gt_centerline_to_pred_mean_mm",
            "gt_centerline_recall_2mm",
        }
    ]
    lines = [
        f"# {proposed} branch sensitivity: {split}",
        "",
        "This exploratory analysis reuses existing final predictions; no model was retrained.",
        (
            "A branch is a GT skeleton segment between junctions and endpoints. "
            "Terminal branches contain a GT endpoint. Small-caliber branches are "
            "defined from the within-case distribution of branch median centerline "
            "radius. The primary terminal-small definition is a terminal branch in "
            "the bottom radius tertile (terminal_small_q33)."
        ),
        (
            "Branch detection is reported when at least 25%, 50%, or 75% of a branch "
            "lies within 2 mm of the predicted centerline. All threshold variants are "
            "retained and FDR corrected."
        ),
        "",
        f"| Metric | {baseline} | {proposed} | Delta | Residual miss reduction | 95% CI | FDR | Wins/Losses |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["metric"]),
                    format_number(row["baseline_mean"]),
                    format_number(row["proposed_mean"]),
                    format_number(row["mean_delta"]),
                    (
                        f"{float(row['residual_miss_reduction_pct']):.1f}%"
                        if math.isfinite(float(row["residual_miss_reduction_pct"]))
                        else "NA"
                    ),
                    (
                        f"{format_number(row['bootstrap95_low'])} to "
                        f"{format_number(row['bootstrap95_high'])}"
                    ),
                    format_number(row.get("fdr_bh", float("nan")), 3),
                    f"{row['improved_cases']}/{row['worse_cases']}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    predictions = parse_prediction_specs(args.prediction)
    coverage_thresholds = parse_float_list(args.coverage_thresholds)
    caliber_quantiles = parse_float_list(args.caliber_quantiles)
    topology_cache = Path(args.topology_cache_dir) if args.topology_cache_dir else None
    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.case_list, args.split, args.case_limit)
    rows: list[dict[str, object]] = []
    for case_index, case_row in enumerate(cases.itertuples(index=False), start=1):
        case_id = str(case_row.case_id)
        label, spacing, label_geometry = read_image(
            args.label_dir / f"{case_id}.nii.gz"
        )
        label_fg_full = label > 0
        gt_skeleton_full = load_cached_gt_skeleton(case_id, topology_cache, label_fg_full)

        loaded_predictions: list[tuple[str, np.ndarray]] = []
        union = label_fg_full.copy()
        for variant, prediction_dir in predictions:
            prediction_path = prediction_dir / f"{case_id}.nii.gz"
            if not prediction_path.exists():
                raise FileNotFoundError(prediction_path)
            pred, _, prediction_geometry = read_image(prediction_path)
            require_matching_geometry(
                label_geometry,
                prediction_geometry,
                context=f"prediction {variant}/{case_id}",
            )
            pred_fg = pred > 0
            union |= pred_fg
            loaded_predictions.append((variant, pred_fg))

        slices = crop_slices(union)
        label_fg = label_fg_full[slices]
        gt_skeleton = gt_skeleton_full[slices]
        branch_definition = build_branch_definition(
            label_fg=label_fg,
            gt_skeleton=gt_skeleton,
            spacing=spacing,
            caliber_quantiles=caliber_quantiles,
            branch_min_voxels=args.branch_min_voxels,
        )
        groups = branch_definition["groups"]
        caliber_thresholds = branch_definition["caliber_thresholds"]
        assert isinstance(groups, dict)
        assert isinstance(caliber_thresholds, dict)

        static_values: dict[str, float] = {
            "gt_endpoint_count": float(branch_definition["endpoint_count"]),
        }
        for group_name, selector in groups.items():
            static_values[f"{group_name}_branch_count"] = float(np.asarray(selector).sum())
        for suffix, threshold in caliber_thresholds.items():
            static_values[f"caliber_threshold_{suffix}_mm"] = float(threshold)

        for variant, pred_fg_full in loaded_predictions:
            metrics = prediction_branch_metrics(
                pred_fg=pred_fg_full[slices],
                gt_skeleton=gt_skeleton,
                branch_definition=branch_definition,
                spacing=spacing,
                distance_threshold_mm=args.distance_threshold_mm,
                coverage_thresholds=coverage_thresholds,
            )
            rows.append(
                {
                    "split": args.split,
                    "variant": variant,
                    "case_id": case_id,
                    **static_values,
                    **metrics,
                }
            )
            print(
                f"{case_index:02d}/{len(cases):02d} {variant} {case_id} "
                f"terminal_small_q33_cov50="
                f"{metrics.get('terminal_small_q33_branch_recall_2mm_cov50', float('nan')):.4f}",
                flush=True,
            )

    case_path = out_dir / "branch_sensitivity_case_metrics.csv"
    summary_path = out_dir / "branch_sensitivity_summary.csv"
    paired_path = out_dir / "branch_sensitivity_paired_stats.csv"
    findings_path = out_dir / "branch_sensitivity_findings.md"
    write_csv(case_path, rows)
    write_csv(summary_path, summarize_variants(rows))

    baseline = predictions[0][0]
    proposed = predictions[1][0]
    paired_rows = analyze_paired(
        rows,
        baseline=baseline,
        proposed=proposed,
        repetitions=args.bootstrap_repetitions,
    )
    write_csv(paired_path, paired_rows)
    write_findings(findings_path, args.split, paired_rows, baseline, proposed)

    metadata = {
        "split": args.split,
        "case_count": len(cases),
        "models": [name for name, _ in predictions],
        "topology_cache_used": topology_cache is not None,
        "distance_threshold_mm": args.distance_threshold_mm,
        "coverage_thresholds": coverage_thresholds,
        "caliber_quantiles": caliber_quantiles,
        "branch_min_voxels": args.branch_min_voxels,
        "branch_definition": (
            "GT skeleton segments after removing degree>=3 junction voxels; "
            "terminal segments contain degree==1 endpoints; small-caliber groups "
            "use within-case branch median centerline radius quantiles."
        ),
        "geometry_checks": ["size", "spacing", "origin", "direction"],
        "paths_are_intentionally_omitted_from_result_tables": True,
    }
    (out_dir / "branch_sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(case_path)
    print(paired_path)


if __name__ == "__main__":
    main()
