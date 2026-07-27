from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from vdgr_nnunet.image_geometry import read_image, require_matching_geometry
from vdgr_nnunet.metrics import all_metrics, metric_direction


IDENTIFIERS = {"model", "case_id", "split"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate binary vessel predictions with manuscript metrics."
    )
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--strata-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Prediction source as model=directory. Can be repeated.",
    )
    parser.add_argument("--reference-model", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def prediction_specs(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Prediction source must be model=directory: {value}")
        model, directory_text = value.split("=", 1)
        directory = Path(directory_text)
        if not model.strip() or not directory.is_dir():
            raise FileNotFoundError(f"Invalid prediction source: {value}")
        result.append((model.strip(), directory))
    if len({model for model, _ in result}) != len(result):
        raise ValueError("Prediction model names must be unique")
    return result


def case_ids(path: Path, split: str, limit: int) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "case_id" not in rows[0] or "split" not in rows[0]:
        raise ValueError("Case list must contain case_id and split columns")
    selected = sorted(
        str(row["case_id"])
        for row in rows
        if str(row["split"]).strip().lower() == split.strip().lower()
    )
    if limit > 0:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"No cases found for split {split!r}")
    return selected


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric_metric_names(rows: list[dict[str, object]]) -> list[str]:
    return [key for key in rows[0] if key not in IDENTIFIERS]


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = numeric_metric_names(rows)
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    summaries = []
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        summary: dict[str, object] = {"model": model, "n_cases": len(selected)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=float)
            summary[f"{metric}_mean"] = float(np.nanmean(values))
            summary[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        summaries.append(summary)
    return summaries


def bootstrap_interval(
    differences: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    finite = differences[np.isfinite(differences)]
    if finite.size == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, finite.size, size=(repetitions, finite.size))
    means = finite[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    result = np.full(len(p_values), np.nan, dtype=float)
    finite_indices = [index for index, value in enumerate(p_values) if np.isfinite(value)]
    if not finite_indices:
        return result.tolist()
    ordered = sorted(finite_indices, key=lambda index: p_values[index])
    number = len(ordered)
    adjusted = np.empty(number, dtype=float)
    running = 1.0
    for reverse_rank in range(number - 1, -1, -1):
        index = ordered[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, p_values[index] * number / rank)
        adjusted[reverse_rank] = min(running, 1.0)
    for position, index in enumerate(ordered):
        result[index] = adjusted[position]
    return result.tolist()


def paired_analysis(
    rows: list[dict[str, object]],
    reference_model: str,
    repetitions: int,
    seed: int,
) -> list[dict[str, object]]:
    metrics = numeric_metric_names(rows)
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    proposed_models = [model for model in models if model != reference_model]
    by_key = {(str(row["model"]), str(row["case_id"])): row for row in rows}
    cases = sorted(str(row["case_id"]) for row in rows if row["model"] == reference_model)
    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    for model in proposed_models:
        for metric in metrics:
            reference = np.asarray(
                [float(by_key[(reference_model, case)][metric]) for case in cases],
                dtype=float,
            )
            proposed = np.asarray(
                [float(by_key[(model, case)][metric]) for case in cases],
                dtype=float,
            )
            valid = np.isfinite(reference) & np.isfinite(proposed)
            differences = proposed[valid] - reference[valid]
            direction = metric_direction(metric)
            if direction == "higher":
                oriented_differences = differences
            elif direction == "lower":
                oriented_differences = -differences
            else:
                oriented_differences = None
            low, high = bootstrap_interval(differences, repetitions, rng)
            if differences.size == 0 or np.allclose(differences, 0):
                p_value = 1.0 if differences.size else float("nan")
            else:
                p_value = float(stats.wilcoxon(differences).pvalue)
            output.append(
                {
                    "reference_model": reference_model,
                    "model": model,
                    "metric": metric,
                    "n_cases": int(differences.size),
                    "reference_mean": float(np.mean(reference[valid])) if valid.any() else float("nan"),
                    "model_mean": float(np.mean(proposed[valid])) if valid.any() else float("nan"),
                    "mean_delta": float(np.mean(differences)) if differences.size else float("nan"),
                    "direction": direction,
                    "bootstrap95_low": low,
                    "bootstrap95_high": high,
                    "wilcoxon_p": p_value,
                    "improved_cases": (
                        int((oriented_differences > 0).sum())
                        if oriented_differences is not None
                        else ""
                    ),
                    "worse_cases": (
                        int((oriented_differences < 0).sum())
                        if oriented_differences is not None
                        else ""
                    ),
                    "tied_cases": (
                        int((oriented_differences == 0).sum())
                        if oriented_differences is not None
                        else ""
                    ),
                }
            )
    adjusted = benjamini_hochberg([float(row["wilcoxon_p"]) for row in output])
    for row, value in zip(output, adjusted):
        row["fdr_bh"] = value
    return output


def main() -> None:
    args = parse_args()
    predictions = prediction_specs(args.prediction)
    selected_cases = case_ids(args.case_list, args.split, args.case_limit)
    rows: list[dict[str, object]] = []
    for index, case_id in enumerate(selected_cases, start=1):
        label, spacing, label_geometry = read_image(
            args.label_dir / f"{case_id}.nii.gz"
        )
        strata, _, strata_geometry = read_image(
            args.strata_dir / f"{case_id}.nii.gz"
        )
        require_matching_geometry(
            label_geometry,
            strata_geometry,
            context=f"distance strata/{case_id}",
        )
        for model, directory in predictions:
            prediction, _, prediction_geometry = read_image(
                directory / f"{case_id}.nii.gz"
            )
            require_matching_geometry(
                label_geometry,
                prediction_geometry,
                context=f"prediction {model}/{case_id}",
            )
            metrics = all_metrics(prediction > 0, label > 0, strata, spacing)
            rows.append(
                {
                    "model": model,
                    "case_id": case_id,
                    "split": args.split,
                    **metrics,
                }
            )
            print(
                f"{index:03d}/{len(selected_cases):03d} {model} {case_id} "
                f"Dice={metrics['dice']:.4f} clDice={metrics['cldice']:.4f}",
                flush=True,
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "case_metrics.csv", rows)
    write_csv(args.out_dir / "model_summary.csv", summarize(rows))
    reference_model = args.reference_model or predictions[0][0]
    paired = paired_analysis(
        rows,
        reference_model=reference_model,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    write_csv(args.out_dir / "paired_stats.csv", paired)
    metadata = {
        "split": args.split,
        "case_count": len(selected_cases),
        "models": [model for model, _ in predictions],
        "reference_model": reference_model,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "distance_units": "millimetres",
        "geometry_checks": ["size", "spacing", "origin", "direction"],
        "paths_are_intentionally_omitted_from_result_tables": True,
    }
    (args.out_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
