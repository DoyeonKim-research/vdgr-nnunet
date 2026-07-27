from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from vdgr_nnunet.distance_strata import binary_foreground, distance_strata_from_mask
from vdgr_nnunet.image_geometry import image_geometry, require_matching_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate case-adaptive VDGR distance-stratum targets from vessel masks."
    )
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.nii.gz")
    parser.add_argument(
        "--foreground-values",
        default=None,
        help="Optional comma-separated foreground labels. Default: all values > 0.",
    )
    parser.add_argument("--lower-percentile", type=float, default=33.3333)
    parser.add_argument("--upper-percentile", type=float, default=66.6667)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def case_id(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def parse_foreground_values(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("--foreground-values did not contain a label")
    return parsed


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_stats(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            str(row["case_id"]): dict(row)
            for row in csv.DictReader(handle)
            if row.get("case_id")
        }


def derive_strata(
    source: Path,
    foreground_values: tuple[int, ...] | None,
    lower_percentile: float,
    upper_percentile: float,
):
    image = sitk.ReadImage(str(source))
    labels = sitk.GetArrayFromImage(image)
    vessel = binary_foreground(labels, foreground_values)
    strata, stats = distance_strata_from_mask(
        vessel,
        image.GetSpacing(),
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    return image, strata, stats


def main() -> None:
    args = parse_args()
    values = parse_foreground_values(args.foreground_values)
    paths = sorted(args.label_dir.glob(args.pattern))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No labels matched {args.pattern!r} in {args.label_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "distance_stratum_case_stats.csv"
    existing_stats = read_existing_stats(stats_path)
    rows: list[dict[str, object]] = []
    generated_cases = 0
    reused_cases = 0
    for index, source in enumerate(paths, start=1):
        identifier = case_id(source)
        output = args.output_dir / f"{identifier}.nii.gz"
        if output.exists() and not args.overwrite:
            if identifier in existing_stats:
                rows.append(existing_stats[identifier])
            else:
                image, expected_strata, stats = derive_strata(
                    source,
                    values,
                    args.lower_percentile,
                    args.upper_percentile,
                )
                existing_image = sitk.ReadImage(str(output))
                require_matching_geometry(
                    image_geometry(image),
                    image_geometry(existing_image),
                    context=f"existing distance strata/{identifier}",
                )
                existing_strata = sitk.GetArrayFromImage(existing_image)
                if not np.array_equal(existing_strata, expected_strata):
                    raise ValueError(
                        f"Existing target differs from the configured derivation: {output}. "
                        "Use --overwrite to regenerate it."
                    )
                rows.append(
                    {
                        "case_id": identifier,
                        "source_file": source.name,
                        **stats.as_dict(),
                    }
                )
            reused_cases += 1
            print(f"{index:03d}/{len(paths):03d} skip {source.name}", flush=True)
            continue

        image, strata, stats = derive_strata(
            source,
            values,
            args.lower_percentile,
            args.upper_percentile,
        )
        output_image = sitk.GetImageFromArray(strata)
        output_image.CopyInformation(image)
        sitk.WriteImage(output_image, str(output), True)
        rows.append(
            {"case_id": identifier, "source_file": source.name, **stats.as_dict()}
        )
        generated_cases += 1
        print(
            f"{index:03d}/{len(paths):03d} {source.name} "
            f"boundary={stats.boundary_proximal_voxels / max(stats.vessel_voxels, 1):.3f}",
            flush=True,
        )

    if rows:
        write_csv(stats_path, rows)
    metadata = {
        "class_definition": {
            "0": "background",
            "1": "deep-interior vessel voxels",
            "2": "intermediate vessel voxels",
            "3": "boundary-proximal vessel voxels",
        },
        "lower_percentile": args.lower_percentile,
        "upper_percentile": args.upper_percentile,
        "foreground_values": list(values) if values is not None else "all labels > 0",
        "selected_cases": len(rows),
        "generated_cases": generated_cases,
        "reused_cases": reused_cases,
        "distance_units": "millimetres",
    }
    (args.output_dir / "distance_stratum_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
