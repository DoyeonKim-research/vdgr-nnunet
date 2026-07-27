from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path


FILENAME = "nnUNetTrainerVDGR.py"


def trainer_directory() -> Path:
    spec = importlib.util.find_spec("nnunetv2")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("nnunetv2 is not installed in this Python environment")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    directory = package_root / "training" / "nnUNetTrainer"
    if not directory.is_dir():
        raise FileNotFoundError(f"Could not locate nnU-Net trainer directory: {directory}")
    return directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install the VDGR nnU-Net trainer shim.")
    parser.add_argument("--check", action="store_true", help="Report status without changing files.")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(__file__).resolve().parents[1] / "nnunet_extension" / FILENAME
    target = trainer_directory() / FILENAME

    if args.check:
        status = "installed" if target.is_file() else "not installed"
        print(f"{status}: {target}")
        return
    if args.uninstall:
        if target.is_file():
            target.unlink()
            print(f"Removed {target}")
        else:
            print(f"Already absent: {target}")
        return
    if target.exists() and not args.force:
        raise FileExistsError(f"{target} already exists; pass --force to replace it")
    shutil.copy2(source, target)
    print(f"Installed {source.name} -> {target}")


if __name__ == "__main__":
    main()
