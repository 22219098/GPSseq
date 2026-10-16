"""
Patch Neurite 0.2 for the GPSseq TensorFlow/Keras environment.

Usage:
    python scripts/patch_neurite.py
    python scripts/patch_neurite.py --apply

Only these installed Neurite files are modified:

    neurite/tf/metrics.py
    neurite/tf/losses.py

A backup is created before modification.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata as md
from pathlib import Path
import shutil
import sys
import sysconfig


REPLACEMENTS = {
    "from tensorflow.keras.losses import mean_absolute_error as l1":
        "from tensorflow.keras.losses import MAE as l1",

    "from tensorflow.keras.losses import mean_squared_error as l2":
        "from tensorflow.keras.losses import MSE as l2",
}

TARGET_FILES = (
    "metrics.py",
    "losses.py",
)


def get_neurite_tf_dir() -> Path:
    site_packages = Path(sysconfig.get_path("purelib"))
    return site_packages / "neurite" / "tf"


def patch_file(path: Path, apply: bool) -> bool:

    original = path.read_text(encoding="utf-8")
    updated = original

    for old, new in REPLACEMENTS.items():
        updated = updated.replace(old, new)

    # Check syntax before modifying anything.
    ast.parse(
        updated,
        filename=str(path)
    )

    if updated == original:
        print(
            f"OK: {path.name} is already patched "
            f"or requires no change"
        )
        return False

    old_count = sum(
        original.count(old)
        for old in REPLACEMENTS
    )

    print(
        f"PATCH NEEDED: {path} "
        f"({old_count} legacy import(s))"
    )

    if not apply:
        return True

    backup = path.with_name(
        path.name + ".gpsseq-original.bak"
    )

    if not backup.exists():
        shutil.copy2(path, backup)

    path.write_text(
        updated,
        encoding="utf-8"
    )

    print(f"PATCHED: {path}")
    print(f"BACKUP:  {backup}")

    return True


def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply the patch"
    )

    args = parser.parse_args()

    print("Python:", sys.executable)

    try:
        neurite_version = md.version("neurite")
    except md.PackageNotFoundError:

        print(
            "ERROR: neurite is not installed."
        )

        return 2

    print(
        "Neurite version:",
        neurite_version
    )

    if neurite_version != "0.2":

        print(
            "ERROR: GPSseq expects "
            "neurite==0.2."
        )

        return 2

    root = get_neurite_tf_dir()

    print(
        "Target:",
        root
    )

    if not root.is_dir():

        print(
            f"ERROR: Neurite directory "
            f"not found: {root}"
        )

        return 2

    changed = False
    recognized = 0

    for filename in TARGET_FILES:

        path = root / filename

        if not path.is_file():

            print(
                f"ERROR: missing file: {path}"
            )

            return 2

        text = path.read_text(
            encoding="utf-8"
        )

        if any(
            old in text or new in text
            for old, new in REPLACEMENTS.items()
        ):
            recognized += 1

        changed = (
            patch_file(
                path,
                args.apply
            )
            or changed
        )

    if recognized == 0:

        print(
            "ERROR: expected Neurite "
            "imports were not found."
        )

        return 2

    if not changed:

        print(
            "No patch was necessary."
        )

    elif not args.apply:

        print(
            "\nDRY RUN: no files changed."
        )

        print(
            "Run with --apply to apply."
        )

    if args.apply:

        print(
            "\nNeurite patch completed."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())