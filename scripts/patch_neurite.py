"""GPSseq compatibility patch for Neurite 0.2 and TensorFlow 2.16 / Keras 3.

Usage:
    python scripts/patch_neurite.py          # preview, no changes
    python scripts/patch_neurite.py --apply  # patch current environment, backing up files

The script edits *only* the selected Python environment's neurite/tf/metrics.py
and neurite/tf/losses.py; it does not affect the GPSseq source or weights.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shutil
import sys
import sysconfig

OLD_TO_NEW = {
    "from tensorflow.keras.losses import mean_absolute_error as l1":
        "from tensorflow.keras.losses import MAE as l1",
    "from tensorflow.keras.losses import mean_squared_error as l2":
        "from tensorflow.keras.losses import MSE as l2",
}
TARGETS = ("metrics.py", "losses.py")


def run(site_packages: Path, apply: bool) -> int:
    root = site_packages / "neurite" / "tf"
    if not root.is_dir():
        print(f"ERROR: neurite/tf not found in current environment: {root}")
        print("Install neurite==0.2 first and use the intended conda environment.")
        return 2

    pending = []
    discovered = 0
    for name in TARGETS:
        path = root / name
        if not path.is_file():
            print(f"ERROR: expected Neurite source missing: {path}")
            return 2
        original = path.read_text(encoding="utf-8")
        updated = original
        for old, new in OLD_TO_NEW.items():
            if old in original or new in original:
                discovered += 1
            updated = updated.replace(old, new)
        ast.parse(updated, filename=str(path))  # refuse to write invalid syntax
        if updated != original:
            pending.append((path, original, updated))
            print(f"PATCH NEEDED: {path} ({sum(original.count(old) for old in OLD_TO_NEW)} replacements)")
        else:
            print(f"ALREADY COMPATIBLE / UNCHANGED: {path}")

    if not discovered:
        print("ERROR: No recognized legacy import or patched alias was found in expected files.")
        print("Your Neurite source may differ; inspect it manually instead of guessing.")
        return 2

    # Do not claim success when unsupported old usages remain inside the two files.
    for path, _, updated in pending:
        for line_no, line in enumerate(updated.splitlines(), 1):
            if line.lstrip().startswith("from tensorflow.keras.losses import ") and (
                "mean_absolute_error" in line or "mean_squared_error" in line
            ):
                print(f"ERROR: unhandled import at {path}:{line_no}: {line.strip()}")
                return 2

    if not apply:
        print("\nDRY RUN: no changes made. To apply: python scripts/patch_neurite.py --apply")
        return 0

    for path, original, updated in pending:
        backup = path.with_name(path.name + ".gpsseq-original.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(updated, encoding="utf-8")
        print(f"PATCHED: {path} (backup: {backup.name})")

    print("\nFinished. Test in a fresh Python process:")
    print("python -c \"import neurite, voxelmorph; print(\'Imports OK\')\"")
    print("Note: reinstalling neurite with pip may overwrite this patch.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Make the changes (default: dry run)")
    parser.add_argument("--site-packages", type=Path, default=None,
                        help="For diagnostics/tests only; defaults to the active Python environment")
    args = parser.parse_args()
    site = args.site_packages or Path(sysconfig.get_path("purelib"))
    print(f"Active Python: {sys.executable}")
    print(f"Target site-packages: {site}")
    if args.site_packages is not None:
        print("WARNING: explicit site-packages override in use")
    return run(site, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
