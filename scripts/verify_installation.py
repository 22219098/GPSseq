"""
Verify the GPSseq TensorFlow / Neurite / VoxelMorph
compatibility environment.
"""

from __future__ import annotations

import importlib.metadata as md
import inspect
import sys


EXPECTED = {
    "tensorflow": "2.16.1",
    "keras": "3.9.0",
    "neurite": "0.2",
}


print(
    "=== GPSseq environment verification ==="
)

print(
    "Python:",
    sys.version
)

print(
    "Executable:",
    sys.executable
)

print()


for package, expected in EXPECTED.items():

    try:
        actual = md.version(package)

    except md.PackageNotFoundError:

        raise SystemExit(
            f"FAIL: {package} is not installed"
        )

    status = (
        "OK"
        if actual == expected
        else f"WARNING: expected {expected}"
    )

    print(
        f"{package}: {actual} [{status}]"
    )


try:
    voxelmorph_version = md.version(
        "voxelmorph"
    )

except md.PackageNotFoundError:

    raise SystemExit(
        "FAIL: voxelmorph is not installed"
    )


print(
    f"voxelmorph: {voxelmorph_version}"
)

print()


# Import in this fresh Python process.
try:

    import neurite

except Exception as exc:

    raise SystemExit(
        "FAIL: Neurite import failed.\n"
        f"{type(exc).__name__}: {exc}"
    )


try:

    import voxelmorph as vxm

except Exception as exc:

    raise SystemExit(
        "FAIL: VoxelMorph import failed.\n"
        f"{type(exc).__name__}: {exc}"
    )


print(
    "Neurite:",
    neurite.__file__
)

print(
    "VoxelMorph:",
    vxm.__file__
)


# Check GPSseq-required network.
cls = getattr(
    vxm.networks,
    "VxmAffineFeatureDetector",
    None
)


if cls is None:

    raise SystemExit(
        "FAIL: VxmAffineFeatureDetector "
        "is missing.\n"
        "Make sure the GPSseq legacy "
        "VoxelMorph archive was installed."
    )


print(
    "PASS: VxmAffineFeatureDetector exists"
)

print(
    "Signature:",
    inspect.signature(cls)
)


# Actually instantiate the model.
try:

    model = cls(
        (256, 256)
    )

except Exception as exc:

    raise SystemExit(
        "FAIL: VxmAffineFeatureDetector "
        "could not be instantiated.\n"
        f"{type(exc).__name__}: {exc}"
    )


print(
    "PASS: VxmAffineFeatureDetector "
    "initialized successfully"
)

print()

print(
    "=== GPSseq compatibility setup PASSED ==="
)

print(
    "Model weights will be tested separately."
)