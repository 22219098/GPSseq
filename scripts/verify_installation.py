"""Validate the GPSseq legacy Neurite and VoxelMorph environment.

Run this in a fresh Python process, e.g. `python scripts/verify_installation.py`.
This checks network construction, not pretrained weight loading or inference.
"""
import importlib.metadata as md
import inspect
import sys

EXPECTED = {
    "tensorflow": "2.16.1",
    "keras": "3.9.0",
    "neurite": "0.2",
    "voxelmorph": "0.2",
}


def fail(message):
    raise SystemExit("FAIL: " + message)


print("=== GPSseq installation verification ===")
print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
if sys.version_info[:2] != (3, 9):
    fail("GPSseq requires Python 3.9")

for pkg, expected in EXPECTED.items():
    try:
        actual = md.version(pkg)
    except md.PackageNotFoundError:
        fail(f"{pkg} is not installed")
    print(f"{pkg}: {actual}")
    if actual != expected:
        fail(f"{pkg} must be {expected}; found {actual}")

try:
    import neurite as ne
except Exception as exc:
    fail(f"Neurite import failed: {type(exc).__name__}: {exc}")
print("Neurite:", ne.__file__)
constant_cls = getattr(ne.layers, "Constant", None)
if constant_cls is None:
    fail("neurite.layers.Constant is missing. Install third_party/neurite-gpsseq-legacy.zip")
print("PASS: neurite.layers.Constant exists")
print("Signature:", inspect.signature(constant_cls))

try:
    import voxelmorph as vxm
except Exception as exc:
    fail(f"VoxelMorph import failed: {type(exc).__name__}: {exc}")
print("VoxelMorph:", vxm.__file__)
cls = getattr(vxm.networks, "VxmAffineFeatureDetector", None)
if cls is None:
    fail("VxmAffineFeatureDetector is missing. Install third_party/voxelmorph-gpsseq-legacy.zip")
print("PASS: VxmAffineFeatureDetector exists")
print("Signature:", inspect.signature(cls))

try:
    affine = cls((256, 256))
except Exception as exc:
    fail(f"Affine network initialization failed: {type(exc).__name__}: {exc}")
print("PASS: VxmAffineFeatureDetector initialized successfully")
print("=== GPSseq compatibility setup PASSED ===")
print("Pretrained model weights and inference are not checked by this script.")
