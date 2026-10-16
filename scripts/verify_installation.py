"""Check Neurite, VoxelMorph and GPSseq affine network after installation."""
from __future__ import annotations
import sys
import importlib.metadata as md

print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)
for name in ("tensorflow", "keras", "neurite", "voxelmorph"):
    try:
        print(f"{name}: {md.version(name)}")
    except md.PackageNotFoundError:
        raise SystemExit(f"Missing dependency: {name}")

import neurite  # noqa: E402
import voxelmorph as vxm  # noqa: E402

print("Neurite imported from:", neurite.__file__)
print("VoxelMorph imported from:", vxm.__file__)
cls = getattr(vxm.networks, "VxmAffineFeatureDetector", None)
if cls is None:
    raise SystemExit(
        "ERROR: VxmAffineFeatureDetector missing. "
        "Check that the GPSseq legacy source archive, NOT PyPI voxelmorph==0.2, was installed."
    )
print("PASS: VxmAffineFeatureDetector exists")
print("NOTE: This test does not validate loading model weights or real inference.")
