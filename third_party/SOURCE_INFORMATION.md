# GPSseq third-party source archives

The archives in this folder are **pinned historical TensorFlow source snapshots** used for GPSseq.
The internal package version `0.2` does not uniquely identify either snapshot and should
not be replaced with an ordinary `pip install neurite==0.2` or `pip install voxelmorph==0.2`.

| Archive | SHA256 | Essential GPSseq functionality |
| --- | --- | --- |
| `neurite-gpsseq-legacy.zip` | `4aada0bdcb05e92511d98cfa6b84a2ec018d893444686444debd95ea3788b44e` | `neurite.tf.layers.Constant`, Keras 3 compatible MAE/MSE imports |
| `voxelmorph-gpsseq-legacy.zip` | `06c4dff583805fbb2e74da6263470d4637ac97ee571376da5933f71ac0ec335c` | `voxelmorph.tf.networks.VxmAffineFeatureDetector` |

## Provenance and licensing

- Neurite source comes from the developer-provided installed `neurite.zip`.
  Its 22 Python source files were preserved byte-for-byte; the packaging files
  were created for GPSseq and `.pyc`/cache files were removed. This snapshot does not
  identify an exact upstream Git commit. See the `GPSSEQ_PACKAGING_NOTES.md` inside the archive.
- The uploaded Neurite ZIP lacked a license file. The archive includes a copy of the
  standard Apache-2.0 license text, consistent with the upstream repository's
  Apache-2.0 declaration. The historical upstream commit/license state should be confirmed
  before formal public redistribution.
- VoxelMorph is a repackaged historical development snapshot with bundled Apache-2.0
  license. See its internal `GPSSEQ_PACKAGING_NOTES.md`.
- Upstream projects: https://github.com/adalca/neurite and https://github.com/voxelmorph/voxelmorph

## Installation

Install other dependencies from `requirements.txt` first. Then, from the GPSseq root:

```bash
python -m pip install --no-deps --force-reinstall ./third_party/neurite-gpsseq-legacy.zip
python -m pip install --no-deps --force-reinstall ./third_party/voxelmorph-gpsseq-legacy.zip
python scripts/verify_installation.py
```

No `patch_neurite.py` step is needed when the GPSseq Neurite snapshot is installed.
Tests with the actual model weights are still required.
