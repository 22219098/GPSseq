# GPSseq

### GPS-seq maps brain-wide transcriptional networks to guide schizophrenia therapeutic strategy

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![TensorFlow 2.16.1](https://img.shields.io/badge/TensorFlow-2.16.1-orange)](https://www.tensorflow.org/)
[![Allen Mouse Brain Atlas](https://img.shields.io/badge/Reference-Allen%20CCF-informational)](https://atlas.brain-map.org/)

**GPS-seq** (*Global-brain Position System coupled with RNA sequencing*) is an anatomically guided framework for brain-region-resolved transcriptomic profiling. The associated study combines deep-learning-based brain atlas registration, laser capture microdissection (LCM), and spatially barcoded transcriptome recovery to characterize molecular responses across anatomically defined mouse brain regions.

This repository provides the Python implementation of the GPS-seq brain-registration workflow together with installation scripts, Allen Mouse Brain Atlas preparation tools, and the resources required to reproduce the registration pipeline.

> **Current validation environment:** Python 3.9, TensorFlow 2.16.1, Keras 3.9.0, Neurite 0.2, and the GPSseq-compatible VoxelMorph source snapshot included in `third_party/`.

---

### GPS-seq workflow

![GPS-seq brain registration model: style transfer, spatial localization, and affine/nonlinear registration](assets/Fig2.png)

**Figure 1.** Overview of the GPS-seq framework and the deep-learning-based brain-registration module.

The registration component contains three major stages:

1. **Image translation / style transfer** using a ResNet-based network.
2. **Spatial localization** using a ConvNeXt-based network to identify the corresponding position in the Allen CCF.
3. **Atlas registration** using VoxelMorph-based affine and deformable registration.

The resulting anatomical annotation is then used to guide ROI selection and downstream laser capture microdissection.

### Software workflow

![GPS-seq software workflow: landmark selection, spatial calibration, ROI selection, and LCM export](assets/Fig1.png)

**Figure 2.** GPS-seq software workflow from anatomical registration and landmark-based calibration to ROI selection and LCM-compatible XML generation.

---

# Overview

GPSseq is organized as follows:

```text
GPSseq/
│
├── README.md
├── requirements.txt
│
├── assets/
│   └── Fig1.png
│
├── brain_registration/
│   ├── __init__.py
│   ├── ...
│   └── metadata/
│       └── ...
│
├── scripts/
│   ├── patch_neurite.py
│   └── verify_installation.py
│
├── third_party/
│   ├── SOURCE_INFORMATION.md
│   └── voxelmorph-gpsseq-legacy.zip
│
└── tutorials/
    ├── 01_install_environment.ipynb
    └── 02_prepare_allen_atlas.ipynb
```

Large atlas files and trained model weights are **not** stored in the GitHub repository. They are downloaded separately during setup.

---

# Quick start

## Prerequisites

GPSseq is currently validated on **Windows with Python 3.9**.

You should have:

- Windows 10/11 or another supported Python 3.9 environment;
- Conda or Miniconda;
- Git (recommended);
- sufficient disk space for the Allen atlas data and model weights.

You do **not** need to install Python separately if you create the environment through Conda.

---

## 1. Create the GPSseq environment

Open **Anaconda Prompt** or PowerShell and run:

```bash
conda create -n gpsseq python=3.9 -y
conda activate gpsseq
python --version
```

The last command should report Python `3.9.x`.

---

## 2. Download the GPSseq repository

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/GPSseq.git
cd GPSseq
```

Replace `YOUR_USERNAME` with the GitHub account or organization that hosts GPSseq.

Alternatively, select **Code → Download ZIP** on GitHub and extract the repository.

All commands below should be executed from the **GPSseq repository root**.

---

## 3. Install the standard Python dependencies

With the `gpsseq` environment activated:

```bash
python -m pip install -r requirements.txt
```

# 4. Complete the VoxelMorph + Neurite compatibility setup

After Step 3, open:

```text
tutorials/01_install_environment.ipynb
```

Run the cells **in order** using the same `gpsseq` Python environment.

You can use either:

- VS Code with the Jupyter extension; or
- Jupyter Notebook/JupyterLab.

If you use browser-based Jupyter Notebook and it is not installed, run:

```bash
python -m pip install notebook
python -m notebook
```

Then open:

```text
tutorials/01_install_environment.ipynb
```

## What the installation notebook does

The notebook performs four important tasks.

### 4.1 Check the Python environment

It checks the active interpreter and installed versions of:

- Python
- TensorFlow
- Keras
- Neurite

This helps prevent accidentally running the notebook with a different Conda environment.

### 4.2 Install the GPSseq-compatible VoxelMorph

The notebook installs:

```text
third_party/voxelmorph-gpsseq-legacy.zip
```

using the already installed dependencies.

Conceptually, this is:

```bash
python -m pip install --no-deps --force-reinstall ./third_party/voxelmorph-gpsseq-legacy.zip
```

The `--no-deps` option is intentional: TensorFlow, Keras, Neurite and the other dependencies have already been installed from `requirements.txt`.

### 4.3 Patch Neurite 0.2

Neurite 0.2 contains legacy imports that are incompatible with the validated Keras 3 environment, for example:

```python
from tensorflow.keras.losses import mean_absolute_error as l1
from tensorflow.keras.losses import mean_squared_error as l2
```

GPSseq provides:

```text
scripts/patch_neurite.py
```

which updates these imports to the aliases used by the validated environment:

```python
from tensorflow.keras.losses import MAE as l1
from tensorflow.keras.losses import MSE as l2
```

The script creates a backup before modifying the installed Neurite files.

### 4.4 Verify the installation

The notebook runs:

```text
scripts/verify_installation.py
```

The verification checks that:

- TensorFlow 2.16.1 is installed;
- Keras 3.9.0 is installed;
- Neurite 0.2 is installed;
- VoxelMorph imports successfully;
- `VxmAffineFeatureDetector` exists;
- `VxmAffineFeatureDetector((256, 256))` can be instantiated.

A successful installation should end with:

```text
PASS: VxmAffineFeatureDetector exists
PASS: VxmAffineFeatureDetector initialized successfully

=== GPSseq compatibility setup PASSED ===
```

If this step fails, **do not continue to the Allen atlas download**. Resolve the environment problem first.

### Command-line alternative

The same setup can be performed from the repository root:

```bash
python -m pip install --no-deps --force-reinstall ./third_party/voxelmorph-gpsseq-legacy.zip
python scripts/patch_neurite.py --apply
python scripts/verify_installation.py
```

---

# 5. Download and prepare the Allen Mouse Brain Atlas

GPSseq uses public Allen Mouse Brain Atlas resources as registration references.

The large atlas files are **not committed to GitHub**. Instead, GPSseq provides:

```text
tutorials/02_prepare_allen_atlas.ipynb
```

This notebook downloads the required files and prepares the local directory structure.

## 5.1 Required annotation mapping file

Before running the atlas notebook, make sure the following file exists:

```text
brain_registration/metadata/Allen_nrrd/brain_region_annotations.csv
```

The CSV must contain these columns:

```text
ARA structure ID
ARA abbreviation
```

This file is used only for mapping Allen structure IDs to the anatomical abbreviations used by GPSseq.

> **Important:** `brain_region_annotations.csv` is not one of the Allen files downloaded by the notebook. The GPSseq repository should provide this file, or provide a separate download location if redistribution is not permitted.

---

## 5.2 Run the atlas preparation notebook

Open:

```text
tutorials/02_prepare_allen_atlas.ipynb
```

Run the cells in order.

The notebook is intentionally configured with:

```python
RUN_DOWNLOADS = False
```

as the safe default.

This prevents a user from accidentally downloading the complete atlas simply by running every notebook cell.

After confirming the output folders and available disk space, change:

```python
RUN_DOWNLOADS = True
```

Then run the download section.

For the first test, we recommend:

```python
MASK_LIMIT = 5
```

After the test succeeds, change it to:

```python
MASK_LIMIT = None
```

to download all available structure masks.

---

## 5.3 Files downloaded from the Allen Institute

The notebook downloads the following public reference data.

| File | Resolution | Purpose |
|---|---:|---|
| `annotation_10.nii.gz` | 10 µm | Allen anatomical annotation volume |
| `average_template_10.nii.gz` | 10 µm | Allen reference template |
| `annotation_boundary_10.nii.gz` | 10 µm | Anatomical boundary reference |
| `average_template_25.nrrd` | 25 µm | 25-µm Allen reference template |
| `annotation_25.nrrd` | 25 µm | 25-µm Allen annotation volume |
| `structure_*.nrrd` | 25 µm | Individual anatomical structure masks |

The direct download locations are:

```text
https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/image_volumes/Allen-CCF-2020/20250331/annotation_10.nii.gz

https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/image_volumes/Allen-CCF-2020/20250331/average_template_10.nii.gz

https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/image_volumes/Allen-CCF-2020/20250331/annotation_boundary_10.nii.gz

https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/average_template/average_template_25.nrrd

https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd

https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_masks/structure_masks_25/
```

The notebook uses the official file URLs and preserves the original Allen file formats.

---

## 5.4 Automatic structure-mask renaming

The Allen directory contains files such as:

```text
structure_1014.nrrd
```

GPSseq does **not** match the entire filename against the CSV.

Instead, it extracts the number after:

```text
structure_
```

so:

```text
structure_1014.nrrd
        ↓
structure ID = 1014
```

It then searches for:

```text
ARA structure ID = 1014
```

in:

```text
brain_region_annotations.csv
```

and obtains the corresponding:

```text
ARA abbreviation
```

The resulting mask is renamed to:

```text
<ARA abbreviation>.nrrd
```

For example:

```text
structure_1014.nrrd
        ↓
1014
        ↓
ARA abbreviation from brain_region_annotations.csv
        ↓
<ARA abbreviation>.nrrd
```

This means that the actual final filename is determined by your CSV mapping.

Masks with missing, invalid, or ambiguous mappings are not blindly renamed. A manifest is written to:

```text
brain_registration/metadata/Allen_nrrd/mask_rename_manifest.csv
```

so that the mapping process can be inspected.

---

## 5.5 Expected atlas directory structure

After successful atlas preparation:

```text
brain_registration/
└── metadata/
    ├── Allen_nii/
    │   ├── annotation_10.nii.gz
    │   ├── average_template_10.nii.gz
    │   └── annotation_boundary_10.nii.gz
    │
    └── Allen_nrrd/
        ├── brain_region_annotations.csv
        ├── average_template_25.nrrd
        ├── annotation_25.nrrd
        ├── mask_rename_manifest.csv
        │
        └── mask/
            ├── <ARA abbreviation>.nrrd
            ├── <ARA abbreviation>.nrrd
            └── ...
```

---

# 6. Download the pretrained GPSseq model weights

The trained GPSseq model weights are **not stored in this GitHub repository** because of their large file size.

The required weight files should be downloaded from the permanent model-data repository selected for the project, for example Zenodo, Figshare, or another approved research-data host.

> **Before publishing this README, replace the placeholder below with the final public model-weight URL.**

### Model weights

**Download:** `TODO: ADD MODEL-WEIGHTS URL`

After downloading, place the weight files in the directories expected by the GPSseq source code.

The required paths and filenames should match the paths used by:

```text
brain_registration/initialize_models.py
```

Do not rename the weight files unless the corresponding paths in the source code are also changed.

### Recommended release practice

For reproducibility, the model-weight release should provide:

- exact filenames;
- expected directory structure;
- file sizes;
- SHA256 checksums;
- model/version identifier;
- date of release.

For example:

```text
GPSseq_weights/
├── style_transfer/
│   └── ...
├── localization/
│   └── ...
├── affine/
│   └── ...
└── nonlinear/
    └── ...
```

**The exact structure above is only an example. Use the actual paths required by your `initialize_models.py`.**

---

# 7. Run GPSseq

After completing:

```text
Step 4  → VoxelMorph + Neurite setup
Step 5  → Allen atlas preparation
Step 6  → model-weight download
```

the GPSseq environment is ready.

## 7.1 Run through Jupyter Notebook

Open the GPSseq notebooks supplied with the repository and select the `gpsseq` Python environment as the kernel.

For example:

```python
from brain_registration import BrainRegistration

brain_reg = BrainRegistration(voxel="nrrd")
```

If the object is initialized successfully, the GPSseq registration model has loaded correctly.

The exact analysis notebook and workflow depend on the application being performed. Please follow the examples supplied in the `tutorials/` directory.

---

# 8. Optional: build a Windows executable

GPSseq also contains the files used to package the application as a Windows executable, including the PyInstaller specification and resource-path handling.

This is an **optional** route. Users who prefer the Python/Notebook workflow do not need to build an executable.

After the Python environment, Allen atlas data, and model weights have been prepared and tested, run the PyInstaller build from the GPSseq repository root.

If the repository contains:

```text
app.spec
```

run:

```bash
pyinstaller app.spec
```

The resulting application is normally placed under:

```text
dist/
```

The `hooks/` directory and the `resource_path(...)` helper should be retained if they are referenced by `app.spec` or the application code.

> **Important:** The executable does not remove the requirement for the external Allen atlas data and model weights. These resources still need to be placed in the locations expected by the application unless they are explicitly bundled into the executable.

---

# 9. Keep large files out of GitHub

The following files should normally **not** be committed to the source repository:

- Allen 10-µm NIfTI volumes;
- Allen 25-µm NRRD volumes;
- the complete structure-mask collection;
- trained GPSseq model weights;
- generated output files.

Add the corresponding paths to `.gitignore`:

```gitignore
# Allen reference data
brain_registration/metadata/Allen_nii/
brain_registration/metadata/Allen_nrrd/*.nrrd
brain_registration/metadata/Allen_nrrd/mask/
brain_registration/metadata/Allen_nrrd/mask_rename_manifest.csv

# Model weights
# Adjust these paths to the actual GPSseq weight directories.
# brain_registration/weights/

# Python cache
__pycache__/
*.py[cod]

# Jupyter
.ipynb_checkpoints/

# PyInstaller output
build/
dist/
*.spec.bak
```

Do **not** ignore source files that are required to reproduce the software.

---

# 10. Troubleshooting

## `ImportError: cannot import name 'mean_absolute_error'`

If you see:

```text
ImportError: cannot import name 'mean_absolute_error'
from 'tensorflow.keras.losses'
```

the Neurite compatibility patch has not been applied correctly.

From the GPSseq root:

```bash
python scripts/patch_neurite.py --apply
python scripts/verify_installation.py
```

Do not manually edit multiple files unless necessary.

---

## `VxmAffineFeatureDetector` does not exist

If you see:

```text
AttributeError:
module 'voxelmorph.tf.networks'
has no attribute 'VxmAffineFeatureDetector'
```

you are probably using the public PyPI VoxelMorph package rather than the GPSseq-compatible source snapshot.

Reinstall the bundled version:

```bash
python -m pip install --no-deps --force-reinstall ./third_party/voxelmorph-gpsseq-legacy.zip
python scripts/verify_installation.py
```

---

## `Cannot locate the GPSseq repository root`

The installation notebooks search for:

```text
requirements.txt
brain_registration/
third_party/
```

to locate the repository root.

Make sure the notebook is being run from inside the GPSseq project.

---

## `brain_region_annotations.csv` is missing

The Allen downloader requires:

```text
brain_registration/metadata/Allen_nrrd/brain_region_annotations.csv
```

before mask renaming can be performed.

Check that the file exists and contains:

```text
ARA structure ID
ARA abbreviation
```

---

## Allen mask files cannot be discovered

The downloader reads the official Allen directory listing and accepts filenames matching:

```text
structure_<integer>.nrrd
```

For example:

```text
structure_1014.nrrd
```

If the Allen directory structure changes in the future, the notebook may need to be updated.

---

# Reproducibility

For a reproducible GPSseq installation, record:

```text
Python version
TensorFlow version
Keras version
Neurite version
GPSseq VoxelMorph archive SHA256
GPSseq model-weight version
Allen atlas release/date
```

The bundled VoxelMorph source is documented in:

```text
third_party/SOURCE_INFORMATION.md
```

The current archive checksum is:

```text
06c4dff583805fbb2e74da6263470d4637ac97ee571376da5933f71ac0ec335c
```

---

# Third-party software and data

GPSseq uses third-party software and public reference resources.

In particular:

- **VoxelMorph** is included as a GPSseq-compatible historical source snapshot. See `third_party/SOURCE_INFORMATION.md` and the license included in the archive.
- **Neurite 0.2** is installed from PyPI and receives a small compatibility patch during GPSseq environment setup.
- **Allen Mouse Brain Atlas / Allen CCF** data are downloaded directly from Allen Institute resources.

Please retain the relevant third-party licenses, attribution, and source information when redistributing GPSseq.

---

# Citation

If you use GPSseq, please cite the associated manuscript:

> **GPS-seq maps brain-wide transcriptional networks to guide schizophrenia therapeutic strategy**

Full bibliographic information, DOI, and journal information should be added here once the manuscript is formally published.

---

# Contact

For questions about GPSseq, please use the contact information provided with the associated manuscript or open an issue in this GitHub repository.
