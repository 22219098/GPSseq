# GPSseq

### GPS-seq maps brain-wide transcriptional networks to guide schizophrenia therapeutic strategy

[![Allen Mouse Brain Atlas](https://img.shields.io/badge/Reference-Allen%20CCF-informational)](https://atlas.brain-map.org/)

**GPS-seq** (*Global-brain Position System coupled with RNA sequencing*) is an anatomically guided framework for brain-region-resolved transcriptomic profiling. The associated study combines deep learning-based brain atlas registration, region-of-interest (ROI) selection, laser capture microdissection (LCM), and spatially barcoded transcriptome recovery to characterize molecular responses across the mouse brain.

This repository provides the Python-based brain-registration software and supporting resources for the GPS-seq workflow. The instructions below cover **environment setup and preparation of the public Allen Mouse Brain Atlas reference data**. Instructions for downloading trained model weights and running the registration software will be added separately.

**Associated manuscript:** *GPS-seq maps brain-wide transcriptional networks to guide schizophrenia therapeutic strategy* (bibliographic information to be added when available).

## Overview

### Brain registration model

The GPS-seq brain-registration model integrates three stages: **(1) ResNet-based style transfer** to generate an atlas-like representation of the input histological image; **(2) ConvNeXt-based spatial localization** within the Allen brain atlas; and **(3) VoxelMorph-based affine and nonlinear registration** to align the reference image and transfer anatomical annotations.

![GPS-seq brain registration model: style transfer, vector identification, and affine/nonlinear registration](docs/model.png)

*Figure 1. Overview of the GPS-seq brain-registration model.*

### Atlas-guided ROI selection and LCM export

The software uses registered anatomical annotations to select ROIs, supports landmark-based spatial calibration, and prepares ROI information for **LCM-compatible XML export**.

![GPS-seq software workflow: landmark selection, spatial calibration, ROI selection, and XML generation for laser capture microdissection](docs/workflow.png)

*Figure 2. Registration-guided ROI selection, landmark calibration, and LCM export workflow.*

## Quick start

### 1. Create a Python 3.9 environment

We recommend using [Conda](https://docs.conda.io/) to keep GPSseq dependencies separate from other Python projects. **Conda or Miniconda should already be installed**; you do not need to install Python manually.

Open a terminal (or **Anaconda Prompt** on Windows) and run:

```bash
conda create -n gpsseq python=3.9 -y
conda activate gpsseq
python --version
```

The Python version should be `3.9.x`.

### 2. Download the GPSseq repository

```bash
git clone https://github.com/YOUR_USERNAME/GPSseq.git
cd GPSseq
```

> Replace `YOUR_USERNAME` with the GitHub account or organization that hosts GPSseq. Alternatively, choose **Code → Download ZIP** on the repository page and open a terminal in the extracted `GPSseq` folder.

### 3. Install Python dependencies

With the `gpsseq` environment activated and your terminal in the repository root:

```bash
python -m pip install -r requirements.txt
```

To open the data-preparation notebook, install Jupyter Notebook if it is not already included among the dependencies:

```bash
python -m pip install notebook
```

> Package compatibility, including GPU-specific dependencies, depends on your operating system and the versions listed in `requirements.txt`.

### 4. Download and prepare the Allen Mouse Brain Atlas

The reference atlas files are **not stored in this GitHub repository** because they are large and are distributed by the [Allen Institute](https://alleninstitute.org/). Instead, GPSseq provides a Jupyter notebook that downloads the official files into the required folders and renames anatomical structure masks according to an annotation table.

#### 4.1. Prepare the brain-region annotation table

Before running the notebook, make sure this file exists:

```text
brain_registration/metadata/Allen_nrrd/brain_region_annotations.csv
```

The CSV must contain the following columns (spelled exactly as shown):

- `ARA structure ID`
- `ARA abbreviation`

**Important:** `brain_region_annotations.csv` is a GPSseq mapping table and **is not downloaded from the six Allen URLs below**. The repository maintainer must supply it in the location shown above, or provide a separate download link if redistribution is restricted.

#### 4.2. Launch the downloader notebook

From the `GPSseq` root folder, run:

```bash
python -m notebook
```

Open [`tutorials/01_prepare_allen_atlas.ipynb`](tutorials/01_prepare_allen_atlas.ipynb) in the browser and execute its cells in order.

In the **Configure output folders** section, set the following options:

```python
RUN_DOWNLOADS = True
DOWNLOAD_NIFTI = True
DOWNLOAD_NRRD = True
DOWNLOAD_MASKS = True
RENAME_MASKS = True
MASK_LIMIT = None    # Download all available masks
```

**Recommended first test:** Set `MASK_LIMIT = 5`, run all cells, and confirm that the files download and rename correctly. Then change `MASK_LIMIT` to `None` and rerun the notebook to obtain the complete mask collection. Existing valid downloaded files and masks already recorded as renamed are normally skipped.

Downloads may take considerable time and require substantial free disk space, especially for the 10-µm image volumes and the full mask directory. Keep your internet connection stable.

#### 4.3. Public reference datasets

The notebook downloads the following files from Allen Institute resources:

| Dataset | Resolution | Official source |
| --- | --- | --- |
| `annotation_10.nii.gz` | 10 µm | [Allen Brain Cell Atlas](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/image_volumes/Allen-CCF-2020/20250331/annotation_10.nii.gz) |
| `average_template_10.nii.gz` | 10 µm | [Allen Brain Cell Atlas](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/image_volumes/Allen-CCF-2020/20250331/average_template_10.nii.gz) |
| `annotation_boundary_10.nii.gz` | 10 µm | [Allen Brain Cell Atlas](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/image_volumes/Allen-CCF-2020/20250331/annotation_boundary_10.nii.gz) |
| `average_template_25.nrrd` | 25 µm | [Allen Mouse CCF](https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/average_template/average_template_25.nrrd) |
| `annotation_25.nrrd` | 25 µm | [Allen Mouse CCF, 2017 annotations](https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd) |
| `structure_*.nrrd` | 25 µm | [All structure masks](https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_masks/structure_masks_25/) |

The notebook uses the original Allen file formats and does not alter the internal image or mask data.

#### 4.4. Automatic mask renaming

The source mask filenames contain an Allen structure ID. GPSseq extracts the number after `structure_` and matches it against `ARA structure ID` in `brain_region_annotations.csv`, then uses the corresponding `ARA abbreviation` as the new filename.

For example:

```text
structure_1014.nrrd → [abbreviation from CSV].nrrd
```

This is a naming-rule example; the actual abbreviation comes from your CSV. Masks without a valid or unique match keep their original names. Existing files are not overwritten. A `mask_rename_manifest.csv` report records the rename results.

#### 4.5. Expected folder structure

Once preparation is complete, the local project should contain:

```text
GPSseq/
├── README.md
├── requirements.txt
├── tutorials/
│   └── 01_prepare_allen_atlas.ipynb
├── docs/
│   ├── model.png
│   └── workflow.png
└── brain_registration/
    └── metadata/
        ├── Allen_nifti/
        │   ├── annotation_10.nii.gz
        │   ├── average_template_10.nii.gz
        │   └── annotation_boundary_10.nii.gz
        └── Allen_nrrd/
            ├── brain_region_annotations.csv
            ├── average_template_25.nrrd
            ├── annotation_25.nrrd
            ├── mask_rename_manifest.csv
            └── mask/
                ├── <ARA abbreviation>.nrrd
                └── ...
```

> The final mask filenames depend on the CSV mapping; masks with unmatched IDs may still be named `structure_<ID>.nrrd`. The output folders are created automatically by the notebook.

#### 4.6. Keep downloaded datasets out of Git

Download the atlas files locally rather than uploading them to the source-code repository. Add the following entries to your project's `.gitignore` (without deleting any rules you already have):

```gitignore
brain_registration/metadata/Allen_nifti/
brain_registration/metadata/Allen_nrrd/*.nrrd
brain_registration/metadata/Allen_nrrd/mask/
brain_registration/metadata/Allen_nrrd/mask_rename_manifest.csv
```

Keep `brain_region_annotations.csv` available to users **if you are permitted to distribute it**. Model-weight download instructions will follow in a later section.
