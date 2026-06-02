# Performance Benchmark of Various Models for Brain Metastases Segmentation

## Models
### Pretrained Models for Brain Metastases Segmentations
- [AURORA](https://github.com/BrainLesion/AURORA): Automated Brain Metastasis Segmentation
    - Link to the paper: https://doi.org/10.1016/j.radonc.2022.11.014

- [Met-Seg](https://github.com/xmindflow/Met-Seg)
    - Link to the paper: https://doi.org/10.48550/arXiv.2407.14011

### Pretrainied Foundation Models in Brain Imaging
- [BrainFounder](https://github.com/lab-smile/BrainSegFounder)
    - Link to the paper: https://arxiv.org/abs/2406.10395
- [Brain-SAM](https://github.com/DLbrainsam/Brain-SAM)
    - Link to the paper: https://www.medrxiv.org/content/10.64898/2026.01.30.26345164v2.full 
- [In-Context Learning (ICL) for 3D Medical Imaging](https://github.com/hujiesi/Neuroverse3D)
    - Link to the paper: https://arxiv.org/abs/2503.02410

## Setups & Requirements

### 1. AURORA
*   **Required Modalities**: Primarily **T1-CE (Contrast-Enhanced T1)** and **T2-FLAIR**. T1-CE is optimal for core tumor segmentation, while T2-FLAIR hyperintensities are critical for segmenting surrounding peritumoral edema.
*   **Space Registration**: Requires strict spatial alignment and co-registration (often resampled to standard spacing, e.g., 1.0mm isotropic SRI-24 or MNI-152 space). Standard preprocessing in the BrainLesion Suite involves co-registration, skull-stripping, and N4 bias field correction.
*   **Setup Method**: **Conda environment**. Installed as a Python package via PyPI/GitHub (compatible with Python 3.8+) under a dedicated Conda virtual environment.

### 2. Met-Seg
*   **Required Modalities**: Requires **T1-CE (T1c)**, native **T1-weighted**, and **T2-FLAIR** modalities. Research shows that this specific combination yields superior segmentation results.
*   **Space Registration**: Requires multi-modal co-registration to align all sequence grids (T1, T1-CE, FLAIR) voxel-for-voxel.
*   **Setup Method**: **Conda environment** (PyTorch based). Setup involves cloning the `xmindflow/Met-Seg` repository and installing python package dependencies under a PyTorch Conda environment.

### 3. BrainSegFounder (BrainFounder)
*   **Required Modalities**: High flexibility—supports arbitrary 3D MRI modalities (e.g., T1, T2, FLAIR). Can adapt dynamically to downstream inputs with variable channels (e.g., if a channel is missing, it can process duplicated inputs).
*   **Space Registration**: Requires standardized preprocessing and alignment matching the pretraining cohort.
*   **Setup Method**: **Conda or Docker**. Runs on a standard deep learning PyTorch environment (with MONAI and Swin Transformer libraries). Can be set up via a dedicated Conda environment or compiled using a custom Dockerfile.

### 4. Brain-SAM
*   **Required Modalities**: 3D volumetric MRI (e.g., standard tumor or lesion sequences). Expects input formatted as `.npz` arrays containing images and ground truth segmentation arrays (`(D, W, H)` dimensions).
*   **Space Registration**: Expects preprocessed, aligned, and skull-stripped 3D volumes.
*   **Setup Method**: **Conda environment**. Clone the repository and install it in a dedicated Conda environment running **Python 3.10** and **PyTorch 2.3.1**:
    ```bash
    conda create -n brainsam python=3.10 -y
    conda activate brainsam
    pip install -e .
    ```

### 5. In-Context Learning (ICL) for 3D Medical Imaging (Neuroverse3D)
*   **Required Modalities**: Highly universal and modality-agnostic. Supports multiple 3D modalities (MRI-T1, MRI-T2, CT) and tasks (segmentation, denoising, inpainting) out-of-the-box by leveraging in-context exemplars (image-mask pairs).
*   **Space Registration**: Expects inputs structured and resampled in the standard **nnU-Net** format (`imagesTr` and `labelsTr` style spacing and layout).
*   **Setup Method**: **Docker (Highly Recommended) or Conda**. The repository provides a pre-configured Docker image containing all environments. Alternatively, it can be installed via a fast Pyproject virtual environment using `uv` (runs `uv sync` to install dependencies) or standard `pip install -r requirements.txt`.

---

## Docker Setup: Building & Running with Workspace Mounts

Since Docker environment setups are standard and highly recommended for local verification, you can build each model's Docker image and run it by mounting your local project workspace.

### 1. Building the Docker Images
From the project root directory (`/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_main`), build the Docker image for the desired model using the custom Dockerfiles stored in `benchmark/docker/`:

*   **AURORA**:
    ```bash
    docker build -t brainmet-aurora -f benchmark/docker/Dockerfile.aurora .
    ```
*   **Met-Seg**:
    ```bash
    docker build -t brainmet-metseg -f benchmark/docker/Dockerfile.metseg .
    ```
*   **BrainSegFounder**:
    ```bash
    docker build -t brainmet-brainfounder -f benchmark/docker/Dockerfile.brainfounder .
    ```
*   **Brain-SAM**:
    ```bash
    docker build -t brainmet-brainsam -f benchmark/docker/Dockerfile.brainsam .
    ```
*   **Neuroverse3D**:
    ```bash
    docker build -t brainmet-neuroverse3d -f benchmark/docker/Dockerfile.neuroverse3d .
    ```

> [!TIP]
> To build all five images sequentially in a single loop, run:
> ```bash
> for model in aurora metseg brainfounder brainsam neuroverse3d; do
>   docker build -t brainmet-$model -f benchmark/docker/Dockerfile.$model .
> done
> ```

### 2. Mounting the Workspace and Running Containers
To execute a model container interactively while persisting files and scripts between WSL2/Windows and the container, use the `-v $(pwd):/workspace` mount. This maps your entire local workspace root to `/workspace` inside the container:

```bash
docker run --gpus all --ipc=host -it --name run-<model_name> \
  -v $(pwd):/workspace \
  brainmet-<model_name>
```

#### Path Mappings & Working in the Workspace:
*   **Workspace root** (`/mnt/.../brain_metastases_main`) &rarr; `/workspace`
*   **Benchmark directory** &rarr; `/workspace/benchmark`
*   Any file or script modified, or prediction saved inside `/workspace/benchmark/` within the container is instantly visible and persistent in your host's local `benchmark/` folder.
*   GPU acceleration and system memory pass-through are preconfigured via WSL2.

---


## Handling Your Benchmarking Datasets (UCSF & Yale)

When benchmarking, you will encounter two types of model requirements. Here is exactly how to adapt your two datasets to ensure successful runs:

### A. How to Run on MNI / SRI-24 Template-Space Models (AURORA, Neuroverse3D)
Since these models expect brain images to be aligned to a standardized atlas grid, you cannot feed native UCSF or Yale scans directly. Instead, implement a **Forward/Backward Warp Pipeline**:

1.  **Forward Warp**:
    *   **UCSF (UCSF-BSMR)**: Use the automated **BraTS Toolkit** (installed via pip) or an ANTs diffeomorphic script to warp the skull-stripped native post-contrast and FLAIR images onto the **SRI-24 atlas** space.
    *   **Yale (Brain-Mets-Lung)**: Warp the native T1CE and FLAIR scans into the **SRI-24** space.
2.  **Inference**:
    *   Pass the warped images into **AURORA** or **Neuroverse3D**. The model will cleanly output a high-quality tumor segmentation mask matching the SRI-24 template grid.
3.  **Reverse Warp (Recombination)**:
    *   To compare the model's predictions with your manual native-space segmentations (e.g., UCSF's `*_BraTS-seg.nii.gz` or Yale's `*_core_seg.nii.gz`), take the predicted mask and apply the **ANTs inverse warp matrix** (generated during Step 1) to bring the mask back to the patient's native space.
    *   Calculate your DSC, HD95, and IoU metrics directly in **native space**!

### B. How to Run on Native-Space Models (Met-Seg, Brain-SAM)
Since these models are trained in native space, the workflow is much faster:

*   **UCSF (UCSF-BSMR)**: The dataset is already co-registered (T1-pre and FLAIR aligned to T1-post) and skull-stripped. It is 100% ready to be fed directly into **Met-Seg** or converted to `.npz` for **Brain-SAM** with zero extra warping!
*   **Yale (Brain-Mets-Lung)**: 
    *   If using the raw PACS exports where FLAIR and T1CE are in different native grids, run our newly implemented **`logging_COREG_RainScript.py`** alignment script first to align the FLAIR to the T1CE baseline grid.
    *   Once co-registered, they are 100% ready to be fed directly into **Met-Seg** or converted into `.npz` format for **Brain-SAM**.