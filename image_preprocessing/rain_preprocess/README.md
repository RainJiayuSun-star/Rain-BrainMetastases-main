# IDiA Lab Unified Preprocessing Pipeline (rain_preprocess)
==========================================================

Welcome to the **Unified Preprocessing Pipeline**! This folder contains a highly modular, clean, and reusable Python framework designed to run standard clinical neuroimaging operations on diverse datasets (such as **Yale/Brain-Mets-Lung** and **UCSF-BSMR**) from a single execution interface.

Instead of running separate standalone scripts for co-registration, skull-stripping, N4 bias correction, warping, and normalization, you can now define your entire preprocessing sequence inside a simple **YAML configuration file** and run it in a single command.

---

## 📂 Folder Layout
*   `rain_preprocess.py`: The core pipeline orchestrator containing all processing algorithm wrappers.
*   `config.yaml`: The user-configurable parameters file mapping directories and step lists.
*   `README.md`: This documentation.

---

## ⚙️ How it Works & Selected Step Numbers

The pipeline assigns a standard running number to each key step in the **IDiA Lab Preprocessing Guideline**:

| Step Number | Operation Name | Description | Suffix Output |
| :--- | :--- | :--- | :--- |
| **Step 1** | **Co-registration** | Rigid mutual-information alignment of secondary modalities (FLAIR) to baseline (T1CE). | `_aligned.nii.gz` |
| **Step 2** | **Skull Stripping** | Automated brain tissue extraction using the HD-BET neural network. | `_SS.nii.gz` & `_SS_bet.nii.gz` |
| **Step 3** | **N4 Bias Correction** | Removes low-frequency RF coil intensity inhomogeneities using SimpleITK. | `_BC.nii.gz` |
| **Step 5** | **Template Warping** | Rigid/Affine multi-resolution alignment onto SRI-24 or MNI152 template space. | `_MNI.nii.gz` |
| **Step 6** | **Intensity Normalization** | Rescales brain voxels linearly (between 1st & 99th percentiles) to a standard scale (e.g., 0–4000). | `_IN.nii.gz` |

### 💡 The "Fork in the Road" Strategy
You can execute **any arbitrary combination** of steps by specifying them in your `config.yaml` steps list:
*   To prepare scans in **Native Clean Space** (for models like *Met-Seg* or *Brain-SAM*):
    ```yaml
    steps: [1, 2, 3] # Aligned, Skull-Stripped, Bias-Corrected in Patient Space
    ```
*   To prepare scans in **Standardized MNI/SRI Space** (for models like *AURORA* or *Neuroverse3D*):
    ```yaml
    steps: [1, 2, 3, 5, 6] # Aligned, Skull-Stripped, Bias-Corrected, Template-aligned, Normalized
    ```
*   To run **N4 Bias Correction and Intensity Normalization Only**:
    ```yaml
    steps: [3, 6]
    ```

---

## 🛠️ Usage Instructions

### 1. Environment Activation
We recommend running this pipeline within the **`nnUnet`** Conda environment, which already has `SimpleITK`, `pandas`, `numpy`, and `PyTorch` pre-installed:
```bash
conda activate nnUnet
```

### 2. Configure `config.yaml`
Open `config.yaml` and adjust the paths and settings:
*   `input_dir`: Path to the input dataset containing patient subdirectories.
*   `output_dir`: Path where preprocessed outputs will be saved.
*   `steps`: An array of selected running step numbers (e.g. `[3, 6]`).
*   `device`: Set to `"cuda"` if you are running Step 2 (HD-BET) on a GPU-enabled machine, or `"cpu"` otherwise.

### 3. Run the Pipeline
To start the pipeline execution:
```bash
python rain_preprocess.py config.yaml
```

---

## 🐳 Docker Deployment

A pre-configured premium Docker environment with PyTorch (CUDA 12.1), SimpleITK, HD-BET, nnUNetv2, and standard Python image libraries is provided to ensure consistent and portable execution of the pipeline.

### 1. Build the Docker Image
From the repository root directory, run:
```bash
docker build -t rain_preprocess:latest -f image_preprocessing/rain_preprocess/Dockerfile image_preprocessing/rain_preprocess/
```

### 2. Run the Docker Container
Run the container with GPU support and mount your local directories (e.g., datasets and templates):
```bash
docker run --gpus all -it \
  -v /path/to/local/input:/data/input \
  -v /path/to/local/output:/data/output \
  -v /path/to/local/templates:/data/templates \
  rain_preprocess:latest
```

### 3. Run Preprocessing inside Container
Once inside the container shell, configure your `/workspace/rain_preprocess/config.yaml` to point to the mounted paths:
```yaml
input_dir: "/data/input"
output_dir: "/data/output"
template_path: "/data/templates/sri24_anatomy.nifti.zip"
device: "cuda"  # for GPU skull stripping
```
Then start the execution:
```bash
cd /workspace/rain_preprocess
python rain_preprocess.py config.yaml
```

---

## 📈 Intermediate Outputs & Logging
*   **Final Outputs**: Are mirrored perfectly by Patient ID directly under `output_dir/`.
*   **Intermediates**: To keep your final directory pristine, intermediate outputs of each step are stored separately under `output_dir/intermediates/step_X/` (e.g., `output_dir/intermediates/step_3/YG_0AXGKD8AFJGS/`).
*   **Zero-Dependency Fallback**: In case your active Python environment lacks PyYAML, the script includes a custom native parser fallback to parse the `config.yaml` seamlessly without throwing import crashes.
