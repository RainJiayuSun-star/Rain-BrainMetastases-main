# IDiA Lab Brain Metastases Image Preprocessing Directory

This directory contains the pipeline guideline and automated scripts for preprocessing patient brain MRI scans (T1CE, FLAIR) and tumor segmentation masks to prepare them for clinical deep learning training and radiomics feature extraction.

---

## 1. Preprocessing Directory Structure

* **`Preprocessing Pipeline-May 2023 1.pdf`**: The official lab guideline detailing the step-by-step mathematical and coordinate transformations.
* **`align/`**: Contains scripts and outputs for Step 1 (co-registration and spatial alignment).
  * **`logging_COREG_RainScript.py`**: Automated pipeline script for co-registering FLAIR and tumor masks onto the baseline T1CE grid and modal stacking.
  * self note: use nnUnet conda env
* **`Jared/`**: Contains automated pipeline scripts implementing the core downstream preprocessing steps (Skull Stripping, Bias Correction, and Intensity Normalization).

---

## 2. Preprocessing Guideline to Workspace Script Mapping

Based on the **IDiA Preprocessing Pipeline Guideline**, here is the exact mapping of each step to the corresponding automated script in this workspace:

| Guideline Step | Description | Workspace Script / Source | Script File Location |
| :--- | :--- | :--- | :--- |
| **Step 1** | **Co-register all modalities** | **`logging_COREG_RainScript.py`** <br>*(Header resampling / Mutual-Info co-registration)* | [image_preprocessing/align/](file:///mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_main/image_preprocessing/align/logging_COREG_RainScript.py) |
| **Step 2** | **Do skull stripping** | **`logging_jbHDBET_BrambleScript.py`** <br>*(Automated neural network brain extraction using HD-BET)* | [image_preprocessing/Jared/](file:///mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_main/image_preprocessing/Jared/logging_jbHDBET_BrambleScript.py) |
| **Step 3** | **Perform bias correction on Step 2** | **`logging_jbBiasCorrection_BrambleScript_V2.py`** <br>*(Automated N4 Bias Field Correction)* | [image_preprocessing/Jared/](file:///mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_main/image_preprocessing/Jared/logging_jbBiasCorrection_BrambleScript_V2.py) |
| **Step 4** | **Obtain the tumor mask** | **Manual segmentation** *(provided in the dataset as `_core_seg` and `_whole_seg`)* or predicted using **`nnUnet`**. | `Medical_Images_Public/` / Deep Learning Model |
| **Step 5** | **Perform ANTs diffeomorphic transformation to MNI Space** | **ANTs Registration** <br>*(Using ANTsPy or ANTs command-line to warp native images/masks to standard MNI152 template space)* | Standard ANTs Registration Workflow |
| **Step 6** | **Perform intensity normalization on MNI result** | **`logging_jbIntensityNormalization_BrambleScript_V2.py`** <br>*(Voxel intensity standardization and scaling)* | [image_preprocessing/Jared/](file:///mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_main/image_preprocessing/Jared/logging_jbIntensityNormalization_BrambleScript_V2.py) |

---

## 3. Preprocessing Execution Flow

To preprocess your brain metastasis dataset correctly following the IDiA Lab pipeline, run the operations sequentially in this order:

```text
  [ Raw Images & Masks ]  -->  Step 1: Alignment (logging_COREG_RainScript.py)
                                           │
                                           ▼
                               Step 2: Skull Stripping (logging_jbHDBET_BrambleScript.py)
                                           │
                                           ▼
                               Step 3: Bias Correction (logging_jbBiasCorrection_BrambleScript_V2.py)
                                           │
                                           ▼
                               Step 5: ANTs Diffeomorphic Warp to MNI Space
                                           │
                                           ▼
                               Step 6: Intensity Normalization (logging_jbIntensityNormalization_BrambleScript_V2.py)
                                           │
                                           ▼
                           [ Final Preprocessed Volumes in MNI Space ]
                              (Ready for Feature Extraction / Training)
```

---

## 4. Quick Execution: How to run Step 1 (Co-registration)

To run the modal alignment and spatial resampling script using the lab's pre-configured SimpleITK Conda environment:

```bash
# Run the pipeline script using the nnUnet conda environment python
/home/rainsun/miniconda3/envs/nnUnet/bin/python /mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_main/image_preprocessing/align/logging_COREG_RainScript.py
```
This script reads the raw dataset, maps all moving images onto the baseline T1CE grid, stacks structural sequences into a multi-channel NIfTI volume (`_stacked_modalities.nii.gz`), and merges segmentation masks into a single-component categorical label map (`_multilabel_seg.nii.gz`) that is fully compatible with standard viewers like ITK-SNAP.

## Docker Environment

If you need to run the preprocessing pipeline in an isolated environment (such as your lab's virtual machine), you can use the provided `Dockerfile.imgpreprocess`. This Dockerfile sets up a lightweight Python 3.10 environment containing all the necessary clinical imaging libraries (`numpy`, `pandas`, `SimpleITK`, `PyYAML`, `fpdf2`, `openpyxl`).

### 1. Build the Docker Image
Navigate to the directory containing the Dockerfile (`image_preprocessing`) and run the following command to build the image. We will tag it as `imgpreprocess`:

```bash
cd /home/sunji/IDIA/Rain-BrainMetastases-main/image_preprocessing
docker build -t imgpreprocess -f Dockerfile.imgpreprocess .
```

### 2. Run the Container and Mount Your Data
To give the Docker container access to your code and datasets on the host machine, use the `-v` (volume) flag. This maps the host's `IDIA` directory to `/app` inside the container:

```bash
docker run -it -v /home/sunji/IDIA:/app imgpreprocess
```

### 3. Run Preprocessing Scripts Inside Docker
Once you are inside the Docker container's terminal, your files are located in `/app`. You can run the pipeline scripts directly from there. For example, to run the unified preprocessing pipeline:

```bash
python /app/Rain-BrainMetastases-main/image_preprocessing/rain_preprocess/rain_preprocess.py -c /app/your_config.yaml
```
*(Ensure any paths in your configuration YAML file are updated to point to the `/app/` directory paths instead of the host `/home/sunji/` paths!)*
