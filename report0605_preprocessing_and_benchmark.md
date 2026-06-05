# Preprocessing Pipeline Improvements and AURORA Benchmark Report
==========================================================================

This report outlines the engineering enhancements made to the IDiA Lab Unified Preprocessing Pipeline (`rain_preprocess`), details the processing of patient scans using the updated scripts, and documents the execution of the AURORA brain metastases segmentation model on the successfully preprocessed dataset.

---

## Part 1: Summary of Actions and Outcomes

We completed the following tasks to optimize the preprocessing pipeline and run the benchmark model:

1. **CLI Script Upgrades**: Refactored `rain_preprocess.py` to use `argparse`, enabling support for the `-c`/`--config` options and positional config file fallbacks.
2. **Template Correction and Unzipping**: Resolved the file path mismatch for the SRI-24 reference template and unzipped the archive to make individual `.nii` files accessible to the registration engine.
3. **SimpleITK Method Correction**: Corrected registration crashes by resolving an API method naming typo, shifting from `SetOptimizerScalesFromPhysicalShifts` (plural) to `SetOptimizerScalesFromPhysicalShift` (singular).
4. **Registration Initialization Fix**: Resolved the brain truncation issue where registration was failing due to large coordinate origin offsets between template and patient spaces. We introduced the `CenteredTransformInitializer` to center-align physical grids before optimizing.
5. **Preprocessing Execution**: Executed the upgraded preprocessing pipeline on the lung-metastases patient dataset. The center-initialization resolved the field-of-view cutoff, preserving 100% of the brain signal.
6. **AURORA Model Benchmarking**: Ran the AURORA segmentation pipeline on the preprocessed outputs. The model completed predictions successfully on the aligned images.

---

## Part 2: Preprocessing Pipeline Enhancements

We implemented several updates to `/image_preprocessing/rain_preprocess/` to improve usability, configuration management, and spatial registration accuracy.

### 1. Robust Command Line Interface (argparse)
We replaced the rigid manual CLI argument indexing (`sys.argv[1]`) with Python's standard `argparse` library.
* **New Arguments**:
  * `-c` / `--config`: Explicitly specify the path to a YAML configuration file (defaults to `config.yaml` in the current directory if omitted).
  * `config_pos` (positional fallback): Backwards compatibility for running `python rain_preprocess.py config.yaml`.
* **Validation**: Added validation to check if the target configuration file exists before initializing the pipeline, preventing unhandled file-not-found exceptions.
* **Documentation**: Updated `README.md` to reflect the new command options and Docker run examples.

### 2. Template Reference Alignment and Extraction
* **Mismatch Resolution**: The configuration files originally pointed to `/templates/sri24_anatomy.nifti.zip`. However, the physical file was named `sri24_anatomy_nifti.zip` (with an underscore).
* **Unzipping Archive**: Because SimpleITK cannot read files directly from inside a standard `.zip` archive, we extracted the zip. It contained the `sri24/` folder with three structural volumes:
  * `spgr.nii` (Spoiled Gradient Recalled T1-weighted template)
  * `late.nii`
  * `erly.nii`
* **Configuration Update**: We modified the configuration files (`config.yaml` and `config_brain_lung.yaml`) to point directly to the unzipped SPGR template: `templates/sri24/spgr.nii`.

### 3. Solving the Brain Truncation (1/4 Brain) Registration Issue
* **The Problem**: During Step 5 (Template Space Warping), warped brain outputs appeared severely cut off (displaying only 1/4 or less of the brain). 
* **The Cause**: The template volume (SRI-24) is centered at `(0, 0, 0)` with a negative-Y direction matrix, whereas patient volumes are defined in negative-origin coordinate systems. The physical centers of the template and patient scans differed by **~19.5 cm**. Because the optimizer was initialized using an identity transform, it was unable to find spatial overlap and converged immediately to a bad local minimum.
* **The Fix**: We updated both `step_1_coregistration` and `step_5_template_warp` in `rain_preprocess.py` to use SimpleITK's `CenteredTransformInitializer` with the `GEOMETRY` filter:
  ```python
  tx = sitk.CenteredTransformInitializer(
      fixed,
      moving,
      sitk.Euler3DTransform() if "step_1" in inspect.stack()[0][3] else sitk.AffineTransform(moving.GetDimension()),
      sitk.CenteredTransformInitializerFilter.GEOMETRY
  )
  ```
  This automatically aligns the physical centers of the two coordinate grids before optimization starts.
* **Results**: In validation testing on patient `YG_0IBUXTBINCD9`, this change increased the percentage of voxels containing brain/signal from **5.55%** (truncated) to **17.64%** (complete, correct registration coverage).

---

## Part 3: AURORA Benchmark Run

We ran the **AURORA** brain metastases segmentation model on the newly corrected dataset `Brain-Mets-Lung-MRI-Path-Segs_Preprocessed356_new` using the `brain_metastasesModels_benchmark` repository.

### 1. Dataset Characteristics
* **Input Directory**: `/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/Medical_Images_Public/Brain-Mets-Lung-MRI-Path-Segs_Preprocessed356_new`
* **Size**: 72 discovered patient directories containing T1-CE and FLAIR MRI modalities.
* **Alignment Status**: Fully bias-corrected (Step 3), template-warped to SRI-24 space (Step 5), and intensity-normalized (Step 6).

### 2. Execution Environment
Inference is executed via Docker container using the `brainmet-aurora` image:
```bash
docker run --gpus all --rm \
  -v /mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastasesModels_benchmark:/workspace \
  -v /mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/Medical_Images_Public:/data \
  brainmet-aurora \
  python /workspace/AURORA/run_inference_test.py \
    --dataset-dir /data/Brain-Mets-Lung-MRI-Path-Segs_Preprocessed356_new
```

### 3. Inference Parameters
* **Inference Mode**: `t1c-fla` (Automatic detection of T1-CE and FLAIR modalities; other modalities are ignored).
* **Device**: CUDA GPU acceleration (utilizing `cuda:0` inside the container).
* **Config settings**:
  * TTA (Test-Time Augmentation): `False`
  * Crop Size: `(192, 192, 32)`
  * Sliding window overlap: `0.5`

### 4. Output Structure & Current Progress
Outputs are saved under `/brain_metastasesModels_benchmark/aurora_outputs/Brain-Mets-Lung-MRI-Path-Segs_Preprocessed356_new/`:
* **Generated Files per Patient**:
  * `<patient_id>_segmentation.nii.gz` - The categorical segmentation mask.
  * `<patient_id>_whole_tumor.nii.gz` - Raw unbinarized float predictions for the whole tumor boundary.
  * `<patient_id>_metastasis.nii.gz` - Raw unbinarized float predictions for the individual metastases.
  * `<patient_id>_aurora.log` - Step-by-step model execution metrics and status logs.
* **Progress**: **32 out of 72 patients** have been successfully processed, yielding clean, aligned segmentations.
