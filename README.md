# Brain Metastases Main Repository
Author: Rain Sun
Supervised by Jordan Bramble @ UW Madison

**Objectives**: Develop state-of-art Brain Metastases Segmentation Tool for clinical use using advanced Deep Learning techniques.

**Procedures**
- Data Collection and Proprocessing
    - Image co-registering
    - Image preprocessing following the IDIA lab guideline
    - Labeling checking
- Training
    - Architecture
    - Hyperparameter & Finetuning
    - Data augmentation
    - Evaluation (training loss/validation & performance)
- Benchmarking
    - Using what dataset
    - Using what models to compare againt
    - Evaluation Scores
        - Dicescore Coefficient (DSC, also called F1-score)
        - HD95 (95th percentile Hausdorff distance)
        - IoU (intersection over Union)

## Submodules

This repository uses [nnUNet_rain](https://github.com/RainJiayuSun-star/nnUNet_rain) as a Git submodule for model training using nnUnet architecture under `train/nnUNet_rain`.

**To clone this repository with the submodule included:**
```bash
git clone --recurse-submodules https://github.com/RainJiayuSun-star/Rain-BrainMetastases-main.git
```

**If you already cloned this repository without the submodule:**
You can fetch and initialize the submodule at any time by running the following command from the root of this repository:
```bash
git submodule update --init --recursive
```
