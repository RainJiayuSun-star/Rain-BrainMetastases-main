# IDiA Lab - Preprocessing & Training Pipeline Report (June 22, 2026)

## 1. Summary
This report documents the recent end-to-end execution of our unified MRI preprocessing pipeline and the subsequent configuration of the nnU-Net v2 training environment. The objective was to standardize the **Brain-Mets-Lung** and **UCSF** datasets through skull-stripping, bias correction, and spatial alignment, and then launch a distributed multi-GPU training job to train a highly optimized brain metastases segmentation model using the newly released nnU-Net v2 framework. 

---

## 2. Preprocessing Pipeline

### 2.1 Datasets Processed
We successfully formatted and preprocessed two major cohorts for training:
* **UCSF Brain Mets Dataset**
* **Brain-Mets-Lung Dataset**

### 2.2 Methodology: rain_preprocess vs. BrambleScript_V2
During the Bias Correction phase, we evaluated two separate processing implementations. Both scripts rely on the underlying SimpleITK `N4BiasFieldCorrectionImageFilter` to mathematically eliminate low-frequency RF coil inhomogeneities (using a shrink factor of 2, 50 iterations, and a convergence threshold of 1e-6). 

**Differences & Use Cases:**
* **`logging_jbBiasCorrection_BrambleScript_V2.py`:** Jared's standalone script is strictly dependent on the pre-existence of highly accurate brain masks (expecting `_SS_bet.nii.gz` files from a prior skull-stripping step) to restrict the N4 calculation. 
* **`rain_preprocess.py` (Unified Pipeline):** We utilized this script because **our dataset did not have existing brain masks available.** To solve this, `rain_preprocess.py` implements an automatic fallback mechanism: when no mask is provided, it applies **Otsu Thresholding** (`sitk.OtsuThreshold`) directly to the image on the fly. 

**Impact on Quality:**
Otsu thresholding automatically calculates the optimal pixel intensity to separate the bright foreground tissue (the head/brain) from the dark background (the empty air surrounding the head). By forcing the N4 algorithm to only calculate the bias field inside this Otsu-generated foreground mask, the algorithm doesn't waste computational fitting power trying to model the noisy, empty air. This drastically improves the mathematical accuracy and quality of the bias field estimation inside the actual brain parenchyma.

---

## 3. Model Training Setup (nnU-Net v2)

Following preprocessing, we migrated the data into the strict `nnUnet_raw` folder structure (separating `imagesTr` and `labelsTr`) and initiated the nnU-Net v2 pipeline on the lab's virtual machine.

### 3.1 Fingerprinting and Experiment Planning
nnU-Net is a "self-configuring" method. We executed `nnUNetv2_plan_and_preprocess`, which scanned our preprocessed NIfTI files to extract dataset fingerprints (image dimensions, voxel spacings, and class ratios). 

**VRAM Optimization:** By default, nnU-Net plans for standard 8GB GPUs, which resulted in a severely restricted batch size that crashed PyTorch's distributed engine. Because our lab server utilizes four massive 40GB NVIDIA L40 GPUs, we injected the environment variable `-e nnUNet_vram_target_GB=35` into the Docker container. This forced the mathematical heuristic to drastically scale up the 3D Patch Size and Batch Size, maximizing our hardware utilization and solving the crash.

### 3.2 Multi-GPU Training Execution (DDP)
We initiated the training using the `3d_fullres` configuration. The training job was launched inside a customized Docker container with the following engineering optimizations:

1. **Distributed Data Parallel (DDP):** We utilized the `-num_gpus 2` (and eventually 4) flag to spread the massive 3D batches across the L40 GPUs.
2. **Shared Memory Bottleneck Resolution:** Multithreaded data augmentation in PyTorch often crashes Docker due to the default 64MB `/dev/shm` limit. We resolved this `[Errno 28] No space left on device` error by allocating 32GB of system RAM to shared memory (`--shm-size=32g`), ensuring the GPUs are continuously fed with data.
3. **PyTorch Compilation (JIT):** nnU-Net v2 enables PyTorch 2.0's `torch.compile` by default to mathematically fuse kernels and speed up training by up to 20%. To support this, we updated our `Dockerfile` to install the `gcc` and `build-essential` C-compilers, allowing the neural network graph to successfully compile on the fly.

### 3.3 Training Status
*(Training is currently actively running across the GPU cluster. Loss metrics, pseudo-dice scores, and final validation results will be documented in this section upon completion of the 1000 epochs).*
