# -*- coding: utf-8 -*-
"""
IDiA Lab Unified Preprocessing Pipeline
============================================================
A highly robust, configurable, and reusable pipeline that wraps co-registration,
skull stripping (HD-BET), N4 bias field correction, template space registration,
and percentile intensity normalization into a single, unified execution chain.

Requirements:
    - SimpleITK, pandas, numpy (Standard clinical imaging libraries)
    - PyYAML (Optional, with custom native YAML parser fallback built-in)
    - HD-BET (Required only if running Step 2: Skull Stripping)
"""

import os
import sys
import time
import gc
import json
import traceback
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import SimpleITK as sitk

# Try importing YAML, fallback to custom parser if missing
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Try importing torch/HD-BET, warn if missing (only needed for Step 2)
try:
    import torch
    from HD_BET.hd_bet_prediction import hdbet_predict
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from HD_BET.paths import folder_with_parameter_files
    HAS_HDBET = True
except ImportError:
    HAS_HDBET = False


# ==========================================
# 0. Custom Native YAML Parser Fallback
# ==========================================
def parse_yaml_fallback(filepath):
    """
    A lightweight, zero-dependency parser for YAML files.
    Allows loading configurations without PyYAML installed.
    """
    config = {}
    current_key = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                parts = line.split(':', 1)
                k = parts[0].strip()
                v = parts[1].strip()
                
                # Check if it starts a list or dict block
                if not v:
                    current_key = k
                    config[k] = [] if k == 'steps' else {}
                else:
                    # Clean inline comments
                    if ' #' in v:
                        v = v.split(' #', 1)[0].strip()
                    # Parse basic types
                    if v.lower() == 'true':
                        v = True
                    elif v.lower() == 'false':
                        v = False
                    elif v.lower() == 'none':
                        v = None
                    elif v.startswith('[') and v.endswith(']'):
                        # Parse lists like [3, 6] or [2, 3, 5, 6]
                        v = [int(x.strip()) for x in v[1:-1].split(',') if x.strip()]
                    else:
                        try:
                            if '.' in v:
                                v = float(v)
                            else:
                                v = int(v)
                        except ValueError:
                            # Keep as clean string
                            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                                v = v[1:-1]
                    
                    if current_key and isinstance(config[current_key], dict):
                        config[current_key][k] = v
                    else:
                        config[k] = v
                        current_key = None
            elif line.startswith('-') and current_key == 'steps':
                val = line[1:].strip()
                if val:
                    config['steps'].append(int(val))
    return config


# ==========================================
# 1. Pipeline Action Functions
# ==========================================

def step_1_coregistration(img_path, ref_path, output_path):
    """
    Step 1: Mutual-Information-based Rigid Co-registration.
    Aligns a moving image (FLAIR/Mask) to a fixed baseline (T1CE/T1post).
    """
    print(f"      [Step 1] Aligning {os.path.basename(img_path)} -> {os.path.basename(ref_path)}")
    fixed = sitk.ReadImage(ref_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(img_path, sitk.sitkFloat32)
    
    # Standard mutual info rigid registration configuration
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.15)
    R.SetInterpolator(sitk.sitkLinear)
    
    # Rigid transformation (6 degrees of freedom in 3D)
    tx = sitk.Euler3DTransform()
    R.SetInitialTransform(tx, inPlace=True)
    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, estimateLearningRate=R.EachIteration)
    R.SetOptimizerScalesFromPhysicalShifts()
    
    # Run optimization
    out_tx = R.Execute(fixed, moving)
    
    # Resample moving image to fixed reference grid
    aligned = sitk.Resample(moving, fixed, out_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
    sitk.WriteImage(aligned, output_path)
    print(f"      [Step 1 Saved] Co-registered scan: {os.path.basename(output_path)}")


def step_2_skull_stripping(img_path, output_img, output_mask, device="cpu"):
    """
    Step 2: HD-BET Skull Stripping and Brain Extraction.
    """
    if not HAS_HDBET:
        raise ImportError("HD-BET package or dependencies not installed. Ensure torch and hd-bet are in your environment.")
    
    print(f"      [Step 2] Skull Stripping {os.path.basename(img_path)}")
    dev = torch.device(device)
    
    # Setup custom predictor matching lab implementation
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=(device != "cpu"),
        device=dev,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        folder_with_parameter_files, 
        use_folds='all',
        checkpoint_name='checkpoint_final.pth'
    )
    
    # Execute prediction
    hdbet_predict(img_path, output_img, predictor, keep_brain_mask=True)
    
    # Re-save the mask to the correct output path if needed
    hdbet_mask_default = output_img.replace(".nii.gz", "_bet.nii.gz")
    if os.path.exists(hdbet_mask_default):
        if hdbet_mask_default != output_mask:
            os.rename(hdbet_mask_default, output_mask)
        print(f"      [Step 2 Saved] Extracted Brain: {os.path.basename(output_img)}")
        print(f"      [Step 2 Saved] Brain Mask: {os.path.basename(output_mask)}")
    else:
        print(f"      [Warning] Mask not automatically found at {hdbet_mask_default}")


def step_3_n4_bias_correction(img_path, mask_path, output_path, shrink_factor=2, num_iters=50, convergence_thresh=1e-6):
    """
    Step 3: N4 Bias Field Correction using SimpleITK.
    """
    print(f"      [Step 3] Running N4 Bias Correction on {os.path.basename(img_path)}")
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    
    # Read mask if available; otherwise generate a simple bounding mask
    if mask_path and os.path.exists(mask_path):
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    else:
        print("      [Warning] No skull mask found. Generating automatic foreground mask...")
        mask = sitk.OtsuThreshold(img, 0, 1)
        
    # Shrunk image for faster bias estimation
    if shrink_factor > 1:
        img_shrunk = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
        mask_shrunk = sitk.Shrink(mask, [shrink_factor] * img.GetDimension())
    else:
        img_shrunk, mask_shrunk = img, mask
        
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iters] * 4)
    corrector.SetConvergenceThreshold(convergence_thresh)
    corrector.Execute(img_shrunk, mask_shrunk)
    
    log_bias_field = corrector.GetLogBiasFieldAsImage(img)
    corrected_img = img / sitk.Exp(log_bias_field)
    
    sitk.WriteImage(corrected_img, output_path)
    print(f"      [Step 3 Saved] Bias Corrected: {os.path.basename(output_path)}")


def step_5_template_warp(img_path, template_path, output_path):
    """
    Step 5: Diffeomorphic/Affine Registration to standard Template Space (MNI152 or SRI-24).
    Uses SimpleITK multi-resolution image registration.
    """
    print(f"      [Step 5] Registering {os.path.basename(img_path)} -> {os.path.basename(template_path)}")
    fixed = sitk.ReadImage(template_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(img_path, sitk.sitkFloat32)
    
    # Rigid + Affine Multi-resolution registration to Template
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10)
    R.SetInterpolator(sitk.sitkLinear)
    
    tx = sitk.AffineTransform(moving.GetDimension())
    R.SetInitialTransform(tx, inPlace=True)
    R.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=80, estimateLearningRate=R.EachIteration)
    R.SetOptimizerScalesFromPhysicalShifts()
    
    out_tx = R.Execute(fixed, moving)
    warped = sitk.Resample(moving, fixed, out_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
    sitk.WriteImage(warped, output_path)
    print(f"      [Step 5 Saved] Warped to Template: {os.path.basename(output_path)}")


def step_6_intensity_normalization(img_path, mask_path, output_path, lower_p=0.01, upper_p=0.99, target_max=4000.0):
    """
    Step 6: Percentile-based Intensity Normalization.
    """
    print(f"      [Step 6] Normalizing intensities on {os.path.basename(img_path)}")
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    
    if mask_path and os.path.exists(mask_path):
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    else:
        mask = sitk.OtsuThreshold(img, 0, 1)
        
    img_array = sitk.GetArrayFromImage(img)
    mask_array = sitk.GetArrayFromImage(mask)
    brain_voxels = img_array[mask_array > 0]
    
    p_low = pd.Series(brain_voxels).quantile(lower_p)
    p_high = pd.Series(brain_voxels).quantile(upper_p)
    
    normalized_img = (img - p_low) * (target_max / (p_high - p_low))
    normalized_img = sitk.Clamp(normalized_img, sitk.sitkFloat32, 0.0, target_max)
    
    sitk.WriteImage(normalized_img, output_path)
    print(f"      [Step 6 Saved] Normalized: {os.path.basename(output_path)}")


# ==========================================
# 2. Main Orchestrator Loop
# ==========================================

def run_pipeline(config_path):
    print("==================================================")
    print("      IDiA UNIFIED PREPROCESSING PIPELINE         ")
    print("==================================================\n")
    
    # 1. Load config
    if HAS_YAML:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded YAML configuration from PyYAML: {config_path}")
    else:
        config = parse_yaml_fallback(config_path)
        print(f"Loaded YAML configuration via custom fallback: {config_path}")
        
    input_dir = config.get('input_dir')
    output_dir = config.get('output_dir')
    steps = config.get('steps', [1, 2, 3, 5, 6])
    device = config.get('device', 'cpu')
    
    # Algorithm details
    n4_params = config.get('n4_params', {'shrink_factor': 2, 'num_iters': 50, 'convergence_thresh': 1e-6})
    in_params = config.get('in_params', {'lower_percentile': 0.01, 'upper_percentile': 0.99, 'target_max': 4000.0})
    template_path = config.get('template_path', '')
    
    print("\n--- Configurations ---")
    print(f"  Input Root:  {input_dir}")
    print(f"  Output Root: {output_dir}")
    print(f"  Selected Pipeline Steps: {steps}")
    print(f"  Compute Device: {device}")
    print(f"  Template Reference: {template_path if template_path else 'None (Required for Step 5)'}")
    print("----------------------\n")
    
    if not os.path.exists(input_dir):
        print(f"FATAL: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Intermediate directories setup
    intermediates_root = os.path.join(output_dir, "intermediates")
    os.makedirs(intermediates_root, exist_ok=True)
    
    # Discover patient directories
    all_entries = os.listdir(input_dir)
    patient_ids = sorted([
        d for d in all_entries 
        if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith('.')
    ])
    
    print(f"Discovered {len(patient_ids)} candidate patient folders.")
    
    start_total_time = time.time()
    errors_list = []
    
    # Iterate through patients
    for idx, pid in enumerate(patient_ids):
        print(f"\n[{idx+1}/{len(patient_ids)}] Processing Patient: {pid} ...")
        patient_start = time.time()
        
        try:
            # Core modality mapping for Yale / UCSF matching
            p_in_dir = os.path.join(input_dir, pid)
            all_p_files = os.listdir(p_in_dir)
            
            # Auto-detect modal sequences
            flair_file = next((f for f in all_p_files if "flair" in f.lower() and f.endswith(".nii.gz")), None)
            t1ce_file = next((f for f in all_p_files if ("t1ce" in f.lower() or "t1post" in f.lower()) and f.endswith(".nii.gz")), None)
            mask_file = next((f for f in all_p_files if "mask" in f.lower() or "bet" in f.lower()), None)
            
            # Fallback if no specific naming, just grab first two files
            if not t1ce_file and len(all_p_files) > 0:
                t1ce_file = [f for f in all_p_files if f.endswith(".nii.gz")][0]
            
            if not t1ce_file:
                print(f"   [Warning] No readable NIfTI scans found for patient {pid}. Skipping.")
                continue
                
            print(f"   Detected Sequences:")
            print(f"      - Reference/T1-post: {t1ce_file}")
            print(f"      - Moving/FLAIR:      {flair_file if flair_file else 'N/A'}")
            print(f"      - Pre-existing Mask: {mask_file if mask_file else 'N/A'}")
            
            # Setup intermediate track
            current_t1_path = os.path.join(p_in_dir, t1ce_file)
            current_flair_path = os.path.join(p_in_dir, flair_file) if flair_file else None
            current_mask_path = os.path.join(p_in_dir, mask_file) if mask_file else None
            
            # ==========================================
            # RUN STEPS SEQUENTIALLY
            # ==========================================
            
            # --- STEP 1: CO-REGISTRATION ---
            if 1 in steps:
                print("   >> Running Step 1: Modality Co-registration...")
                if current_flair_path:
                    step_dir = os.path.join(intermediates_root, "step_1", pid)
                    os.makedirs(step_dir, exist_ok=True)
                    out_flair_aligned = os.path.join(step_dir, flair_file.replace(".nii.gz", "_aligned.nii.gz"))
                    step_1_coregistration(current_flair_path, current_t1_path, out_flair_aligned)
                    current_flair_path = out_flair_aligned
                else:
                    print("      [Skipping Step 1] FLAIR sequence not available for registration.")
            
            # --- STEP 2: SKULL STRIPPING ---
            if 2 in steps:
                print("   >> Running Step 2: HD-BET Skull Stripping...")
                step_dir = os.path.join(intermediates_root, "step_2", pid)
                os.makedirs(step_dir, exist_ok=True)
                
                out_ss_t1 = os.path.join(step_dir, t1ce_file.replace(".nii.gz", "_SS.nii.gz"))
                out_mask = os.path.join(step_dir, t1ce_file.replace(".nii.gz", "_SS_bet.nii.gz"))
                
                step_2_skull_stripping(current_t1_path, out_ss_t1, out_mask, device=device)
                
                current_t1_path = out_ss_t1
                current_mask_path = out_mask
                
                # If flair exists, strip it using the generated mask
                if current_flair_path:
                    flair_img = sitk.ReadImage(current_flair_path, sitk.sitkFloat32)
                    brain_mask = sitk.ReadImage(current_mask_path, sitk.sitkUInt8)
                    flair_stripped = sitk.Mask(flair_img, brain_mask)
                    
                    out_ss_flair = os.path.join(step_dir, flair_file.replace(".nii.gz", "_SS.nii.gz"))
                    sitk.WriteImage(flair_stripped, out_ss_flair)
                    current_flair_path = out_ss_flair
            
            # --- STEP 3: N4 BIAS CORRECTION ---
            if 3 in steps:
                print("   >> Running Step 3: N4 Bias Field Correction...")
                step_dir = os.path.join(intermediates_root, "step_3", pid)
                os.makedirs(step_dir, exist_ok=True)
                
                # Bias correct T1CE
                out_bc_t1 = os.path.join(step_dir, os.path.basename(current_t1_path).replace(".nii.gz", "_BC.nii.gz"))
                step_3_n4_bias_correction(
                    current_t1_path, current_mask_path, out_bc_t1,
                    shrink_factor=n4_params['shrink_factor'],
                    num_iters=n4_params['num_iters'],
                    convergence_thresh=n4_params['convergence_thresh']
                )
                current_t1_path = out_bc_t1
                
                # Bias correct FLAIR
                if current_flair_path:
                    out_bc_flair = os.path.join(step_dir, os.path.basename(current_flair_path).replace(".nii.gz", "_BC.nii.gz"))
                    step_3_n4_bias_correction(
                        current_flair_path, current_mask_path, out_bc_flair,
                        shrink_factor=n4_params['shrink_factor'],
                        num_iters=n4_params['num_iters'],
                        convergence_thresh=n4_params['convergence_thresh']
                    )
                    current_flair_path = out_bc_flair
            
            # --- STEP 5: TEMPLATE SPACE WARP ---
            if 5 in steps:
                print("   >> Running Step 5: Template Space Warping...")
                if template_path and os.path.exists(template_path):
                    step_dir = os.path.join(intermediates_root, "step_5", pid)
                    os.makedirs(step_dir, exist_ok=True)
                    
                    # Warp T1CE
                    out_warp_t1 = os.path.join(step_dir, os.path.basename(current_t1_path).replace(".nii.gz", "_MNI.nii.gz"))
                    step_5_template_warp(current_t1_path, template_path, out_warp_t1)
                    current_t1_path = out_warp_t1
                    
                    # Warp FLAIR
                    if current_flair_path:
                        out_warp_flair = os.path.join(step_dir, os.path.basename(current_flair_path).replace(".nii.gz", "_MNI.nii.gz"))
                        step_5_template_warp(current_flair_path, template_path, out_warp_flair)
                        current_flair_path = out_warp_flair
                        
                    # Warp Mask if available
                    if current_mask_path:
                        out_warp_mask = os.path.join(step_dir, os.path.basename(current_mask_path).replace(".nii.gz", "_MNI.nii.gz"))
                        step_5_template_warp(current_mask_path, template_path, out_warp_mask)
                        current_mask_path = out_warp_mask
                else:
                    print("      [Skipping Step 5] Valid template_path reference was not provided in the YAML configuration.")
            
            # --- STEP 6: INTENSITY NORMALIZATION ---
            if 6 in steps:
                print("   >> Running Step 6: Percentile Intensity Normalization...")
                step_dir = os.path.join(intermediates_root, "step_6", pid)
                os.makedirs(step_dir, exist_ok=True)
                
                # Normalize T1CE
                out_norm_t1 = os.path.join(step_dir, os.path.basename(current_t1_path).replace(".nii.gz", "_IN.nii.gz"))
                step_6_intensity_normalization(
                    current_t1_path, current_mask_path, out_norm_t1,
                    lower_p=in_params['lower_percentile'],
                    upper_p=in_params['upper_percentile'],
                    target_max=in_params['target_max']
                )
                current_t1_path = out_norm_t1
                
                # Normalize FLAIR
                if current_flair_path:
                    out_norm_flair = os.path.join(step_dir, os.path.basename(current_flair_path).replace(".nii.gz", "_IN.nii.gz"))
                    step_6_intensity_normalization(
                        current_flair_path, current_mask_path, out_norm_flair,
                        lower_p=in_params['lower_percentile'],
                        upper_p=in_params['upper_percentile'],
                        target_max=in_params['target_max']
                    )
                    current_flair_path = out_norm_flair
            
            # ==========================================
            # MOVE FINAL PRODUCTS TO OUTPUT ROOT
            # ==========================================
            p_out_final_dir = os.path.join(output_dir, pid)
            os.makedirs(p_out_final_dir, exist_ok=True)
            
            # Save final files nicely named
            t1_final_name = os.path.basename(current_t1_path)
            sitk.WriteImage(sitk.ReadImage(current_t1_path), os.path.join(p_out_final_dir, t1_final_name))
            print(f"   [Final Product Saved] T1-CE: {t1_final_name}")
            
            if current_flair_path:
                flair_final_name = os.path.basename(current_flair_path)
                sitk.WriteImage(sitk.ReadImage(current_flair_path), os.path.join(p_out_final_dir, flair_final_name))
                print(f"   [Final Product Saved] FLAIR: {flair_final_name}")
                
            if current_mask_path:
                mask_final_name = os.path.basename(current_mask_path)
                sitk.WriteImage(sitk.ReadImage(current_mask_path), os.path.join(p_out_final_dir, mask_final_name))
                print(f"   [Final Product Saved] Mask:  {mask_final_name}")
                
            elapsed = time.time() - patient_start
            print(f"   [Done] Successfully preprocessed patient {pid} in {elapsed:.1f}s.")
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"   [ERROR] Failed to preprocess patient {pid}: {error_msg}")
            traceback.print_exc(file=sys.stdout)
            errors_list.append({"patient": pid, "error": error_msg})
            
        finally:
            gc.collect()
            
    # Print ending summary
    total_time = time.time() - start_total_time
    print("\n==================================================")
    print("--- PIPELINE ENDING SUMMARY ---")
    print(f"Total patient files attempted: {len(patient_ids)}")
    print(f"Total errors: {len(errors_list)}")
    print(f"Total execution time: {total_time:.1f}s")
    if errors_list:
        print("\nErrors Logged:")
        for err in errors_list:
            print(f"  Patient {err['patient']}: {err['error']}")
    print("==================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rain_preprocess.py <path_to_config.yaml>")
        sys.exit(1)
        
    config_file = sys.argv[1]
    run_pipeline(config_file)
