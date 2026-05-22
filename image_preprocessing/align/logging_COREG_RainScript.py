# ==========================================================================
# ENVIRONMENT SETUP PREAMBLE 
# ==========================================================================
# logging_COREG_RainScript.py — Co-registration, Resampling, and Modality Stacking
#
# This script co-registers MRI modalities (T1CE, FLAIR) and segmentations
# (core tumor, whole tumor) onto a single coordinate grid (T1CE fixed grid).
# It then stacks the structural modalities and the segmentation masks.
#
# It utilizes the lab skeleton template:
#   - Captures all console output into a PDF log file
#   - Tracks per-patient errors in an Excel file
#   - Robust patient discovery from directory scan
#   - Subset selection for quick testing
#
# Dependencies:
#   Python   >= 3.10
#   fpdf2    (PDF log generation)
#   pandas   (Error tracking Excel export)
#   openpyxl (Excel file backend for pandas)
#   SimpleITK (For image registration, resampling, and I/O)
# ==========================================================================

# ==========================================
# 1. Load libraries
# ==========================================

import sys                  # Used to hijack the standard output so print statements can be intercepted and saved.
import platform             # Used to retrieve the current OS and Python environment version details.
import importlib.metadata   # Used to dynamically check and print the installed versions of your libraries.
import traceback            # Used to format and print full, detailed error messages when the code crashes.
import pandas as pd         # Used to structure error logs into a table and easily export them to an Excel file.

# fpdf2 is the actively maintained successor to the abandoned 'fpdf' (PyFPDF) library.
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

import os                   # Used for standard operating system interfaces like checking file paths and creating directories.
import time                 # Used for tracking computation time.
import gc                   # Used for manual garbage collection after each iteration.

# Operation-specific imports
import SimpleITK as sitk    # For co-registration, resampling, stacking, and image I/O

# ==========================================
# 2. ATTENTION: User-Defined Inputs
# ==========================================

# A. Operation Identity
OPERATION_NAME = "Image Co-registration and Stacking"

# B. Operation Parameters
# resample_only:
#   True  - Performs header-based physical resampling (assumes physical coordinates in NIfTI
#           headers are pre-aligned, which is common for CLEAN public datasets). This is fast and distortion-free.
#   False - Runs a robust Mutual Information rigid co-registration (rotation + translation) 
#           to physically align FLAIR to T1CE in case of patient motion or scanner grid offset.
PARAMETERS = {
    "resample_only": True,
}

# C. Root Path Configuration
INPUT_ROOT = r"/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/Medical_Images_Public/Brain-Mets-Lung-MRI-Path-Segs_CLEAN"
OUTPUT_ROOT = r"/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_main/image_preprocessing/align/output"
EXCEL_ROOT = None

# D. Patient Selection (Subset Control)
# Set to None to process ALL discovered patients (from Excel or directory scan).
# Set to a slice range (e.g., "0:5") for quick testing of the first 5 patients.
PATIENT_SUBSET = None

# E. File Naming Conventions
# Suffixes used to identify files inside each patient folder
T1CE_SUFFIX = "_t1ce_img.nii.gz"
FLAIR_SUFFIX = "_flair_img.nii.gz"
CORE_SEG_SUFFIX = "_core_seg.nii.gz"
WHOLE_SEG_SUFFIX = "_whole_seg.nii.gz"

# ==========================================
# 3. Helper Class: Redirect Print to PDF
# ==========================================
class PDFLogger:
    def __init__(self, pdf_path):
        self.terminal = sys.stdout
        self.pdf_path = pdf_path
        self.log_content = []
        
    def write(self, message):
        self.terminal.write(message)
        if message:
            self.log_content.append(message)
            
    def flush(self):
        self.terminal.flush()
        
    def update_pdf(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Courier", size=10) 
        
        full_text = "".join(self.log_content)
        full_text = full_text.encode('latin-1', 'replace').decode('latin-1') 
        
        pdf.multi_cell(0, 5, text=full_text)
        
        try:
            pdf.output(self.pdf_path)
        except PermissionError:
            self.terminal.write(f"\n[Warning] Could not save PDF '{self.pdf_path}'. Is it open in another program?\n")
        except Exception as e:
            self.terminal.write(f"\n[Warning] Could not save PDF '{self.pdf_path}': {e}\n")


class TextLogger:
    """
    Fallback logging class in case fpdf/fpdf2 is not available.
    Saves print output directly to a plain text (.txt) file.
    """
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_path = log_path.replace(".pdf", ".txt")
        self.log_content = []
        
    def write(self, message):
        self.terminal.write(message)
        if message:
            self.log_content.append(message)
            
    def flush(self):
        self.terminal.flush()
        
    def update_pdf(self):
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("".join(self.log_content))
        except Exception as e:
            self.terminal.write(f"\n[Warning] Could not save log to '{self.log_path}': {e}\n")


# ==========================================
# 4. Pipeline Functions
# ==========================================

def run_operation_on_patient(patient_entry, resample_only=True):
    """
    Co-registers FLAIR and segmentation masks onto the fixed T1CE grid,
    resamples them using correct interpolation, and stacks the output modalities.
    """
    pid = patient_entry['id']
    t1ce_path = patient_entry['t1ce']
    flair_path = patient_entry['flair']
    core_seg_path = patient_entry['core_seg']
    whole_seg_path = patient_entry['whole_seg']
    out_dir = patient_entry['output']
    
    # 1. Prepare output file names
    base_prefix = os.path.basename(t1ce_path).replace(T1CE_SUFFIX, "")
    
    out_t1ce_path = os.path.join(out_dir, f"{base_prefix}_t1ce_aligned.nii.gz")
    out_flair_path = os.path.join(out_dir, f"{base_prefix}_flair_aligned.nii.gz")
    out_core_seg_path = os.path.join(out_dir, f"{base_prefix}_core_seg_aligned.nii.gz")
    out_whole_seg_path = os.path.join(out_dir, f"{base_prefix}_whole_seg_aligned.nii.gz")
    out_stacked_modalities = os.path.join(out_dir, f"{base_prefix}_stacked_modalities.nii.gz")
    out_multilabel_seg = os.path.join(out_dir, f"{base_prefix}_multilabel_seg.nii.gz")
    
    # 2. Read images and segmentations
    print(f"   [I/O] Reading images and masks for patient {pid}...")
    t1ce = sitk.ReadImage(t1ce_path)
    flair = sitk.ReadImage(flair_path)
    core_seg = sitk.ReadImage(core_seg_path)
    whole_seg = sitk.ReadImage(whole_seg_path)
    
    print(f"   [Grid] Fixed grid (T1CE) size: {t1ce.GetSize()}, spacing: {t1ce.GetSpacing()}")
    print(f"   [Grid] Moving grid (FLAIR) size: {flair.GetSize()}, spacing: {flair.GetSpacing()}")
    
    # 3. Registration
    if resample_only:
        print("   [Coreg] Resampling mode: Header-based coordinate resampling (Identity Transform)...")
        transform = sitk.Transform()  # Identity
    else:
        print("   [Coreg] Registration mode: Mutual-Information rigid co-registration...")
        # Cast to Float32 for registration
        fixed_image = sitk.Cast(t1ce, sitk.sitkFloat32)
        moving_image = sitk.Cast(flair, sitk.sitkFloat32)
        
        # Center transform initializer
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        registration = sitk.ImageRegistrationMethod()
        
        # Multi-modal metric: Mattes Mutual Information
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingPercentage(0.15, seed=42)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Optimizer tuning (Regular Step Gradient Descent)
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, 
            minStep=1e-4, 
            numberOfIterations=100,
            gradientMagnitudeTolerance=1e-8
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        
        registration.SetInitialTransform(initial_transform, inPlace=False)
        
        transform = registration.Execute(fixed_image, moving_image)
        print(f"   [Coreg] Registration finished. Euler parameters: {transform.GetParameters()}")
        
    # 4. Resample FLAIR to T1CE grid (Linear Interpolation)
    print("   [Resample] Resampling FLAIR image with Linear interpolation...")
    resample_flair = sitk.ResampleImageFilter()
    resample_flair.SetReferenceImage(t1ce)
    resample_flair.SetTransform(transform)
    resample_flair.SetInterpolator(sitk.sitkLinear)
    flair_aligned = resample_flair.Execute(flair)
    
    # 5. Resample masks to T1CE grid (Nearest Neighbor Interpolation to keep labels binary!)
    print("   [Resample] Resampling whole tumor segmentation with Nearest Neighbor...")
    resample_whole_seg = sitk.ResampleImageFilter()
    resample_whole_seg.SetReferenceImage(t1ce)
    resample_whole_seg.SetTransform(transform)
    resample_whole_seg.SetInterpolator(sitk.sitkNearestNeighbor)
    whole_seg_aligned = resample_whole_seg.Execute(whole_seg)
    
    # Resample core tumor segmentation. Since it is already on the T1CE grid, we resample
    # with an identity transform to ensure absolute grid matching in case of metadata roundings.
    print("   [Resample] Standardizing core tumor segmentation onto fixed grid...")
    resample_core_seg = sitk.ResampleImageFilter()
    resample_core_seg.SetReferenceImage(t1ce)
    resample_core_seg.SetTransform(sitk.Transform())
    resample_core_seg.SetInterpolator(sitk.sitkNearestNeighbor)
    core_seg_aligned = resample_core_seg.Execute(core_seg)
    
    # 6. Modality Stacking (Compose T1CE & FLAIR into a single multi-channel volume)
    print("   [Stack] Stacking aligned T1CE and FLAIR structural channels...")
    stacked_modalities = sitk.Compose(t1ce, flair_aligned)
    
    # 7. Segmentation Label Stacking and Merging
    print("   [Stack] Creating merged categorical label map...")
    
    
    # Generate non-overlapping multi-label segmentation map:
    # 0 = Background
    # 1 = Core Tumor (enhancing tumor core + necrotic core)
    # 2 = Peritumoral Edema / Non-core tumor (Whole Tumor - Core Tumor)
    core_binary = core_seg_aligned > 0
    whole_binary = whole_seg_aligned > 0
    
    edema_binary = sitk.And(whole_binary, sitk.Not(core_binary))
    
    multilabel_seg = sitk.Cast(core_binary, sitk.sitkUInt8) * 1 + sitk.Cast(edema_binary, sitk.sitkUInt8) * 2
    
    # 8. Write outputs
    print("   [Write] Saving results...")
    sitk.WriteImage(t1ce, out_t1ce_path)
    sitk.WriteImage(flair_aligned, out_flair_path)
    sitk.WriteImage(core_seg_aligned, out_core_seg_path)
    sitk.WriteImage(whole_seg_aligned, out_whole_seg_path)
    sitk.WriteImage(stacked_modalities, out_stacked_modalities)
    sitk.WriteImage(multilabel_seg, out_multilabel_seg)
    
    print("   [Done] Successfully registered, resampled, and stacked patient volumes!")
    
    # Manual garbage collection to free memory between patients
    gc.collect()


# ==========================================
# 5. Define logging loop function
# ==========================================
def logging_loop(operation, operation_func, paths, params, patient_files_to_process, operation_kwargs=None):
    
    if operation_kwargs is None:
        operation_kwargs = {}
        
    if FPDF_AVAILABLE:
        pdf_logger = PDFLogger(paths["log_pdf"])
    else:
        print("[Warning] fpdf/fpdf2 not available in this environment. Falling back to plain text log (.txt).")
        pdf_logger = TextLogger(paths["log_pdf"])
    sys.stdout = pdf_logger
    
    attempted_patient_files = []
    errors_list = []
    start_time = time.time()
    
    try:
        print("==================================================")
        print(f"OPERATION: {operation}")
        print("==================================================\n")
        
        print("--- ENVIRONMENT INFO ---")
        print(f"Python version: {platform.python_version()}")
        print(f"OS/Platform: {platform.platform()}")
        print("\nLibraries/Packages versions:")
        
        for distribution in importlib.metadata.distributions():
            if distribution.metadata['Name'] in ['SimpleITK', 'pandas', 'openpyxl', 'fpdf2', 'pypdf']:
                print(f" - {distribution.metadata['Name']}: {distribution.version}")
                
        print("\n--- PATHS ---")
        for k, v in paths.items():
            print(f"{k}: {v} \n")
            
        print("\n--- PARAMETERS ---")
        for k, v in params.items():
            print(f"{k}: {v}")
            
        print("\n--- STARTING PATIENT LIST ---")
        print(f"Total starting patient files: {len(patient_files_to_process)}")
        for i, p in enumerate(patient_files_to_process):
            print(f"  {i}. Patient ID {p['id']}, T1CE: {os.path.basename(p['t1ce'])}")
        print("\n--------------------------------------------------")
        
        pdf_logger.update_pdf()
        
        i = 0
        for patient_dict in patient_files_to_process:
            patient_label = patient_dict['id']
            print(f"\n>> Patient: {patient_label} ...")
            print(f"\n>> T1CE (Fixed):  {os.path.basename(patient_dict['t1ce'])}")
            print(f">> FLAIR (Moving): {os.path.basename(patient_dict['flair'])}")
            print(f">> Output Folder: {patient_dict['output']}")
            print("\n--------------------------------------------------")
            attempted_patient_files.append(f"  {i}. Patient ID {patient_dict['id']} (Aligned & Stacked)")
            i = i + 1
            
            patient_start = time.time()
            
            try:
                print(f"{operation} PRINT-OUT START, For Run {i} of {len(patient_files_to_process)}")
                operation_func(patient_dict, **operation_kwargs)
                print(f"{operation} PRINT-OUT END, For Run {i} of {len(patient_files_to_process)}")
                print("\n--------------------------------------------------")
                
                elapsed = time.time() - patient_start
                print(f"   [Done] Successfully processed patient {patient_label} ({elapsed:.1f}s).")
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"   [ERROR] encountered for patient {patient_label}: {error_msg}")
                traceback.print_exc(file=sys.stdout)
                
                errors_list.append({"Patient": patient_label, "Error": error_msg})
                
                try:
                    df_errors = pd.DataFrame(errors_list)
                    df_errors.to_excel(paths["error_excel"], index=False)
                except PermissionError:
                    print("   [Warning] Could not save Excel file. Permission denied.")
            
            finally:
                pdf_logger.update_pdf()
                
    except Exception as e:
        print(f"\nCRITICAL LOOP ERROR: {e}")
        traceback.print_exc(file=sys.stdout)
        
    finally:
        total_time = time.time() - start_time
        print("\n==================================================")
        print("--- ENDING SUMMARY ---")
        print(f"Total patient files attempted: {len(attempted_patient_files)}")
        print(f"Total errors: {len(errors_list)}")
        print(f"Total execution time: {total_time:.1f}s")
        for patient_file in attempted_patient_files:
            print(patient_file)
        print("==================================================")
        
        try:
            pdf_logger.update_pdf()
            
            if errors_list:
                df_errors = pd.DataFrame(errors_list)
                try:
                    df_errors.to_excel(paths["error_excel"], index=False)
                except Exception as e:
                    print(f"   [Warning] Could not save final Excel file: {e}")
        finally:
            sys.stdout = pdf_logger.terminal


# ==========================================
# 6. Main Execution Block
# ==========================================

def main():
    safe_op_name = OPERATION_NAME.replace(" ", "_")
    
    # 1. Prepare log file paths in output directory
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    log_pdf_path = os.path.join(OUTPUT_ROOT, f"MainRun_{safe_op_name}_pipeline_log.pdf")
    error_excel_path = os.path.join(OUTPUT_ROOT, f"MainRun_{safe_op_name}_pipeline_errors.xlsx")
    
    # 2. Discover Patient IDs from Directory Scan
    patient_ids = []
    if os.path.exists(INPUT_ROOT):
        all_entries = os.listdir(INPUT_ROOT)
        patient_ids = sorted([
            d for d in all_entries 
            if os.path.isdir(os.path.join(INPUT_ROOT, d)) and not d.startswith('.') and d.startswith('YG_')
        ])
        print(f"Loaded {len(patient_ids)} candidate folders from directory scan: {INPUT_ROOT}")
    else:
        print(f"FATAL: Input root folder not found: {INPUT_ROOT}")
        sys.exit(1)
    
    # Apply subset if defined in Section 2
    if PATIENT_SUBSET is not None:
        try:
            start, end = map(int, PATIENT_SUBSET.split(':'))
            patient_ids = patient_ids[start:end]
            print(f"NOTE: Processing SUBSET range {PATIENT_SUBSET} ({len(patient_ids)} patients).")
        except Exception as e:
            print(f"WARNING: PATIENT_SUBSET format '{PATIENT_SUBSET}' invalid. Expected 'start:end'. Processing all.")
    
    # 3. Discover and Map all four files per patient
    patient_files = []
    for pid in patient_ids:
        p_dir = os.path.join(INPUT_ROOT, pid)
        files = os.listdir(p_dir)
        
        t1ce_files = [f for f in files if f.endswith(T1CE_SUFFIX)]
        flair_files = [f for f in files if f.endswith(FLAIR_SUFFIX)]
        core_seg_files = [f for f in files if f.endswith(CORE_SEG_SUFFIX)]
        whole_seg_files = [f for f in files if f.endswith(WHOLE_SEG_SUFFIX)]
        
        if t1ce_files and flair_files and core_seg_files and whole_seg_files:
            out_dir = os.path.join(OUTPUT_ROOT, pid)
            os.makedirs(out_dir, exist_ok=True)
            
            patient_files.append({
                'id': pid,
                'input': os.path.join(p_dir, t1ce_files[0]),  # Standard logging key
                'output': out_dir,                           # Standard logging key
                't1ce': os.path.join(p_dir, t1ce_files[0]),
                'flair': os.path.join(p_dir, flair_files[0]),
                'core_seg': os.path.join(p_dir, core_seg_files[0]),
                'whole_seg': os.path.join(p_dir, whole_seg_files[0])
            })
        else:
            missing = []
            if not t1ce_files: missing.append("T1CE")
            if not flair_files: missing.append("FLAIR")
            if not core_seg_files: missing.append("Core Seg")
            if not whole_seg_files: missing.append("Whole Seg")
            print(f"   [Warning] Patient {pid} is missing file(s): {', '.join(missing)}")

    paths = {
        "input_root": INPUT_ROOT,
        "output_root": OUTPUT_ROOT,
        "excel_root": EXCEL_ROOT if EXCEL_ROOT else "N/A (directory scan used)",
        "log_pdf": log_pdf_path,
        "error_excel": error_excel_path
    }

    func_inputs = {
        "resample_only": PARAMETERS["resample_only"]
    }

    logging_loop(
        operation=OPERATION_NAME, 
        operation_func=run_operation_on_patient, 
        paths=paths, 
        params=PARAMETERS, 
        patient_files_to_process=patient_files,
        operation_kwargs=func_inputs
    )

if __name__ == "__main__":
    main()
