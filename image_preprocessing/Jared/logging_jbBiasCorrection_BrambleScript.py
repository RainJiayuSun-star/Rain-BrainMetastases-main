# ==========================================================================
# ENVIRONMENT SETUP PREAMBLE 
# ==========================================================================
# This script uses logging_jb.py's pipeline framework to run N4 Bias Field
# Correction on previously skull-stripped MRI data, with full PDF logging 
# and Excel error tracking.
#
# PREREQUISITE: This script assumes you have already set up the conda/venv
# environment described in logging_jbHDBET_BrambleScript.py. SimpleITK is already
# installed as a transitive dependency of HD-BET (via nnunetv2), so no
# additional packages are required.
#
# ---- Pipeline Order ----
#   1. HD-BET Skull Stripping   (logging_jbHDBET_BrambleScript.py)  [COMPLETED]
#   2. N4 Bias Field Correction (this script)              <-- YOU ARE HERE
#   3. Intensity Normalization  (logging_jbIN_BrambleScript.py)
#
# ---- What This Script Does ----
# N4 Bias Field Correction (Tustison et al., 2010) removes the low-frequency 
# intensity inhomogeneity ("bias field") caused by RF coil non-uniformities 
# in MRI acquisitions. This is a critical preprocessing step before intensity
# normalization or radiomics feature extraction, as uncorrected bias fields
# introduce artificial intensity gradients across the brain volume.
#
# The algorithm operates on a downsampled version of the image (controlled by
# "shrink_factor") for speed, then projects the estimated bias field back to
# the original resolution. The implementation here uses SimpleITK's wrapper
# around the ANTs N4 algorithm.
#
# ---- Warnings & Known Pitfalls ----
#   - **Overwrite Behavior**: This script is intentionally designed to 
#     overwrite previous results in the output folder. This ensures a 
#     consistent final product by replacing any potentially faulty or 
#     incomplete previous runs with the latest results.
#   - **Input Dependency**: This script expects skull-stripped images and
#     their corresponding brain masks from HD-BET. Ensure the HD-BET 
#     pipeline has completed successfully before running this script.
#   - **Shrink Factor Tradeoff**: Higher shrink_factor values speed up
#     bias estimation but reduce accuracy. A value of 2 is a good default;
#     set to 1 to disable downsampling (slower but most accurate).
#   - Windows: Execution logic MUST be wrapped in 'if __name__ == "__main__":'
#     to prevent recursive child-process crashes.
#
# ---- Notes ----
#   - The Excel error log is only created if at least one error is recorded
#     during the loop. A successful run with 0 errors will not produce an
#     Excel file.
#   - This script does NOT perform intensity normalization. Use logging_jbIN_BrambleScript.py 
#     on the output of this script for that step.
# ==========================================================================

# ==========================================================================
# USAGE CHECKLIST
# ==========================================================================
# [ ] Section 2: Update INPUT_ROOT, OUTPUT_ROOT, and EXCEL_ROOT paths
# [ ] Section 2: Review PARAMETERS (shrink_factor, num_iters, etc.)
# [ ] Section 2: Verify file suffix conventions match your HD-BET output
# [ ] Section 2: Set PATIENT_SUBSET (None for all, or "0:5" for testing)
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
# The import name remains 'fpdf' — do NOT have both fpdf and fpdf2 installed simultaneously.
from fpdf import FPDF       # Used to dynamically generate and format the PDF log file.

import os                   # Used for standard operating system interfaces like checking file paths and creating directories.
import time                 # Used for tracking computation time.
import gc                   # Used for manual garbage collection after each prediction.
import SimpleITK as sitk    # Used for N4 Bias Field Correction (wraps the ANTs N4 algorithm).

# ==========================================
# 2. ATTENTION: User-Defined Inputs
# ==========================================

# A. Operation Identity
OPERATION_NAME = "N4 Bias Field Correction"

# B. N4 Algorithm Parameters
# These control the behavior of SimpleITK's N4BiasFieldCorrectionImageFilter.
PARAMETERS = {
    "shrink_factor": 2,          # Downsampling factor for bias estimation speed.
                                 #   1: No downsampling (slowest, most accurate)
                                 #   2: 2x downsampling (good default balance)
                                 #   4: 4x downsampling (fastest, least accurate)
    "num_fitting_levels": 4,     # Number of multi-resolution fitting levels.
    "num_iters": 50,             # Max iterations per fitting level.
    "convergence_thresh": 1e-6,  # Convergence threshold for early stopping.
}

# C. Root Path Configuration
# [ACTION REQUIRED] Update these paths for your PC:
# INPUT_ROOT:  The HD-BET output folder (containing patient subfolders with _SS and _SS_bet files)
# OUTPUT_ROOT: Where bias-corrected results should be saved (mirrored structure)
# EXCEL_ROOT:  The master Excel file containing patient MRNs (set to None to use directory scan)
# INPUT_ROOT = r"<PATH_TO_HDBET_OUTPUT_FOLDER>"
# OUTPUT_ROOT = r"<PATH_TO_BC_OUTPUT_FOLDER>"
# EXCEL_ROOT = r"<PATH_TO_PATIENT_EXCEL_FILE>"
INPUT_ROOT = r"C:\Users\JAB031\Desktop\2-Bramble-NSCLC-BM-RNASeq_SS_Local"
OUTPUT_ROOT = r"C:\Users\JAB031\Desktop\2-Bramble-NSCLC-BM-RNASeq_SS_BC_Local"
EXCEL_ROOT = None

# D. Patient Selection (Subset Control)
# Set to None to process ALL discovered patients (from Excel or directory scan).
# Set to a slice range (e.g., "0:5") for quick testing of the first 5 patients.
PATIENT_SUBSET = None

# E. File Naming Conventions
# These suffixes control how the script identifies input files and names output files.
# Update these if your HD-BET output uses different naming patterns.
INPUT_IMAGE_SUFFIX = "_SS.nii.gz"        # Skull-stripped image suffix from HD-BET
INPUT_MASK_SUFFIX = "_SS_bet.nii.gz"     # Brain mask suffix from HD-BET
OUTPUT_SUFFIX = "_SS_BC.nii.gz"          # Bias-corrected output suffix

# ==========================================
# 3. Helper Class: Redirect Print to PDF
# ==========================================
class PDFLogger:
    """
    Custom logging class designed to hijack Python's standard print output.
    It simultaneously pushes the output to your Spyder console and saves it to a running memory buffer. 
    At the end of each patient loop, this buffer string is repeatedly rendered into your dynamic PDF log file.
    """
    def __init__(self, pdf_path):
        self.terminal = sys.stdout
        self.pdf_path = pdf_path
        self.log_content = []
        
    def write(self, message):
        # 1. Write to the normal console in Spyder
        self.terminal.write(message)
        # 2. Store the string in memory
        if message:
            self.log_content.append(message)
            
    def flush(self):
        self.terminal.flush()
        
    def update_pdf(self):
        """Generates the PDF from the accumulated print statements."""
        pdf = FPDF()
        pdf.add_page()
        # Courier font is good for code/log output as it is fixed-width
        pdf.set_font("Courier", size=10) 
        
        # Join all captured text
        full_text = "".join(self.log_content)
        # Safety fallback: built-in PDF fonts (Courier, Helvetica) only support latin-1.
        # fpdf2 can handle full UTF-8 if you load a .ttf font via pdf.add_font(), but
        # for log output the built-in Courier is fine with this encode/decode guard.
        full_text = full_text.encode('latin-1', 'replace').decode('latin-1') 
        
        # Write text to PDF
        pdf.multi_cell(0, 5, text=full_text)
        
        try:
            # Overwrite the file on each iteration
            pdf.output(self.pdf_path)
        except PermissionError:
            # Common issue on Windows if you have the PDF open in a viewer like Adobe Acrobat
            self.terminal.write(f"\n[Warning] Could not save PDF '{self.pdf_path}'. Is it open in another program?\n")
        except Exception as e:
            self.terminal.write(f"\n[Warning] Could not save PDF '{self.pdf_path}': {e}\n")


# ==========================================
# 4. Pipeline Functions
# ==========================================

def run_bc_on_patient(patient_entry, shrink_factor, num_fitting_levels, num_iters, convergence_thresh):
    """
    Performs N4 Bias Field Correction on a single skull-stripped image.
    
    Parameters
    ----------
    patient_entry : dict
        A dictionary with 'input_img', 'input_mask', and 'output_bc' keys.
    shrink_factor : int
        Downsampling factor for the bias estimation step.
    num_fitting_levels : int
        Number of multi-resolution fitting levels.
    num_iters : int
        Maximum iterations per fitting level.
    convergence_thresh : float
        Convergence threshold for early stopping.
    """
    img_path = patient_entry['input_img']
    mask_path = patient_entry['input_mask']
    output_bc = patient_entry['output_bc']
    
    print(f"   Target Image: {os.path.basename(img_path)}")
    print(f"   Using Mask:   {os.path.basename(mask_path)}")
    
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    
    # --- N4 Bias Correction ---
    print("   Computing bias field...")
    if shrink_factor > 1:
        img_shrunk = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
        mask_shrunk = sitk.Shrink(mask, [shrink_factor] * img.GetDimension())
    else:
        img_shrunk, mask_shrunk = img, mask
        
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iters] * num_fitting_levels)
    corrector.SetConvergenceThreshold(convergence_thresh)
    corrector.Execute(img_shrunk, mask_shrunk)
    
    log_bias_field = corrector.GetLogBiasFieldAsImage(img)
    corrected_img = sitk.Exp(log_bias_field) * img

    
    # Ensure the output directory exists
    output_directory = os.path.dirname(output_bc)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    
    sitk.WriteImage(corrected_img, output_bc)
    print(f"   [Saved] Bias Corrected: {os.path.basename(output_bc)}")
    
    del img, mask, corrected_img, log_bias_field
    gc.collect()


# ==========================================
# 5. Define logging loop function
# ==========================================
def logging_loop(operation, operation_func, paths, params, patient_files_to_process, operation_kwargs=None):
    
    if operation_kwargs is None:
        operation_kwargs = {}
        
    # Initialize the PDF logger and redirect python's print (stdout) to it
    pdf_logger = PDFLogger(paths["log_pdf"])
    sys.stdout = pdf_logger
    
    # We will track patients that actually get touched and any errors
    attempted_patient_files = []
    errors_list = []
    
    # Record the overall start time
    start_time = time.time()
    
    try:
        # Print operation being performed
        print("==================================================")
        print(f"OPERATION: {operation}")
        print("==================================================\n")
        
        # Print versions of python, environment, and packages
        print("--- ENVIRONMENT INFO ---")
        print(f"Python version: {platform.python_version()}")
        print(f"OS/Platform: {platform.platform()}")
        print("\nLibraries/Packages versions:")
        
        # Dynamically check relevant packages
        for distribution in importlib.metadata.distributions():
            print(f" - {distribution.metadata['Name']}: {distribution.version}")
                
        # Print dictionary of filenames, paths, etc
        print("\n--- PATHS ---")
        for k, v in paths.items():
            print(f"{k}: {v} \n")
            
        # Print dictionary of parameters
        print("\n--- PARAMETERS ---")
        for k, v in params.items():
            print(f"{k}: {v}")
            
        # Print starting patient list to loop over
        print("\n--- STARTING PATIENT LIST ---")
        print(f"Total starting patient files: {len(patient_files_to_process)}")
        print("NOTE: Patient ID will be replicated for each file to be processed for that patient.")
        print("Index corresponds to location of particular patient file in list of all patient files to be processed.")
        for i, p in enumerate(patient_files_to_process):
            print(f"  {i}. Patient ID {p['id']}, Input File {p['input_img']}")
        print("\n--------------------------------------------------")
        
        # Initial PDF generation before we start processing
        pdf_logger.update_pdf()
        
        # Loop through each patient
        i = 0
        for patient_dict in patient_files_to_process:
            patient_label = patient_dict['id']
            print(f"\n>> Patient: {patient_label} ...")
            print(f"\n>> Input Filepath: {patient_dict['input_img']}")
            print(f"\n>> Output Filepath: {patient_dict['output_bc']}")
            print("\n--------------------------------------------------")
            attempted_patient_files.append(f"  {i}. Patient ID {patient_dict['id']}, Input File {patient_dict['input_img']}")
            i = i+1
            
            patient_start = time.time()
            
            try:
                # --- EXECUTE THE OPERATION FUNCTION ---
                print(f"{operation} PRINT-OUT START, For Run {i} of {len(patient_files_to_process)}")
                operation_func(patient_dict, **operation_kwargs)
                print(f"{operation} PRINT-OUT END, For Run {i} of {len(patient_files_to_process)}")
                print("\n--------------------------------------------------")
                
                elapsed = time.time() - patient_start
                print(f"   [Done] Successfully processed input file for patient {patient_label} ({elapsed:.1f}s).")
                # --------------------------------------
                
            except Exception as e:
                # Print any errors encountered
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"   [ERROR] encountered for patient {patient_label}, input file {patient_dict['input_img']}: {error_msg}")
                traceback.print_exc(file=sys.stdout) # Print full error traceback to the PDF/console
                
                # Log error to excel dataframe format (patient and error)
                errors_list.append({"Patient": patient_label, "Error": error_msg})
                
                # Save the Excel file immediately so it's up to date on each iteration
                try:
                    df_errors = pd.DataFrame(errors_list)
                    df_errors.to_excel(paths["error_excel"], index=False)
                except PermissionError:
                    print("   [Warning] Could not save Excel file. Permission denied.")
            
            finally:
                # Update/Save the PDF with each iteration of the loop
                pdf_logger.update_pdf()
                
    except Exception as e:
        # Catch unexpected errors that break the entire loop
        print(f"\nCRITICAL LOOP ERROR: {e}")
        traceback.print_exc(file=sys.stdout)
        
    finally:
        # Print ending summary
        total_time = time.time() - start_time
        print("\n==================================================")
        print("--- ENDING SUMMARY ---")
        print(f"Total patient files attempted: {len(attempted_patient_files)}")
        print(f"Total errors: {len(errors_list)}")
        print(f"Total execution time: {total_time:.1f}s")
        for patient_file in attempted_patient_files:
            print(patient_file)
        print("==================================================")
        
        # Important: Restore standard print back to Spyder's internal console 
        # so you don't mess up the console for other runs. We do this inside a nested 
        # try/finally to guarantee your Spyder console doesn't break if file-saving fails.
        try:
            # Save a final time
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
    # ================================================
    # A. Setup Discovery Logic
    # ================================================
    
    # 1. Prepare log file paths
    safe_op_name = OPERATION_NAME.replace(" ", "_")
    log_pdf_path = os.path.join(OUTPUT_ROOT, f"MainRun_{safe_op_name}_pipeline_log.pdf")
    error_excel_path = os.path.join(OUTPUT_ROOT, f"MainRun_{safe_op_name}_pipeline_errors.xlsx")
    
    # 2. Identify Patient IDs
    patient_ids = []
    
    # Try Excel first
    if EXCEL_ROOT and os.path.exists(EXCEL_ROOT):
        try:
            data_frame = pd.read_excel(EXCEL_ROOT)
            patient_ids = data_frame["UW_MRN"].astype(str).tolist()
            # Removes duplicates while preserving order
            patient_ids = list(dict.fromkeys(patient_ids)) 
            print(f"Loaded {len(patient_ids)} patients from Excel: {EXCEL_ROOT}")
        except Exception as e:
            print(f"WARNING: Could not read Excel file at {EXCEL_ROOT}: {e}. Falling back to directory scan.")
    
    # Fallback to directory scan if no Excel provided or loading failed
    if not patient_ids:
        if os.path.exists(INPUT_ROOT):
            all_entries = os.listdir(INPUT_ROOT)
            # Pick any folder that isn't hidden (doesn't start with '.')
            patient_ids = sorted([
                d for d in all_entries 
                if os.path.isdir(os.path.join(INPUT_ROOT, d)) and not d.startswith('.')
            ])
            print(f"Loaded {len(patient_ids)} candidate folders from directory scan: {INPUT_ROOT}")
        else:
            print(f"FATAL: No Excel provided and Input root folder not found: {INPUT_ROOT}")
            sys.exit(1)
    
    # Apply subset if defined in Section 2
    if PATIENT_SUBSET is not None:
        try:
            start, end = map(int, PATIENT_SUBSET.split(':'))
            patient_ids = patient_ids[start:end]
            print(f"NOTE: Processing SUBSET range {PATIENT_SUBSET} ({len(patient_ids)} patients).")
        except:
            print(f"WARNING: PATIENT_SUBSET format '{PATIENT_SUBSET}' invalid. Expected 'start:end' (e.g. '0:5'). Processing all.")
    
    # 3. Determine filepaths
    # Input structure: INPUT_ROOT / <pid> / <filename>_SS.nii.gz + <filename>_SS_bet.nii.gz
    patient_files = []
    for pid in patient_ids:
        p_dir = os.path.join(INPUT_ROOT, pid)
        if not os.path.exists(p_dir):
            print(f"   [Warning] Patient folder not found: {p_dir}")
            continue
        
        all_files = os.listdir(p_dir)
        # Find skull-stripped images (exclude masks)
        ss_images = [f for f in all_files if f.endswith(INPUT_IMAGE_SUFFIX) and not f.endswith(INPUT_MASK_SUFFIX)]
        
        for img_file in ss_images:
            # Derive the expected mask filename from the image filename
            mask_file = img_file.replace(INPUT_IMAGE_SUFFIX, INPUT_MASK_SUFFIX)
            if mask_file in all_files:
                out_dir = os.path.join(OUTPUT_ROOT, pid)
                os.makedirs(out_dir, exist_ok=True)
                patient_files.append({
                    'id': pid,
                    'input_img': os.path.join(p_dir, img_file),
                    'input_mask': os.path.join(p_dir, mask_file),
                    'output_bc': os.path.join(out_dir, img_file.replace(INPUT_IMAGE_SUFFIX, OUTPUT_SUFFIX)),
                })
            else:
                print(f"   [Warning] Mask not found for {img_file} in patient {pid} (expected {mask_file})")

    # ==========================================
    # B. Logging Setup
    # ==========================================

    # --- Optional: Log file incrementing ---
    # Uncomment the block below if you prefer to keep logs from previous runs
    # rather than overwriting them. Each new run will create _1, _2, etc.
    #
    # counter = 1
    # while os.path.exists(log_pdf_path) or os.path.exists(error_excel_path):
    #     log_pdf_path = os.path.join(OUTPUT_ROOT, f"{safe_op_name}_pipeline_log_{counter}.pdf")
    #     error_excel_path = os.path.join(OUTPUT_ROOT, f"{safe_op_name}_pipeline_errors_{counter}.xlsx")
    #     counter += 1

    paths = {
        "input_root": INPUT_ROOT,
        "output_root": OUTPUT_ROOT,
        "excel_root": EXCEL_ROOT if EXCEL_ROOT else "N/A (directory scan used)",
        "log_pdf": log_pdf_path,
        "error_excel": error_excel_path
    }

    # ==========================================
    # C. Run Pipeline
    # ==========================================
    
    # Build kwargs from PARAMETERS (excluding keys not needed by the function)
    func_inputs = {
        "shrink_factor": PARAMETERS["shrink_factor"],
        "num_fitting_levels": PARAMETERS["num_fitting_levels"],
        "num_iters": PARAMETERS["num_iters"],
        "convergence_thresh": PARAMETERS["convergence_thresh"],
    }

    logging_loop(
        operation=OPERATION_NAME, 
        operation_func=run_bc_on_patient, 
        paths=paths, 
        params=PARAMETERS, 
        patient_files_to_process=patient_files,
        operation_kwargs=func_inputs
    )

if __name__ == "__main__":
    main()
