# ==========================================================================
# ENVIRONMENT SETUP PREAMBLE 
# ==========================================================================
# logging_jb_BrambleScript.py — A reusable pipeline skeleton for medical image processing
# operations with full PDF logging and Excel error tracking.
#
# This script is the TEMPLATE from which operation-specific scripts are
# derived. It provides the complete infrastructure for:
#   - Capturing all console output into a PDF log file
#   - Tracking per-patient errors in an Excel file
#   - Logging environment info, parameters, and timing
#   - Robust patient discovery (Excel file or directory scan fallback)
#   - Subset selection for quick testing
#
# ---- How to Use This Template ----
# To create a new pipeline script for a specific operation:
#   1. Copy this file and rename it (e.g., logging_jbCOREG.py)
#   2. Add your operation-specific imports to Section 1
#   3. Fill in Section 2 with your operation name, parameters, and paths
#   4. Write your processing function in Section 4
#   5. Wire up the function call in Section 6C
#
# ---- Derived Scripts ----
# The following scripts were built from this template:
#   - logging_jbHDBET_BrambleScript.py  (HD-BET Skull Stripping)
#   - logging_jbBC_BrambleScript.py          (N4 Bias Field Correction)
#   - logging_jbIN_BrambleScript.py          (Intensity Normalization)
#
# ---- Dependencies (Core Template) ----
#   Python   >= 3.10
#   fpdf2    (PDF log generation)
#   pandas   (Error tracking Excel export)
#   openpyxl (Excel file backend for pandas)
#   
#   Add operation-specific dependencies as needed (e.g., SimpleITK,
#   torch, nibabel, HD-BET, etc.)
#
# ---- Warnings & Known Pitfalls ----
#   - **Overwrite Behavior**: By default, this script will overwrite 
#     previous log files in the output folder. To preserve old logs, 
#     uncomment the log file incrementing block in Section 6B.
#   - Windows: Execution logic MUST be wrapped in 'if __name__ == "__main__":'
#     to prevent recursive child-process crashes.
#   - Spyder/Kernels: Ensure spyder-kernels version matches your Spyder app.
#     If Spyder complains, install the specific version requested in the error.
#   - fpdf2 vs fpdf: fpdf2 is the actively maintained successor to the
#     abandoned 'fpdf' (PyFPDF) library. Both use the same import name.
#     Do NOT have both installed simultaneously.
#
# ---- Notes ----
#   - The Excel error log is only created if at least one error is recorded
#     during the loop. A successful run with 0 errors will not produce an
#     Excel file.
#   - The PDFLogger class uses latin-1 encoding for built-in Courier font
#     compatibility. To support full Unicode (e.g., emoji), load a .ttf
#     font via pdf.add_font().
# ==========================================================================

# ==========================================================================
# USAGE CHECKLIST
# ==========================================================================
# [ ] Section 1: Add operation-specific imports
# [ ] Section 2A: Set OPERATION_NAME
# [ ] Section 2B: Define PARAMETERS for your operation
# [ ] Section 2C: Update INPUT_ROOT, OUTPUT_ROOT, and EXCEL_ROOT paths
# [ ] Section 2D: Set PATIENT_SUBSET (None for all, or "0:5" for testing)
# [ ] Section 2E: Define file suffix conventions for your input/output files
# [ ] Section 4: Write your processing function (see integration guide)
# [ ] Section 6C: Wire up your function and its kwargs
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
import gc                   # Used for manual garbage collection after each iteration.

# --- Add your operation-specific imports below ---
# Examples:
#   import SimpleITK as sitk                  # For N4 bias correction, image I/O
#   import torch                              # For GPU-accelerated deep learning
#   import nibabel as nib                     # For NIfTI file handling
#   from HD_BET.hd_bet_prediction import ...  # For HD-BET skull stripping

# ==========================================
# 2. ATTENTION: User-Defined Inputs
# ==========================================

# A. Operation Identity
OPERATION_NAME = "<YOUR_OPERATION_NAME>"

# B. Operation Parameters
# Define all tunable parameters for your operation here.
# These are logged in the PDF and passed to your processing function.
PARAMETERS = {
    # Example parameters (replace with your own):
    # "shrink_factor": 2,
    # "num_iterations": 100,
    # "use_gpu": False,
}

# C. Root Path Configuration
# [ACTION REQUIRED] Update these paths for your PC:
# INPUT_ROOT:  The folder containing patient subfolders with input files
# OUTPUT_ROOT: Where processed results should be saved (mirrored structure)
# EXCEL_ROOT:  The master Excel file containing patient IDs (set to None to use directory scan)
INPUT_ROOT = r"<PATH_TO_INPUT_FOLDER>"
OUTPUT_ROOT = r"<PATH_TO_OUTPUT_FOLDER>"
EXCEL_ROOT = r"<PATH_TO_PATIENT_EXCEL_FILE>"

# D. Patient Selection (Subset Control)
# Set to None to process ALL discovered patients (from Excel or directory scan).
# Set to a slice range (e.g., "0:5") for quick testing of the first 5 patients.
PATIENT_SUBSET = None

# E. File Naming Conventions
# These suffixes control how the script identifies input files and names output files.
# Update these to match the naming patterns of your input data.
INPUT_SUFFIX = ".nii.gz"                     # Input file suffix to search for
OUTPUT_SUFFIX = "_processed.nii.gz"          # Output file suffix

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
#
# HOW TO INTEGRATE YOUR FUNCTION:
#
# The logging loop calls your function as:
#     operation_func(patient_entry, **operation_kwargs)
#
# Where:
#   - patient_entry: A dict with 'id', 'input', and 'output' keys
#     (or additional keys you define in Section 6A's file discovery).
#   - operation_kwargs: The keyword arguments you define in Section 6C.
#
# --- Pattern A: Simple function ---
#   def my_operation(patient_entry, param1, param2):
#       input_path = patient_entry['input']
#       output_path = patient_entry['output']
#       # ... do your processing ...
#       gc.collect()  # Free memory between patients
#
# --- Pattern B: Function requiring expensive one-time setup ---
#   Some tools (HD-BET, deep learning models) require initializing a
#   model or predictor that takes seconds to minutes. Initialize it ONCE
#   in Section 6C before calling logging_loop(), then pass the initialized
#   object via operation_kwargs.
#
#   def my_operation(patient_entry, model):
#       input_path = patient_entry['input']
#       output_path = patient_entry['output']
#       model.predict(input_path, output_path)
#       gc.collect()
#
#   # In Section 6C:
#   model = load_my_model()
#   func_inputs = {"model": model}
#

def run_operation_on_patient(patient_entry):
    """
    [PLACEHOLDER] Replace this function with your actual processing logic.
    
    Parameters
    ----------
    patient_entry : dict
        A dictionary with at minimum 'id' and 'input' keys.
        Add 'output', 'mask', or other keys as needed in Section 6A.
    
    Add additional parameters as keyword arguments (e.g., model, config)
    and pass them via operation_kwargs in Section 6C.
    """
    input_path = patient_entry['input']
    print(f"   Input: {os.path.basename(input_path)}")
    
    # --- YOUR PROCESSING CODE HERE ---
    #
    # Example:
    #   img = sitk.ReadImage(input_path, sitk.sitkFloat32)
    #   result = my_algorithm(img, **params)
    #   sitk.WriteImage(result, patient_entry['output'])
    #
    # ----------------------------------
    
    # Manual garbage collection to free memory between patients
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
            print(f"  {i}. Patient ID {p['id']}, Input File {p['input']}")
        print("\n--------------------------------------------------")
        
        # Initial PDF generation before we start processing
        pdf_logger.update_pdf()
        
        # Loop through each patient
        i = 0
        for patient_dict in patient_files_to_process:
            patient_label = patient_dict['id']
            print(f"\n>> Patient: {patient_label} ...")
            print(f"\n>> Input Filepath: {patient_dict['input']}")
            print(f"\n>> Output Filepath: {patient_dict.get('output', 'N/A')}")
            print("\n--------------------------------------------------")
            attempted_patient_files.append(f"  {i}. Patient ID {patient_dict['id']}, Input File {patient_dict['input']}")
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
                print(f"   [ERROR] encountered for patient {patient_label}, input file {patient_dict['input']}: {error_msg}")
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
    # A. Patient Discovery & File Mapping
    # ================================================
    
    # 1. Prepare log file paths
    safe_op_name = OPERATION_NAME.replace(" ", "_")
    log_pdf_path = os.path.join(OUTPUT_ROOT, f"MainRun_{safe_op_name}_pipeline_log.pdf")
    error_excel_path = os.path.join(OUTPUT_ROOT, f"MainRun_{safe_op_name}_pipeline_errors.xlsx")
    
    # 2. Identify Patient IDs
    patient_ids = []
    
    # Try Excel first
    if EXCEL_ROOT and EXCEL_ROOT != r"<PATH_TO_PATIENT_EXCEL_FILE>" and os.path.exists(EXCEL_ROOT):
        try:
            data_frame = pd.read_excel(EXCEL_ROOT)
            # [ACTION REQUIRED] Update the column name below to match your Excel file.
            patient_ids = data_frame["PATIENT_ID_COLUMN"].astype(str).tolist()
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
    # [ACTION REQUIRED] Adapt the file discovery logic below to match your 
    # input data structure. The examples below show common patterns.
    #
    # --- Pattern A: Flat structure (files directly in patient folder) ---
    #   INPUT_ROOT / <pid> / <filename>.nii.gz
    #
    # --- Pattern B: Subfolder structure (e.g., NIFTI subfolder) ---
    #   INPUT_ROOT / <pid> / NIFTI / <filename>.nii.gz
    #
    # --- Pattern C: Image + Mask pairs ---
    #   INPUT_ROOT / <pid> / <filename>_SS.nii.gz + <filename>_SS_bet.nii.gz
    
    patient_files = []
    for pid in patient_ids:
        p_dir = os.path.join(INPUT_ROOT, pid)
        if not os.path.exists(p_dir):
            print(f"   [Warning] Patient folder not found: {p_dir}")
            continue
        
        # Find input files matching the expected suffix
        input_files = [f for f in os.listdir(p_dir) if f.endswith(INPUT_SUFFIX)]
        
        if input_files:
            out_dir = os.path.join(OUTPUT_ROOT, pid)
            os.makedirs(out_dir, exist_ok=True)
            
            for file in input_files:
                patient_files.append({
                    'id': pid,
                    'input': os.path.join(p_dir, file),
                    'output': os.path.join(out_dir, file.replace(INPUT_SUFFIX, OUTPUT_SUFFIX)),
                })
        else:
            print(f"   [Warning] No {INPUT_SUFFIX} files found for patient {pid} in {p_dir}")

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
    # C. Initialization & Run
    # ==========================================
    
    # --- Pre-loop initialization (if needed) ---
    # If your operation function requires expensive one-time setup (e.g., loading
    # a deep learning model, connecting to a database), do it HERE before
    # calling logging_loop(). Then pass the initialized object through func_inputs.
    #
    # Example (HD-BET style):
    #   predictor = get_hdbet_predictor_custom(device=device, ...)
    #   func_inputs = {"predictor": predictor, "save_mask": True}
    #
    # Example (SimpleITK style — no pre-init needed, just pass config):
    #   func_inputs = {
    #       "shrink_factor": PARAMETERS["shrink_factor"],
    #       "num_iters": PARAMETERS["num_iters"],
    #   }
    
    func_inputs = {
        # Add your function's keyword arguments here.
        # These are passed to operation_func on every call via **kwargs.
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
