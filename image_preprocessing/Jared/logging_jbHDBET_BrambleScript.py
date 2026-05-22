# ==========================================================================
# ENVIRONMENT SETUP PREAMBLE 
# ==========================================================================
# This script uses logging_jb_BrambleScript.py's pipeline framework to run HD-BET v2.0.1
# (brain extraction / skull stripping) with full PDF logging and
# Excel error tracking. 
#
# CRITICAL: If you are on a restricted/managed network (e.g., University/NHS),
# DO NOT create your environment on a Network Drive (e.g., H: drive). 
# Always build your environment on a LOCAL drive (e.g., C:\Users\...) to 
# avoid PATH, "directory not watchable," and permission errors.
#
# ---- HD-BET v2.0.1 official dependencies (from pyproject.toml) ----
#   Python   >= 3.10
#   torch    >= 2.0.0
#   nnunetv2 >= 2.5.1
#   numpy, scikit-image, SimpleITK   (pulled in automatically)
#
# ---- Warnings & Known Pitfalls ----
#   - **Overwrite Behavior**: This script is intentionally designed to 
#     overwrite previous results in the output folder. This ensures a 
#     consistent final product by replacing any potentially faulty or 
#     incomplete previous runs with the latest results.
#   - Python 3.13+: Older NumPy 1.x versions cannot be built without a C++ 
#     compiler. If using 3.13, allow the installer to pull NumPy 2.x.
#     If your workflow strictly requires NumPy < 2, use Python 3.11 or 3.12.
#   - Model Weights: On the first run, HD-BET will attempt to download ~500MB
#     of weights to ~/hd-bet_params/. If your network blocks this, manually 
#     copy the folder from a machine with existing weights.
#   - PyTorch 2.9.0: Known regression with 3D convolutions + AMP. 
#     Use torch 2.8.x or earlier if you encounter stability issues.
#   - Windows: Execution logic MUST be wrapped in 'if __name__ == "__main__":'
#     to prevent recursive child-process crashes.
#   - Spyder/Kernels: Ensure spyder-kernels version matches your Spyder app.
#     If Spyder complains, install the specific version requested in the error.
#
# =========== INSTALLATION STEPS (ANACONDA / SPYDER) ===========
#
# --- Step 1: Create a fresh conda environment ---
#   conda create -n logging_jb python=3.11 -y
#   conda activate logging_jb
#
# --- Step 2: Install PyTorch via conda FIRST ---
#   CPU-only:
#     conda install -c pytorch pytorch torchvision torchaudio cpuonly -y
#   GPU (CUDA 11.8 example):
#     conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8 -y
#
# --- Step 3: Install HD-BET & Pipeline Tools ---
#   pip install "numpy<2"
#   pip install hd-bet fpdf2 pandas openpyxl nibabel spyder-kernels
#
# --- Step 4: Verification & Launch ---
#   pip check
#   spyder
#   (Note: If using a global Spyder install, point it to this env's 
#    python.exe in Tools > Preferences > Python Interpreter)
#
# ==============================================================
#
# =========== INSTALLATION STEPS (VS CODE / PIP) ===========
#
# --- Step 1: Create the Environment ---
#   Press Ctrl+Shift+P -> "Python: Create Environment" -> ".venv".
#   Select Python 3.11 or 3.12 (3.13 is supported but forces NumPy 2.x).
#   Ensure the folder is on a local drive (e.g., C:\Users\...).
#
# --- Step 2: Install PyTorch (CPU Version) ---
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#
# --- Step 3: Install HD-BET & Dependencies ---
#   pip install hd-bet fpdf2 pandas openpyxl nibabel
#   (This automatically handles the best NumPy version for your Python install)
#
# --- Step 4: Verification ---
#   pip check
#
# --- Step 5: Select Interpreter ---
#   Press Ctrl+Shift+P -> "Python: Select Interpreter"
#   Choose the one that starts with ('.venv': venv).
# ==============================================================
#
# --- FINAL Step for all users: Configure paths below (Section 2) ---
#   Update INPUT_ROOT, OUTPUT_ROOT, and EXCEL_ROOT to match the
#   file locations on the current PC.
#
# --- Notes ---
#   - This script runs HD-BET on CPU by default. To use GPU, change
#     the PARAMETERS["device"] value to "cuda" (requires CUDA toolkit
#     and a GPU-enabled PyTorch install).
#   - The predictor is initialized ONCE before the loop and reused for
#     all patients to avoid redundant model loading.
#   - Unicode emoji characters (checkmarks, X marks) used in the
#     reference program are avoided here because the built-in PDF fonts
#     (Courier) only support latin-1. fpdf2 can render full UTF-8 if you
#     load a .ttf font via add_font(), but the default guard replaces
#     unsupported characters with '?'.
#   - The Excel error log is only created if at least one error is recorded
#     during the loop. A successful run with 0 errors will not produce an
#     Excel file.
# ==========================================================================

# ==========================================================================
# USAGE CHECKLIST FOR PIPELINE EXPORT
# ==========================================================================
# [ ] Section 2: Update INPUT_ROOT, OUTPUT_ROOT, and EXCEL_ROOT paths
# [ ] Section 2: Review PARAMETERS (device, step_size, use_tta, save_mask)
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

# If you don't have fpdf2 installed, you will need to run: `pip install fpdf2` in your Spyder console.
# Note: fpdf2 is the actively maintained successor to the abandoned 'fpdf' (PyFPDF) library.
# The import name remains 'fpdf' — do NOT have both fpdf and fpdf2 installed simultaneously.
from fpdf import FPDF       # Used to dynamically generate and format the PDF log file.

import os                   # Used for standard operating system interfaces like checking file paths and creating directories.
import time                 # Used for tracking computation time.
import gc                   # Used for manual garbage collection after each prediction.
import numpy as np          # Used for numerical operations and dummy test volume generation.
import nibabel as nib       # Used for reading/writing NIfTI neuroimaging files.

import torch                                                            # PyTorch backend used by HD-BET for inference.
from HD_BET.hd_bet_prediction import hdbet_predict                      # Core HD-BET prediction function.
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor    # nnU-Net predictor that HD-BET wraps.
from HD_BET.paths import folder_with_parameter_files                    # Path to HD-BET model weights.

# ==========================================
# 2. ATTENTION: User-Defined Inputs
# ==========================================

# A. Operation Identity
OPERATION_NAME = "HD-BET Skull Stripping"

# B. HD-BET Parameters
# These control HD-BET's predictor initialization and behavior.
PARAMETERS = {
    "device": "cpu",        # "cpu" or "cuda" (GPU requires CUDA-enabled PyTorch)
    "step_size": 0.5,       # Tile step size in (0,1). Lower = more overlap = better quality but slower.
                            #   0.5 (default): ~1-2 min/patient on CPU
                            #   0.25:          minor time increase, still < 2 min
                            #   0.1:           significant increase, up to ~10 min
    "use_tta": False,       # Test-Time Augmentation. True = better quality but much slower.
    "verbose": False,       # Whether to print detailed nnU-Net progress info.
    "save_mask": True       # Saves the mask generated for the patient
}

# C. Root Path Configuration
# [ACTION REQUIRED] Update these paths for your PC:
# INPUT_ROOT: The path to 'Folder A' (containing patient folders)
# OUTPUT_ROOT: The path to 'Folder B' (where results should go)
# EXCEL_ROOT: The path to the master Excel file containing patient MRNs
INPUT_ROOT = r"<PATH_TO_INPUT_FOLDER>"
OUTPUT_ROOT = r"<PATH_TO_OUTPUT_FOLDER>"
EXCEL_ROOT = r"<PATH_TO_PATIENT_EXCEL_FILE>"


# D. Patient Selection (Subset Control)
# Set to 'None' to process ALL discovered patients (from Excel or directory scan).
# Set to a slice range (e.g., "0:5") for quick testing of the first 5 patients.
PATIENT_SUBSET = None 

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

def get_hdbet_predictor_custom(device=torch.device('cpu'), use_tta=False, verbose=False, step_size=0.5):
    """
    Initializes and returns an nnUNetPredictor loaded with HD-BET model weights.
    
    This is a custom version of HD-BET's get_hdbet_predictor that exposes the
    step_size parameter for controlling tile overlap during prediction.
    
    Parameters
    ----------
    device : torch.device
        Device to run inference on ('cpu' or 'cuda').
    use_tta : bool
        Enable Test-Time Augmentation (mirroring). Improves quality but slower.
    verbose : bool
        Print detailed nnU-Net preprocessing and inference info.
    step_size : float
        Tile step size in (0, 1). Lower values = more overlap = better quality but slower.
    
    Returns
    -------
    nnUNetPredictor
        Initialized predictor ready for hdbet_predict() calls.
    """
    # Handling CPU-specific settings:
    if device.type == 'cpu':
        os.environ['nnUNet_compile'] = 'F'  # Disable JIT compilation for compatibility
        perform_on_device = False
    else:
        perform_on_device = True
    
    # Initialization of predictor:
    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=perform_on_device,
        device=device,
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=True
    )
    
    # Load model weights:
    predictor.initialize_from_trained_model_folder(
        folder_with_parameter_files, 
        use_folds='all',            # 'all' = Accurate mode (ensemble of 5 models)
        checkpoint_name='checkpoint_final.pth'
    )
    
    return predictor


def run_hdbet_on_patient(patient_entry, predictor, save_mask=False):
    """
    Wrapper function that bridges HD-BET's hdbet_predict() with the
    logging_loop's expected call signature: operation_func(patient, **kwargs).
    
    Parameters
    ----------
    patient_entry : dict
        A dictionary with 'input' and 'output' keys specifying file paths.
    predictor : nnUNetPredictor
        The pre-initialized HD-BET predictor instance.
    """
    input_path = patient_entry['input']
    output_path = patient_entry['output']
    
    print(f"   Input:  {os.path.basename(input_path)}")
    print(f"   Output: {os.path.basename(output_path)}")
    
    # Ensure the output directory exists
    output_directory = os.path.dirname(output_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    
    # Core HD-BET prediction call
    hdbet_predict(input_path, output_path, predictor, keep_brain_mask=save_mask)
    
    # Manual garbage collection to free memory between patients
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
            print(f"\n>> Output Filepath: {patient_dict['output']}")
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
    # A. Setup Discovery Logic
    # ================================================
    
    # 1. Load files
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
    patient_files = []
    for pid in patient_ids:
        
        # Expected structure: Folder_A / <pid> / NIFTI / <file>.nii.gz
        nifti_dir = os.path.join(INPUT_ROOT, pid, "NIFTI")
        
        if os.path.exists(nifti_dir):
            # Look for the .nii.gz file inside the NIFTI folder
            files = [f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")]
            
            if files:
                
                # Use loop to define which nifti files to process if not all of them
                # input_files = []
                # for file in files:
                #     if <insert condition>:
                #         input_files.append(os.path.join(nifti_dir, file))
                
                # Otherwise, process all nifti files present
                input_files = [os.path.join(nifti_dir, file) for file in files]
                                
                # Construct mirrored output path: Folder_B / <pid> / <filename>_SS.nii.gz
                patient_output_dir = os.path.join(OUTPUT_ROOT, pid)
                os.makedirs(patient_output_dir, exist_ok=True) # Create mirrored subfolder
                
                # Construct output filenames
                output_files = [os.path.join(patient_output_dir, file.replace(".nii.gz","_SS.nii.gz")) for file in files]
                
                # Patient filepaths 
                # Pair inputs and outputs correctly
                patient_files.extend([
                    {'id': pid, 'input': input_file, 'output': output_file}
                    for input_file, output_file in zip(input_files, output_files)
                ])
                            
            else:
                print(f"   [Warning] No .nii.gz file found for patient {pid} in {nifti_dir}")
        else:
            print(f"   [Warning] NIFTI subfolder not found for patient {pid}")

    # ==========================================
    # B. Legacy Dummy Setup (Commented out)
    # ==========================================
    # Use this section instead of Section A if you want to run a quick internal test again.
    
    # input_dir = os.path.join(os.path.expanduser("~"), "Documents", "HDBET_LogTest", "input")
    # output_dir = os.path.join(os.path.expanduser("~"), "Documents", "HDBET_LogTest", "output")
    # os.makedirs(input_dir, exist_ok=True)
    # os.makedirs(output_dir, exist_ok=True)
    
    # dummy_patient_ids = ["DummyP01", "DummyP02", "DummyP03"]
    # for pid in dummy_patient_ids:
    #     dummy_path = os.path.join(input_dir, f"{pid}_T1.nii.gz")
    #     if not os.path.exists(dummy_path):
    #         dummy_data = np.random.rand(64, 64, 64).astype(np.float32) * 1000
    #         dummy_img = nib.Nifti1Image(dummy_data, affine=np.eye(4))
    #         nib.save(dummy_img, dummy_path)
    
    # patient_list = [
    #     {'input': os.path.join(input_dir, f"{pid}_T1.nii.gz"),
    #      'output': os.path.join(output_dir, f"{pid}_T1_SS.nii.gz")}
    #     for pid in dummy_patient_ids
    # ]
    # output_root = output_dir # For logging paths below
    

    # ==========================================
    # C. Logging Setup
    # ==========================================

    # Check for existing logs and increment counter if needed
    counter = 1
    while os.path.exists(log_pdf_path) or os.path.exists(error_excel_path):
        log_pdf_path = os.path.join(OUTPUT_ROOT, f"{safe_op_name}_pipeline_log_{counter}.pdf")
        error_excel_path = os.path.join(OUTPUT_ROOT, f"{safe_op_name}_pipeline_errors_{counter}.xlsx")
        counter += 1

    paths = {
        "input_root": INPUT_ROOT,
        "output_root": OUTPUT_ROOT,
        "excel_root": EXCEL_ROOT if EXCEL_ROOT else "N/A (directory scan used)",
        "log_pdf": log_pdf_path,
        "error_excel": error_excel_path
    }

    # ==========================================
    # D. Initialization & Loop
    # ==========================================
    device = torch.device(PARAMETERS["device"])
    print(f"Initializing HD-BET predictor on '{PARAMETERS['device']}'...")
    try:
        predictor = get_hdbet_predictor_custom(
            device=device,
            use_tta=PARAMETERS["use_tta"],
            verbose=PARAMETERS["verbose"],
            step_size=PARAMETERS["step_size"]
        )
        print("Predictor initialized successfully.")
    except Exception as e:
        print(f"FATAL: Could not initialize HD-BET predictor: {e}")
        sys.exit(1)

    func_inputs = {"predictor": predictor,
                   "save_mask": PARAMETERS["save_mask"]
    }

    logging_loop(
        operation=OPERATION_NAME, 
        operation_func=run_hdbet_on_patient, 
        paths=paths, 
        params=PARAMETERS, 
        patient_files_to_process=patient_files,
        operation_kwargs=func_inputs
    )

if __name__ == "__main__":
    main()


