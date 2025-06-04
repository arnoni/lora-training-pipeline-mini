#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/training/zenml_pipeline.py
# ----------------------------------------
# ZenML pipeline definition for LoRA fine-tuning.
# ----------------------------------------

# Import error handling tools first to avoid NameError
from src.lora_training_pipeline.utils.helpers import safe_run

# Check required dependencies first
try:
    import sys
    from pathlib import Path
    import os
    import time
    import subprocess
    import atexit
    import signal
    import json
    from typing import Dict
    # Add the project root to PATH so we can import our module
    root_dir = Path(__file__).resolve().parents[3]  # Go up 3 levels from current file
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.lora_training_pipeline.utils.helpers import check_dependencies
except ImportError as e:
    print(f"ERROR: Missing critical dependency: {e}")
    print("Please make sure all dependencies are installed with: uv pip install -e .")
    sys.exit(1)

# Check specific dependencies for this script
check_dependencies(['zenml', 'pandas', 'mlflow'])

# Third-party imports
import pandas as pd
import mlflow
from zenml.pipelines import pipeline
from zenml import step, get_pipeline_context  # Import step directly from zenml
from typing import Tuple, Dict, Any
# Try different ways to import Annotated for compatibility with different Python/ZenML versions
try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    try:
        from typing_extensions import Annotated  # Backport for < 3.9
    except ImportError:
        # Create a dummy Annotated if it's not available
        class Annotated:
            def __new__(cls, tp, *args, **kwargs):
                return tp
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

# Local imports with proper package paths
from src.lora_training_pipeline.data_cleaning.clean_filter import clean_and_filter_data
from src.lora_training_pipeline.training.hyperparameter_tuning import hyperparameter_tuning

# --- Constants ---
INFERENCE_SERVER_PORT = 8001

# Make sure file paths work on both Windows and Linux
def get_file_path(name):
    """Creates a platform-independent file path that works on both Windows and Linux."""
    return Path(os.path.join(".", name))

# Create file paths
FIRST_RUN_INDICATOR_FILE = get_file_path(".first_run")
MODEL_UPDATE_SIGNAL_FILE = get_file_path(".model_update")
INFERENCE_PROCESS_PID_FILE = get_file_path('inference_process.pid')
BEST_MODEL_METRICS_FILE = get_file_path("best_model_metrics.json")
BEST_MODEL_PATH = Path(os.path.join(".", "output", "best_model"))
TRAINING_LOCK_FILE = get_file_path(".training_lock") # Lock file to prevent concurrent training

@step
def check_data_sufficiency() -> Annotated[Tuple[bool, str], "sufficiency_check"]:
    """
    Checks data sufficiency and retrieves training file path from previous steps.
    
    This step also logs clear messages when data is insufficient, ensuring
    early pipeline termination with proper status reporting.
    
    Returns:
        Tuple containing:
        - dataset_ok: Boolean indicating if data is sufficient for training
        - training_file_path: Path to the training file if available
    """
    print("\n" + "="*80)
    print("PIPELINE STATUS: Performing data sufficiency check")
    print("="*80)
    
    try:
        # Get pipeline context to access previous steps
        within_pipeline = get_pipeline_context()
        print("✅ check_data_sufficiency is running within a ZenML pipeline")
    except RuntimeError:
        print("❌ check_data_sufficiency is NOT running within a pipeline")
        
        # Log the error as a permanent record
        from src.lora_training_pipeline.utils.helpers import log_pending_error
        import datetime
        log_pending_error(f"[{datetime.datetime.now()}] Failed to access ZenML pipeline context in check_data_sufficiency")
        
        # Update training status for dashboard and monitoring
        try:
            update_training_status(success=False)
            print("✅ Status recorded: Context access failure in sufficiency check")
        except Exception as status_error:
            print(f"[DEBUG] Training status update error type: {type(status_error).__name__}")
            print(f"[DEBUG] Training status update error details: {status_error}")
            print(f"⚠️ Failed to update training status: {status_error}")
            import traceback
            print(f"[DEBUG] Status update error traceback: {traceback.format_exc()}")
        
        # Return early with failure status
        print("❌ THRESHOLD CHECK FAILED: Cannot verify data sufficiency")
        print("="*80 + "\n")
        return False, ""

    # Default values
    dataset_ok = False
    training_file_path = ""

    try:
        # Get the last successful run with better error handling
        try:
            runs = within_pipeline.get_runs()
            if not runs:
                print("❌ No previous pipeline runs found")
                return False, ""
                
            last_successful_run = runs[0]
        except Exception as runs_error:
            print(f"[DEBUG] Pipeline runs access error type: {type(runs_error).__name__}")
            print(f"[DEBUG] Pipeline runs access error details: {runs_error}")
            print(f"❌ Failed to access pipeline runs: {runs_error}")
            import traceback
            print(f"[DEBUG] Pipeline runs error traceback: {traceback.format_exc()}")
            return False, ""
        
        # Find the clean_and_filter_data step
        step_found = False
        for step_run in last_successful_run.steps:
            if step_run.name == "clean_and_filter_data":
                step_found = True
                # Load outputs using ZenML's artifact system
                try:
                    dataset_ok = step_run.outputs["dataset_ok"].load()
                except Exception as load_error:
                    print(f"[DEBUG] Dataset loading error type: {type(load_error).__name__}")
                    print(f"[DEBUG] Dataset loading error details: {load_error}")
                    print(f"❌ Failed to load dataset_ok output: {load_error}")
                    import traceback
                    print(f"[DEBUG] Dataset loading error traceback: {traceback.format_exc()}")
                    return False, ""
                    
                try:
                    training_file_path = step_run.outputs["training_file_path"].load()
                except Exception as path_error:
                    print(f"❌ Failed to load training_file_path output: {path_error}")
                    return False, ""
                
                print(f"✅ Retrieved data from previous step:")
                print(f"  - Dataset OK for training: {dataset_ok}")
                print(f"  - Training file path: {training_file_path}")
                break
                
        if not step_found:
            print("❌ Data cleaning step not found in pipeline run")
            return False, ""
        
        # Validate the results
        if not dataset_ok:
            print("❌ Data threshold not met in cleaning stage")
            # Log status before early return for tracking purposes
            update_training_status(success=False)
            print("✅ Status recorded: Insufficient data in cleaning stage")
        elif not training_file_path:
            print("❌ No training file path provided from cleaning stage")
            dataset_ok = False  # Override to False if no path
            # Log status before early return for tracking purposes
            update_training_status(success=False)
            print("✅ Status recorded: Missing training file path")
        else:
            print("✅ Data is sufficient for training")
            print(f"✅ Training will use file: {training_file_path}")
            print("✅ Proceeding with training pipeline")
    except Exception as e:
        print(f"❌ Error retrieving data from previous steps: {e}")
        
        # Log the error for permanent record
        from src.lora_training_pipeline.utils.helpers import log_pending_error
        import datetime
        log_pending_error(f"[{datetime.datetime.now()}] Error in check_data_sufficiency: {e}")
        
        # Update training status for early termination record
        try:
            update_training_status(success=False)
            print("✅ Status recorded: Error occurred during sufficiency check")
        except Exception as status_error:
            print(f"⚠️ Failed to update training status: {status_error}")
            
        # Release the lock early if error occurs
        try:
            release_training_lock()
            print("🔓 Released training lock early due to error")
        except Exception as lock_error:
            print(f"⚠️ Warning: Error releasing lock: {lock_error}")
            
        print("❌ THRESHOLD CHECK FAILED: Error during data verification")
        print("="*80 + "\n")
        return False, ""

    print(f"SUFFICIENCY CHECK RESULT: {'PASSED' if dataset_ok else 'FAILED'}")
    print("="*80)
    
    return bool(dataset_ok), training_file_path

def is_first_run() -> bool:
    """Checks if this is the first training run."""
    return not FIRST_RUN_INDICATOR_FILE.exists()

def mark_first_run_complete():
    """Marks the first run as complete."""
    FIRST_RUN_INDICATOR_FILE.touch()
    
def update_training_status(success=True):
    """Updates the training cycle information with success/failure status."""
    import json
    import time
    import os
    from pathlib import Path

    # Path to the training cycle info file
    training_cycle_file = Path(os.path.join(".", "training_cycle_info.json"))

    # Load existing data
    if training_cycle_file.exists():
        try:
            with open(training_cycle_file, "r") as f:
                cycle_info = json.load(f)
        except Exception as e:
            print(f"Error loading training cycle info: {e}")
            cycle_info = {}
    else:
        cycle_info = {}

    # Update timestamp based on success or failure
    current_time = time.time()

    # Always update attempt time
    cycle_info["last_training_attempt_time"] = current_time

    # Update success time only if succeeded
    if success:
        cycle_info["last_successful_training_time"] = current_time

    # Set success status
    cycle_info["training_success"] = success

    # Save back to file
    try:
        with open(training_cycle_file, "w") as f:
            json.dump(cycle_info, f, indent=2)
        print(f"Updated training status: success={success}")
    except Exception as e:
        print(f"Error updating training cycle info: {e}")
    
def is_training_running() -> bool:
    """
    Checks if a training cycle is currently running.
    
    The function first checks if the lock file exists. If it does, it reads the PID
    from the file and verifies if that process is still running.
    This prevents false positives if a previous run crashed without removing the lock.
    
    Returns:
        bool: True if a training cycle is currently running, False otherwise
    """
    if not TRAINING_LOCK_FILE.exists():
        return False
    
    try:
        # Read PID from lock file
        pid = int(TRAINING_LOCK_FILE.read_text().strip())
        
        # Check if process is running - works on both Windows and Linux
        try:
            if os.name == 'nt':  # Windows
                # On Windows, use the subprocess module to check if the process exists
                import subprocess
                output = subprocess.check_output(f'tasklist /FI "PID eq {pid}" /NH', shell=True)
                return len(output.strip()) > 0
            else:  # Linux/Unix
                # On Unix-like systems, we can use os.kill with signal 0
                os.kill(pid, 0)
                
                # On Linux, we can also verify it's the right process by checking cmdline
                try:
                    with open(f"/proc/{pid}/cmdline", "r") as f:
                        cmdline = f.read()
                        if "python" in cmdline and "zenml_pipeline.py" in cmdline:
                            return True
                        else:
                            # PID exists but it's not our Python process, remove stale lock
                            TRAINING_LOCK_FILE.unlink()
                            return False
                except FileNotFoundError:
                    # If we can't check the cmdline, assume it's our process
                    return True
                
        except (OSError, subprocess.CalledProcessError):
            # Process doesn't exist, remove stale lock
            TRAINING_LOCK_FILE.unlink()
            return False
        
        except (ValueError, IOError) as e:
            # Invalid PID in file or couldn't read file
            print(f"⚠️ WARNING: Training lock file exists but could not be read: {e}")
            TRAINING_LOCK_FILE.unlink()
            return False
    
    except Exception as training_check_err:
        print(f"[DEBUG] Training lock check error type: {type(training_check_err).__name__}")
        print(f"[DEBUG] Training lock check error details: {training_check_err}")
        # Catch-all for any other exceptions
        if TRAINING_LOCK_FILE.exists():
            TRAINING_LOCK_FILE.unlink()
        return False

    return False  # Fallback

def acquire_training_lock() -> bool:
    """
    Tries to acquire the training lock.
    
    Returns:
        bool: True if lock was acquired, False otherwise
    """
    if is_training_running():
        return False
    
    # Create lock file with current PID
    try:
        TRAINING_LOCK_FILE.write_text(str(os.getpid()))
        return True
    except IOError as e:
        print(f"❌ ERROR: Failed to acquire training lock: {e}")
        return False

def release_training_lock() -> None:
    """Releases the training lock."""
    if TRAINING_LOCK_FILE.exists():
        try:
            TRAINING_LOCK_FILE.unlink()
        except IOError as e:
            print(f"❌ ERROR: Failed to release training lock: {e}")

def signal_model_update(model_path: str):
    """Signals the inference server to reload the model."""
    MODEL_UPDATE_SIGNAL_FILE.write_text(model_path)
    print(f"Signaled model update. New model path: {model_path}")

def start_inference_server(model_path: str):
    """Starts the FastAPI inference server (if not already running)."""

    if INFERENCE_PROCESS_PID_FILE.exists():
        print("Inference server already running.")
        return

    print(f"Starting inference server with model path: {model_path}")
    env = os.environ.copy()
    env["LORA_MODEL_PATH"] = str(model_path)
    env["MODEL_UPDATE_SIGNAL_FILE"] = str(MODEL_UPDATE_SIGNAL_FILE)

    try:
        process = subprocess.Popen(
            ["uvicorn", "src.inference.fastapi_inference:app", "--reload", "--host", "0.0.0.0", "--port", str(INFERENCE_SERVER_PORT)],
            env=env
        )
        INFERENCE_PROCESS_PID_FILE.write_text(str(process.pid))

        time.sleep(5)
        print("Inference server started.")
    except Exception as e:
        print(f"Error starting inference server: {e}")

def stop_inference_server():
    """Stops the FastAPI inference server."""
    if INFERENCE_PROCESS_PID_FILE.exists():
        try:
            pid = int(INFERENCE_PROCESS_PID_FILE.read_text())
            print(f"Stopping inference server (PID: {pid})...")
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            INFERENCE_PROCESS_PID_FILE.unlink()
            print("Inference server stopped.")

        except ProcessLookupError:
            print("Inference server process not found.")
            INFERENCE_PROCESS_PID_FILE.unlink()
        except Exception as e:
             print(f"Error stopping inference server: {e}")
    else:
        print("Inference server not running.")

# Only stop the inference server on exit if specifically requested by environment variable
def stop_inference_server_if_requested():
    """Only stop the inference server if specifically requested by setting an environment variable."""
    if os.environ.get("STOP_INFERENCE_ON_EXIT", "false").lower() == "true":
        print("Environment variable STOP_INFERENCE_ON_EXIT is set to true, stopping inference server...")
        stop_inference_server()
    else:
        print("Preserving FastAPI inference server on exit (set STOP_INFERENCE_ON_EXIT=true to change this behavior)")

# Register the safer version that checks the environment variable
atexit.register(stop_inference_server_if_requested)

def signal_handler(sig, frame):
    print('\nSignal received. Terminating pipeline...')
    print('Note: The FastAPI inference server will continue running.')
    print('To stop the inference server later, run: python -c "from src.lora_training_pipeline.training.zenml_pipeline import stop_inference_server; stop_inference_server()"')
    
    # Optional: Release any locks or resources
    try:
        if 'release_training_lock' in globals():
            release_training_lock()
            print("Training lock released")
    except Exception as e:
        print(f"Error releasing resources: {e}")
        
    # Exit immediately (default behavior)
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def get_best_model_metrics():
    """Loads the best model metrics from the JSON file (if it exists)."""
    if BEST_MODEL_METRICS_FILE.exists():
        try:
            with open(BEST_MODEL_METRICS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading best model metrics: {e}")
            return None
    else:
        return None

def save_best_model_metrics(metrics: dict):
    """Saves the best model metrics to the JSON file."""
    try:
        with open(BEST_MODEL_METRICS_FILE, "w") as f:
            json.dump(metrics, f)
    except Exception as e:
        print(f"Error saving best model metrics: {e}")

@step
@safe_run
def check_and_update_best_model(best_config: Dict, best_checkpoint_path: str, best_model_path:str) -> Annotated[str, "updated_model_path"]:
    """Checks if the new model is better than the current best and updates if necessary."""
    print("\n" + "="*80)
    print("PIPELINE STATUS: Checking and updating best model")
    print("="*80)
    
    # Get previous best metrics
    previous_best_metrics = get_best_model_metrics()
    if previous_best_metrics:
        print(f"ℹ️ MODEL STATUS: Previous best model available with validation loss: {previous_best_metrics['best_val_loss']:.4f}")
    else:
        print("ℹ️ MODEL STATUS: No previous best model found. This will be the first model saved.")
    
    # Get current model metrics from MLflow
    print("ℹ️ LOGGING STATUS: Retrieving model metrics from MLflow")
    mlflow.set_tracking_uri(get_tracking_uri())
    experiment_name = "lora_finetuning_final"
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    
    if runs.empty:
        print("⚠️ WARNING: No MLflow runs found. This is unexpected.")
        # Create a fallback validation loss that's worse than any reasonable loss
        best_val_loss = float('inf')
        best_run_id = None
    else:
        # Sort runs by validation loss and get the best one
        best_run = runs.sort_values(by='metrics.best_val_loss', ascending=True).iloc[0]
        best_run_id = best_run['run_id']
        best_val_loss = best_run['metrics.best_val_loss']
        print(f"ℹ️ LOGGING STATUS: Found best run (ID: {best_run_id}) with validation loss: {best_val_loss:.4f}")
    
    # Compare with previous best and update if better
    if previous_best_metrics is None or best_val_loss < previous_best_metrics["best_val_loss"]:
        if previous_best_metrics is None:
            print("ℹ️ MODEL STATUS: First run - saving as best model")
        else:
            print(f"✅ MODEL STATUS: New model is better than previous best!")
            print(f"ℹ️ Validation loss improved: {previous_best_metrics['best_val_loss']:.4f} → {best_val_loss:.4f}")
        
        # Save new metrics
        metrics = {"best_val_loss": best_val_loss, "model_path": str(BEST_MODEL_PATH)}
        save_best_model_metrics(metrics)
        
        # Copy model files to best model directory
        if os.path.exists(best_model_path):
            # Clear existing directory if it exists
            if os.path.exists(BEST_MODEL_PATH):
                print(f"ℹ️ MODEL STATUS: Clearing previous best model directory")
                for file in os.listdir(BEST_MODEL_PATH):
                    file_path = os.path.join(BEST_MODEL_PATH, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            import shutil
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"⚠️ WARNING: Failed to delete {file_path}. Reason: {e}")
            
            # Copy new model files
            print(f"ℹ️ MODEL STATUS: Copying new best model to {BEST_MODEL_PATH}")
            import shutil
            shutil.copytree(best_model_path, str(BEST_MODEL_PATH), dirs_exist_ok=True)
            print("✅ MODEL STATUS: Best model updated successfully")
        else:
            print(f"❌ ERROR: Model path does not exist: {best_model_path}")
            if previous_best_metrics is not None:
                print("ℹ️ MODEL STATUS: Keeping previous best model")
                return previous_best_metrics["model_path"]
        
        print("="*80 + "\n")
        return str(BEST_MODEL_PATH)
    else:
        print(f"ℹ️ MODEL STATUS: Current model (val_loss: {best_val_loss:.4f}) is not better than previous best (val_loss: {previous_best_metrics['best_val_loss']:.4f})")
        print("ℹ️ MODEL STATUS: Keeping previous best model")
        print("="*80 + "\n")
        return previous_best_metrics["model_path"]



@step
@safe_run
def launch_or_update_inference(model_path: str) -> None:
    """Launches or updates the inference server with the new model.
    
    This step either starts a new inference server if none is running,
    or signals the existing server to update its model.
    
    Args:
        model_path: Path to the model to be used for inference
    """
    from src.lora_training_pipeline.utils.helpers import ProgressTracker, check_zenml_connection
    
    # First check ZenML connection for this critical step
    print("\n" + "="*80)
    print("CHECKING ZENML CONNECTION STATUS BEFORE MODEL DEPLOYMENT")
    connection_ok = check_zenml_connection(max_retries=2, retry_delay=3)
    if not connection_ok:
        print("⚠️ WARNING: ZenML server connection issues detected")
        print("ℹ️ Will continue with deployment but metadata logging may be affected")
    print("="*80)
    
    deployment_progress = ProgressTracker("Model Deployment")
    deployment_progress.start("Deploying model for inference")
    
    print("\n" + "="*80)
    print("PIPELINE STATUS: Deploying model for inference")
    print("="*80)
    
    if INFERENCE_PROCESS_PID_FILE.exists():
        try:
            pid = int(INFERENCE_PROCESS_PID_FILE.read_text())
            # Check if process is actually running
            try:
                deployment_progress.update(message="Checking existing inference server")
                os.kill(pid, 0)  # Signal 0 doesn't kill the process, just checks if it exists
                print(f"✅ INFERENCE STATUS: Inference server is already running (PID: {pid})")
                print(f"ℹ️ Signaling model update to: {model_path}")
                
                deployment_progress.update(message="Signaling model update to existing server")
                signal_model_update(model_path)
                print("✅ Update signal sent successfully")
                
            except OSError:
                deployment_progress.update(message="Found stale inference server PID, restarting server")
                print("⚠️ INFERENCE STATUS: Inference server PID file exists but process is not running")
                print("ℹ️ Starting new inference server...")
                INFERENCE_PROCESS_PID_FILE.unlink()  # Remove stale PID file
                
                # Start timer for server initialization
                deployment_progress.start_timer("Starting inference server")
                start_inference_server(model_path)
                
        except Exception as e:
            deployment_progress.update(message=f"Error with inference server: {str(e)}")
            print(f"❌ ERROR: Failed to check/update inference server: {e}")
            print("ℹ️ Attempting to start a new inference server...")
            
            # Start timer for server initialization
            deployment_progress.start_timer("Starting inference server")
            start_inference_server(model_path)
    else:
        deployment_progress.update(message="No inference server running, starting new server")
        print("ℹ️ INFERENCE STATUS: No inference server running")
        print(f"ℹ️ Starting inference server with model: {model_path}")
        
        # Start timer for server initialization
        deployment_progress.start_timer("Starting inference server")
        start_inference_server(model_path)
    
    # Verify inference server is running and perform final connection check
    deployment_progress.update(message="Verifying inference server status")
    if INFERENCE_PROCESS_PID_FILE.exists():
        deployment_progress.complete("Inference server successfully deployed")
        print("✅ INFERENCE STATUS: Inference server is now running")
        print(f"ℹ️ Server port: {INFERENCE_SERVER_PORT}")
        print("ℹ️ You can use the Inference UI to interact with the model")
        
        # Perform one final ZenML connection check
        print("\nPerforming final ZenML connection verification...")
        check_zenml_connection(max_retries=1, retry_delay=1)
    else:
        deployment_progress.complete("Failed to start inference server")
        print("❌ INFERENCE STATUS: Failed to start inference server")
        print("ℹ️ RECOMMENDED ACTION: Check logs for errors and try restarting the pipeline")
    
    print("="*80 + "\n")

def verify_zenml_local_server(max_retries=3, retry_delay=5):
    """
    Verifies that ZenML is using a local server configuration.
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Seconds to wait between retry attempts
        
    Returns:
        bool: True if using local server, False otherwise
    """
    from src.lora_training_pipeline.utils.helpers import ProgressTracker, log_pending_error
    
    for attempt in range(1, max_retries + 1):
        try:
            # Create a connection progress tracker
            if attempt > 1:
                print(f"\n🔄 RECONNECTION ATTEMPT {attempt}/{max_retries} TO ZENML SERVER")
            
            connection_progress = ProgressTracker("ZenML Connection")
            connection_progress.start("Connecting to ZenML server")
            
            from zenml.client import Client
            client = Client()
            connection_progress.update(message="Connected, retrieving server info")
            
            # Define dark green color for ZenML-specific prints
            DARK_GREEN = "\033[32m"
            RESET = "\033[0m"
            
            # Get server info with more detailed printing
            print(f"\n{DARK_GREEN}[DEBUG-ZENML] Getting ZenML server info...{RESET}")
            server_info = client.zen_store.get_store_info()
            print(f"{DARK_GREEN}[DEBUG-ZENML] Server info type: {type(server_info)}{RESET}")
            print(f"{DARK_GREEN}[DEBUG-ZENML] Server info attributes: {dir(server_info)[:10]}...{RESET}")
            connection_progress.complete("Successfully connected to ZenML server")
            
            # Check if using local store with improved detection
            try:
                # --- Improved server type detection ---
                server_type = None
                if hasattr(server_info, 'type'):
                    server_type = server_info.type
                elif hasattr(server_info, 'get_type') and callable(server_info.get_type):
                    server_type = server_info.get_type()
                elif hasattr(server_info, 'database_type'):
                    server_type = server_info.database_type
                print(f"{DARK_GREEN}[DEBUG-ZENML] Server type: {server_type}{RESET}")
                
                # --- Improved URL detection ---
                server_url = None
                if hasattr(server_info, 'url'):
                    server_url = server_info.url
                elif hasattr(server_info, 'get_url') and callable(server_info.get_url):
                    server_url = server_info.get_url()
                elif hasattr(server_info, 'url_str'):
                    server_url = server_info.url_str
                elif hasattr(server_info, 'server_url'):
                    server_url = server_info.server_url
                print(f"{DARK_GREEN}[DEBUG-ZENML] Server URL: {server_url}{RESET}")
                
                # --- Check deployment type ---
                deployment_type = None
                if hasattr(server_info, 'deployment_type'):
                    deployment_type = server_info.deployment_type
                print(f"{DARK_GREEN}[DEBUG-ZENML] Deployment type: {deployment_type}{RESET}")
                
                # Convert to strings for safety
                server_type_str = str(server_type) if server_type is not None else ""
                server_url_str = str(server_url) if server_url is not None else ""
                deployment_type_str = str(deployment_type) if deployment_type is not None else ""
                
                # Log raw info for debugging
                print(f"{DARK_GREEN}[DEBUG-ZENML] Raw server info: {server_info}{RESET}")
                
                # Improved local detection - much more generous with what counts as local
                is_local = True  # Default to assuming local for safety
                is_definitely_remote = False
                
                # Check for clear signs of remote server
                if server_url_str and (
                    "http:" in server_url_str or 
                    "https:" in server_url_str
                ) and not (
                    "localhost" in server_url_str or
                    "127.0.0.1" in server_url_str
                ):
                    # If URL is a non-localhost HTTP URL, it's definitely remote
                    is_definitely_remote = True
                
                # Check for explicit remote deployment type
                if "remote" in deployment_type_str.lower() or "cloud" in deployment_type_str.lower():
                    is_definitely_remote = True
                
                # Override default if we're sure it's remote
                if is_definitely_remote:
                    is_local = False
                
                # Final determination with clear logging
                if is_local:
                    print("\n✅ ZenML is using a local server")
                    print(f"ℹ️ Server type: {server_type_str}")
                    print(f"ℹ️ URL: {server_url_str}")
                    print(f"ℹ️ Deployment: {deployment_type_str}")
                else:
                    print("\n⚠️ WARNING: ZenML appears to be using a remote server")
                    print(f"ℹ️ Server info: {server_info}")
                    print("ℹ️ The pipeline will continue, but be aware data is stored remotely")
                    
                    # Log warning to pending_errors.txt
                    log_pending_error(f"ZenML using remote server: {server_info}")
            except Exception as attr_error:
                # Handle attribute access errors with more robustness
                # Define dark green color if not already defined
                if 'DARK_GREEN' not in locals():
                    DARK_GREEN = "\033[32m"
                    RESET = "\033[0m"
                
                print(f"{DARK_GREEN}[DEBUG-ZENML] Error accessing server attributes: {attr_error}{RESET}")
                server_info_str = str(server_info)
                print(f"{DARK_GREEN}[DEBUG-ZENML] Server info string: {server_info_str[:200]}...{RESET}")
                
                # Very generous local detection from string representation
                remote_indicators = [
                    "http://", "https://", 
                    "remote", "cloud",
                    "aws", "gcp", "azure", 
                    "kubernetes", "k8s"
                ]
                
                # Only consider it remote if we find clear evidence
                is_remote = any(indicator in server_info_str.lower() for indicator in remote_indicators)
                is_local = not is_remote
                
                print(f"{DARK_GREEN}[DEBUG-ZENML] String analysis determined server is {'remote' if is_remote else 'local'}{RESET}")
                
                if is_remote:
                    log_pending_error(f"ZenML might be using remote server: {server_info_str[:200]}...")
                
                # Log the decision
                print(f"\n{'✅' if is_local else '⚠️'} ZenML server detected as {'LOCAL' if is_local else 'REMOTE'}")
            
            return is_local
            
        except Exception as e:
            if "Connection refused" in str(e) or "Failed to connect" in str(e):
                error_msg = f"ZENML SERVER CONNECTION LOST - {str(e)}"
                print(f"\n❌ ERROR: {error_msg}")
                
                # Log connection error to pending_errors.txt
                log_pending_error(error_msg)
                
                if attempt < max_retries:
                    reconnect_wait = ProgressTracker("Reconnection Wait")
                    reconnect_wait.start(f"Waiting {retry_delay}s before reconnection attempt {attempt+1}/{max_retries}")
                    reconnect_wait.start_timer("Reconnection delay", interval=1)
                    import time
                    time.sleep(retry_delay)
                    reconnect_wait.complete("Retry delay complete")
                else:
                    critical_error = f"ALL {max_retries} ZENML RECONNECTION ATTEMPTS FAILED - {str(e)}"
                    print(f"\n❌ CRITICAL ERROR: {critical_error}")
                    print("ℹ️ Check that the ZenML server is running and accessible")
                    print("ℹ️ You can try restarting the ZenML server with: zenml up")
                    
                    # Log critical error to pending_errors.txt
                    log_pending_error(f"CRITICAL: {critical_error}")
            else:
                warning_msg = f"Could not verify ZenML server configuration: {e}"
                print(f"\n⚠️ WARNING: {warning_msg}")
                print("ℹ️ Will assume local configuration and continue")
                
                # Log warning to pending_errors.txt
                log_pending_error(warning_msg)
                return True
    
    print("ℹ️ WARNING: Proceeding without ZenML server connection")
    log_pending_error("Proceeding without ZenML server connection after multiple failures")
    return True

@pipeline
def lora_training_pipeline(run_local: bool = True):
    """
    Main training pipeline using ZenML's native artifact system.
    
    The pipeline follows these steps:
    1. Data cleaning and validation (clean_and_filter_data)
    2. Threshold verification (check_data_sufficiency)
    3. Model training (hyperparameter_tuning) - only if threshold is met
    4. Model deployment - only if training succeeds
    
    Each step communicates with subsequent steps through ZenML artifacts,
    maintaining proper isolation between pipeline stages.
    """
    # Import progress tracking and ZenML monitoring
    from src.lora_training_pipeline.utils.helpers import (
        ProgressTracker, 
        monitor_zenml_connection, 
        check_pending_errors,
        log_pending_error,
        clear_pending_errors
        # safe_run already imported at the top of the file
    )
    
    # We don't need to check for pending errors here since the main pipeline already
    # initialized the error log at startup. This pipeline will log its own errors
    # to pending_errors.txt during execution for later review.
    
    # Create a master progress tracker for the pipeline
    pipeline_progress = ProgressTracker("ZenML Pipeline", total=4)
    pipeline_progress.start("Starting LoRA training pipeline")
    
    print("\n" + "="*80)
    print("STARTING LORA TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Check environment and acquire lock
    pipeline_progress.update(message="Verifying environment and checking locks")
    
    # Track the verification process
    verify_progress = ProgressTracker("Environment Verification")
    verify_progress.start("Checking ZenML configuration")
    
    # Verify ZenML is using local server
    connection_ok = verify_zenml_local_server()
    
    # Log error if connection failed
    if not connection_ok:
        log_pending_error("ZenML server connection failed at pipeline startup")
        print("\n⚠️ WARNING: This error has been logged to pending_errors.txt")
        print("ℹ️ The pipeline will continue but may have limited functionality")
    
    # Start ZenML connection monitoring in background thread
    if connection_ok:
        monitor = monitor_zenml_connection(interval=30, max_retries=2)
    else:
        print("\n⚠️ WARNING: ZenML connection monitoring not started due to initial connection failure")
        
    verify_progress.complete("ZenML configuration verified")
    
    # Check for existing training process
    lock_progress = ProgressTracker("Lock Management")
    lock_progress.start("Checking for existing training processes")
    
    # Check if another training cycle is already running
    if is_training_running():
        lock_progress.complete("Another training cycle already running")
        pipeline_progress.complete("Pipeline exiting - another process is already running")
        print("\n" + "="*80)
        print("❌ PIPELINE STATUS: Another training cycle is already running")
        print("ℹ️ REASON: Training lock is held by another process")
        print("ℹ️ RECOMMENDED ACTION: Wait for the current training cycle to complete")
        print("ℹ️ The current run will exit now")
        print("="*80 + "\n")
        
        # Log this event for debugging
        from src.lora_training_pipeline.utils.helpers import log_pending_error
        import datetime
        log_pending_error(f"[{datetime.datetime.now()}] ZenML pipeline execution skipped due to active training lock")
        
        return
        
    # Check for threshold before acquiring the lock - this is a fast pre-check
    # Retrieve the dataset_ok status from the previous steps to make sure we don't
    # proceed when we shouldn't
    try:
        print("\n" + "="*80)
        print("🔍 PERFORMING PRE-TRAINING VALIDATION CHECK")
        print("="*80)
        
        # Use within_pipeline to get the last run's result if we're part of a pipeline
        try:
            within_pipeline = get_pipeline_context()
            print("✅ Running within ZenML pipeline - can access step outputs")
            
            # Using a more robust approach to get pipeline runs
            try:
                # Try different methods to get runs depending on ZenML version
                if hasattr(within_pipeline, 'get_runs') and callable(within_pipeline.get_runs):
                    runs = within_pipeline.get_runs()
                    if runs:
                        last_successful_run = runs[0]
                    else:
                        print("❌ No pipeline runs found")
                        return
                elif hasattr(within_pipeline, 'runs') and within_pipeline.runs:
                    last_successful_run = within_pipeline.runs[0]
                else:
                    # Try accessing client directly
                    from zenml.client import Client
                    client = Client()
                    pipeline_runs = client.list_pipeline_runs()
                    if pipeline_runs:
                        last_successful_run = pipeline_runs[0]
                    else:
                        print("❌ No pipeline runs found via client")
                        return
                
                print(f"✅ Found pipeline run: {last_successful_run.name}")
                dataset_ok = None
                
                # Check for the clean_and_filter_data step
                for step_run in last_successful_run.steps:
                    print(f"🔍 Examining step: {step_run.name}")
                    if step_run.name == "clean_and_filter_data":
                        try:
                            dataset_ok = step_run.outputs["dataset_ok"].load()
                            print(f"🔎 Found clean_and_filter_data step output: dataset_ok={dataset_ok}")
                        except Exception as step_error:
                            print(f"⚠️ Error loading step output: {step_error}")
                        break
            except Exception as e:
                print(f"⚠️ Error accessing pipeline run details: {e}")
                dataset_ok = None
                    
            if dataset_ok is not None and not dataset_ok:
                print("❌ PRE-CHECK FAILED: Data validation step reported insufficient data")
                print("ℹ️ REASON: The dataset_ok output from the cleaning step was False")
                print("ℹ️ RECOMMENDED ACTION: This run was triggered incorrectly - collect more data")
                print("ℹ️ The pipeline will exit now")
                print("="*80 + "\n")
                
                # Log this event for debugging
                log_pending_error(f"[{datetime.datetime.now()}] ZenML pipeline execution incorrect - threshold not met")
                return
                
            print("✅ PRE-CHECK PASSED: Proceeding with lock acquisition")
        except Exception as e:
            print(f"ℹ️ Could not access pipeline context: {e}")
            print("ℹ️ Skipping pre-check validation - will proceed with lock acquisition")
        
        print("="*80 + "\n")
    except Exception as e:
        print(f"ℹ️ Error during pre-check: {e}")
        print("ℹ️ Proceeding with caution")
    
    # Try to acquire training lock
    lock_progress.update(message="Attempting to acquire training lock")
    if not acquire_training_lock():
        lock_progress.complete("Failed to acquire lock")
        pipeline_progress.complete("Pipeline exiting - failed to acquire lock")
        print("\n" + "="*80)
        print("❌ PIPELINE STATUS: Failed to acquire training lock")
        print("ℹ️ RECOMMENDED ACTION: Check if another process is running or manually delete the lock file")
        print("ℹ️ Lock file location: " + str(TRAINING_LOCK_FILE))
        print("="*80 + "\n")
        return
    
    lock_progress.complete("Training lock acquired successfully")
    
    try:
        # Register function to release lock on exit
        atexit.register(release_training_lock)
        pipeline_progress.update(message="Lock acquired, starting data processing")
        print("✅ PIPELINE STATUS: Training lock acquired, proceeding with pipeline")
        
        # Step 2: Data cleaning and filtering with extensive debug logging
        data_progress = ProgressTracker("Data Processing")
        data_progress.start("Cleaning and filtering data")
        data_progress.start_timer("Processing data samples")
        
        # Add timestamp to track when data cleaning starts
        import datetime
        import os
        import sys
        
        # Define dark green color for ZenML-specific prints (ANSI color code for dark green)
        DARK_GREEN = "\033[32m"
        RESET = "\033[0m"
        
        print(f"\n{DARK_GREEN}[DEBUG-ZENML] Starting data cleaning process at: {datetime.datetime.now()}{RESET}")
        print(f"{DARK_GREEN}[DEBUG-ZENML] Current working directory: {os.getcwd()}{RESET}")
        print(f"{DARK_GREEN}[DEBUG-ZENML] Python executable: {sys.executable}{RESET}")
        
        # Log environment variables that might affect the data cleaning
        env_vars_to_check = ["PYTHONPATH", "PATH", "VIRTUAL_ENV", "SIMULATE_VALIDATION"]
        print(f"{DARK_GREEN}[DEBUG-ZENML] Relevant environment variables:{RESET}")
        for var in env_vars_to_check:
            print(f"{DARK_GREEN}[DEBUG-ZENML]   {var}: {os.environ.get(var, 'Not set')}{RESET}")
        
        # Check if the data directory exists and has data files
        from pathlib import Path
        data_dir = Path("./data")
        print(f"{DARK_GREEN}[DEBUG-ZENML] Checking data directory: {data_dir.absolute()}{RESET}")
        if data_dir.exists():
            print(f"{DARK_GREEN}[DEBUG-ZENML] Data directory exists{RESET}")
            # Check for data files
            parquet_files = list(data_dir.glob("*.parquet"))
            print(f"{DARK_GREEN}[DEBUG-ZENML] Found {len(parquet_files)} parquet files in data directory{RESET}")
            if parquet_files:
                for file in parquet_files[:5]:  # Show the first 5 files
                    print(f"{DARK_GREEN}[DEBUG-ZENML]   - {file.name} ({file.stat().st_size} bytes){RESET}")
        else:
            print(f"{DARK_GREEN}[DEBUG-ZENML] ⚠️ Data directory does not exist{RESET}")
        
        # Double-check that the clean_and_filter_data function is correctly imported
        from src.lora_training_pipeline.data_cleaning.clean_filter import clean_and_filter_data as imported_func
        print(f"{DARK_GREEN}[DEBUG-ZENML] Imported clean_and_filter_data function: {imported_func}{RESET}")
        
        # Detailed Ollama info before cleaning
        print(f"{DARK_GREEN}[DEBUG-ZENML] Checking for Ollama availability...{RESET}")
        try:
            import ollama
            print(f"{DARK_GREEN}[DEBUG-ZENML] ✅ Ollama library successfully imported{RESET}")
            print(f"{DARK_GREEN}[DEBUG-ZENML] Ollama version: {getattr(ollama, '__version__', 'unknown')}{RESET}")
            print(f"{DARK_GREEN}[DEBUG-ZENML] Ollama library path: {getattr(ollama, '__file__', 'unknown')}{RESET}")
            
            # Check if Ollama server is running
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                result = s.connect_ex(('localhost', 11434))
                if result == 0:
                    print(f"{DARK_GREEN}[DEBUG-ZENML] ✅ Ollama server is running on port 11434{RESET}")
                else:
                    print(f"{DARK_GREEN}[DEBUG-ZENML] ❌ Ollama server not detected on port 11434{RESET}")
                s.close()
            except Exception as conn_err:
                print(f"{DARK_GREEN}[DEBUG-ZENML] ⚠️ Could not check Ollama server: {conn_err}{RESET}")
        except ImportError as e:
            print(f"{DARK_GREEN}[DEBUG-ZENML] ❌ Failed to import Ollama: {e}{RESET}")
            print(f"{DARK_GREEN}[DEBUG-ZENML] This will affect data validation!{RESET}")
        
        # Detailed file check before cleaning
        print(f"{DARK_GREEN}[DEBUG-ZENML] Detailed file count check before cleaning:{RESET}")
        # Import DATASET_NAME and DATA_VERSION from config
        from src.lora_training_pipeline.config import DATASET_NAME, DATA_VERSION
        
        all_original = list(Path("./data").glob(f"{DATASET_NAME}_{DATA_VERSION}_original_*.parquet"))
        all_valid = list(Path("./data/valid").glob(f"{DATASET_NAME}_{DATA_VERSION}_valid_*.parquet")) if Path("./data/valid").exists() else []
        all_invalid = list(Path("./data/rejected").glob(f"{DATASET_NAME}_{DATA_VERSION}_invalid_*.parquet")) if Path("./data/rejected").exists() else []
        
        print(f"{DARK_GREEN}[DEBUG-ZENML] Original files: {len(all_original)}{RESET}")
        print(f"{DARK_GREEN}[DEBUG-ZENML] Valid files: {len(all_valid)}{RESET}")
        print(f"{DARK_GREEN}[DEBUG-ZENML] Invalid files: {len(all_invalid)}{RESET}")
        
        # Clean and filter data with comprehensive error handling
        try:
            print(f"{DARK_GREEN}[DEBUG-ZENML] Calling clean_and_filter_data() at {datetime.datetime.now()}{RESET}")
            print(f"{DARK_GREEN}[DEBUG-ZENML] Calling with parameters: dataset_name={DATASET_NAME}, version={DATA_VERSION}{RESET}")
            clean_data, dataset_ok, training_file_path = clean_and_filter_data(dataset_name=DATASET_NAME, version=DATA_VERSION)
            print(f"{DARK_GREEN}[DEBUG-ZENML] Data cleaning completed at: {datetime.datetime.now()}{RESET}")
            
            # Convert ZenML artifacts to actual values if needed
            if hasattr(dataset_ok, 'load') and callable(dataset_ok.load):
                dataset_ok = dataset_ok.load()
                print(f"{DARK_GREEN}[DEBUG-ZENML] Loaded dataset_ok from ZenML artifact: {dataset_ok}{RESET}")
                
            if hasattr(training_file_path, 'load') and callable(training_file_path.load):
                training_file_path = training_file_path.load()
                print(f"{DARK_GREEN}[DEBUG-ZENML] Loaded training_file_path from ZenML artifact: {training_file_path}{RESET}")
            
            # Try to get the length of clean_data (handle both DataFrame and artifact cases)
            clean_data_size = "unknown"
            if hasattr(clean_data, 'load') and callable(clean_data.load):
                try:
                    loaded_data = clean_data.load()
                    clean_data_size = len(loaded_data) if hasattr(loaded_data, '__len__') else "unknown (loaded)"
                    # Update clean_data to the loaded version
                    clean_data = loaded_data
                except Exception as load_err:
                    print(f"{DARK_GREEN}[DEBUG-ZENML] Error loading clean_data artifact: {load_err}{RESET}")
            elif hasattr(clean_data, '__len__'):
                clean_data_size = len(clean_data)
                
            print(f"{DARK_GREEN}[DEBUG-ZENML] Result: dataset_ok={dataset_ok}, clean_data_size={clean_data_size}, training_file_path={training_file_path}{RESET}")
            
            # Use actual boolean value for dataset_ok in the message
            if isinstance(dataset_ok, bool):
                threshold_status = 'sufficient data available' if dataset_ok else 'insufficient data'
            else:
                # Try to convert to boolean if possible
                try:
                    threshold_status = 'sufficient data available' if bool(dataset_ok) else 'insufficient data'
                except Exception as conversion_error:
                    print(f"Warning: Could not convert dataset_ok to boolean: {conversion_error}")
                    threshold_status = 'unknown status'
                    
            data_progress.complete(f"Data processing complete - {threshold_status}")
        except Exception as e:
            # Log the error for debugging with enhanced details
            from src.lora_training_pipeline.utils.helpers import log_pending_error
            import traceback
            error_msg = f"[{datetime.datetime.now()}] Error in data cleaning step: {str(e)}"
            trace_msg = traceback.format_exc()
            print(f"\n❌ ERROR: {error_msg}")
            print(f"{DARK_GREEN}[DEBUG-ZENML] Exception type: {type(e).__name__}{RESET}")
            print(f"{DARK_GREEN}[DEBUG-ZENML] Exception details: {str(e)}{RESET}")
            print(f"{DARK_GREEN}[DEBUG-ZENML] Traceback:\n{trace_msg}{RESET}")
            log_pending_error(error_msg)
            log_pending_error(f"[TRACEBACK] {trace_msg}")
            data_progress.complete(f"Data processing failed with error: {str(e)}")
            # Return with failure status
            pipeline_progress.complete("Pipeline failed during data cleaning")
            print("\n" + "="*80)
            print("❌ PIPELINE STATUS: Data cleaning failed with error")
            print(f"ℹ️ ERROR: {str(e)}")
            print("="*80 + "\n")
            return
        
        # Double-check threshold with hard validation
        print("\n" + "="*80)
        print("🔍 PERFORMING FINAL THRESHOLD VALIDATION")
        print("="*80)
        
        # Count valid files directly to be 100% sure
        valid_dir = Path("./data/valid")
        valid_files = []
        if valid_dir.exists():
            valid_files = list(valid_dir.glob(f"{DATASET_NAME}_{DATA_VERSION}_valid_*.parquet"))
        
        # Also check consolidated files
        consolidated_files = list(Path("./data").glob(f"{DATASET_NAME}_{DATA_VERSION}_consolidated_*.parquet"))
        
        # Get valid count from both sources
        valid_count = len(valid_files)
        if consolidated_files:
            try:
                # Get the newest consolidated file
                consolidated_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                consolidated_df = pd.read_parquet(consolidated_files[0])
                valid_count = max(valid_count, len(consolidated_df))
                print(f"📊 Found consolidated file with {len(consolidated_df)} data points")
            except Exception as e:
                print(f"⚠️ Error reading consolidated file: {e}")
        
        # Calculate threshold
        min_required = 10  # Minimum required valid data points
        
        # Final validation - must pass BOTH checks
        threshold_ok = valid_count >= min_required
        double_check_passed = dataset_ok and threshold_ok
        
        print(f"📊 Valid data points found: {valid_count}")
        print(f"📊 Required threshold: {min_required}")
        print(f"📊 Dataset_ok from cleaning stage: {dataset_ok}")
        print(f"📊 Double-check threshold validation: {threshold_ok}")
        print(f"📊 FINAL RESULT: {'PASSED' if double_check_passed else 'FAILED'}")
        print("="*80)
        
        # Strict threshold enforcement - training only proceeds if BOTH checks pass
        if double_check_passed:
            # Step 3: Model training
            pipeline_progress.update(message="Starting model training")
            print("\n" + "="*80)
            print("✅ THRESHOLD CHECK PASSED: Proceeding with training")
            print("PIPELINE STATUS: Starting hyperparameter tuning and training")
            print("="*80)
            
            # Check for and use Hugging Face token if available
            if "HUGGING_FACE_HUB_TOKEN" in os.environ:
                print(f"✅ HUGGING_FACE_HUB_TOKEN is set in environment variables and will be used for model access")
            else:
                # Try to load from .env file if it exists
                try:
                    env_path = Path('.env')
                    if env_path.exists():
                        print(f"Loading Hugging Face token from .env file")
                        import re
                        with open(env_path, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line and 'HUGGING_FACE_HUB_TOKEN' in line and not line.startswith('#'):
                                    key, value = re.split(r'=', line, 1)
                                    os.environ[key] = value
                                    print(f"✅ Successfully loaded {key} from .env file")
                                    break
                except Exception as e:
                    print(f"Note: Could not load from .env file: {e}")
            
            # Clean and filter data (producing training file artifact)
            clean_data, dataset_ok, training_file_path = clean_and_filter_data(dataset_name="user_data_ui", version="v1", disable_cache=False)
            
            # Get sufficiency check (consumes clean_data_ok and training_file_path artifacts)
            sufficiency_check = check_data_sufficiency()
            
            # Extract values from sufficiency check
            is_sufficient, checked_training_file = sufficiency_check
            
            # Only proceed with training if data is sufficient
            if is_sufficient and checked_training_file:
                print(f"\n{'='*80}\nTRAINING STATUS: Data is sufficient, proceeding with training\n{'='*80}")
                print(f"✅ Will use verified training file: {checked_training_file}")
                
                # This step will use the artifact from the sufficiency check
                # The hyperparameter_tuning step will track used/trained files using ZenML artifact metadata
                best_config, best_checkpoint_path, best_model_path = hyperparameter_tuning(
                    training_file_path=checked_training_file, 
                    run_local=run_local
                )
            else:
                print(f"\n{'='*80}\nTRAINING STATUS: Data is insufficient, skipping training\n{'='*80}")
                print("ℹ️ Will not proceed to training stage")
                return  # Exit pipeline
            
            # Step 4: Model deployment (only if we reached this point, meaning training was successful)
            pipeline_progress.update(message="Deploying trained model")
            
            # Track model update process
            deployment_progress = ProgressTracker("Model Deployment")
            deployment_progress.start("Checking and updating best model")
            
            updated_model_path = check_and_update_best_model(best_config, best_checkpoint_path, best_model_path)
            
            deployment_progress.update(message="Launching inference server with new model")
            launch_or_update_inference(updated_model_path)
            
            deployment_progress.complete("Model successfully deployed to inference server")
            pipeline_progress.complete("Training pipeline completed successfully")
            
            print("\n" + "="*80)
            print("✅ PIPELINE STATUS: Training pipeline completed successfully")
            print("ℹ️ Model is now available for inference")
            print("="*80 + "\n")
        else:
            # If dataset_ok is False, skip training and end the pipeline
            pipeline_progress.update(message="Skipping training due to insufficient data")
            print("\n" + "="*80)
            print("❌ THRESHOLD CHECK FAILED: Training will not proceed")
            print("The pipeline will not continue to the training stage because the minimum")
            print("required number of valid data points was not met.")
            print("ℹ️ The inference server will continue to use the previous model (if available)")
            print("ℹ️ RECOMMENDED ACTION: Use the data collection interface to submit more data samples")
            print("="*80)
            
            # Update training status to record this failed attempt
            try:
                # Record the attempt but mark it as failed
                update_training_status(success=False)
                print("✅ Status recorded: Training threshold check failed")
            except Exception as e:
                print(f"⚠️ Failed to record training status: {e}")
            
            # Complete the pipeline with a message about insufficient data
            pipeline_progress.complete("Pipeline completed - insufficient data for training")
            
            # Explicitly release the lock immediately when threshold check fails
            # This will allow other processes to run sooner rather than waiting
            try:
                print("🔓 Releasing training lock early due to failed threshold check")
                release_training_lock()
            except Exception as e:
                print(f"⚠️ Warning: Error releasing lock: {e}")
                
            return
    
    finally:
        # Always release the lock when done, even if an exception occurred
        release_training_lock()
        print("ℹ️ PIPELINE STATUS: Training lock released")
        
if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXECUTING ZENML PIPELINE DIRECTLY")
    print("="*80)
    
    # Install global exception handler for the ZenML pipeline
    def zenml_exception_handler(exc_type, exc_value, exc_traceback):
        """Handle any unhandled exceptions in ZenML pipeline with comprehensive logging."""
        import traceback
        import datetime
        
        error_msg = f"""
CRITICAL UNHANDLED EXCEPTION IN ZENML PIPELINE
==============================================
Time: {datetime.datetime.now().isoformat()}
Exception Type: {exc_type.__name__}
Exception Value: {exc_value}
Process ID: {os.getpid()}
Working Directory: {os.getcwd()}
Python Executable: {sys.executable}

FULL TRACEBACK:
{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}
"""
        
        # Print to stderr for immediate visibility
        print(error_msg, file=sys.stderr)
        
        # Also try to write to error log file
        try:
            from pathlib import Path
            error_file = Path("zenml_critical_errors.log")
            with open(error_file, "a", encoding="utf-8") as f:
                f.write(error_msg + "\n" + "="*80 + "\n")
            print(f"[CRITICAL] ZenML error logged to {error_file}", file=sys.stderr)
        except Exception as log_err:
            print(f"[CRITICAL] Failed to write ZenML error log: {log_err}", file=sys.stderr)
        
        # Try to clean up training lock before exiting
        try:
            release_training_lock()
            print("[CLEANUP] Training lock released due to critical error", file=sys.stderr)
        except Exception as cleanup_err:
            print(f"[CLEANUP] Failed to release training lock: {cleanup_err}", file=sys.stderr)
        
        # Call the original exception handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    # Install the exception handler
    sys.excepthook = zenml_exception_handler
    print("[DEBUG] ZenML global exception handler installed")
    
    # Create and run pipeline using ZenML 0.74.0 API
    try:
        # Import the ZenML client
        from zenml.client import Client
        
        # Print ZenML version info
        import zenml
        zenml_version = getattr(zenml, "__version__", "unknown")
        print(f"ZenML version: {zenml_version}")
        print(f"ZenML module path: {getattr(zenml, '__file__', 'unknown')}")
        
        # Initialize the client
        client = Client()
        print(f"ZenML client initialized: {type(client)}")
        
        # Get and validate the active stack
        try:
            active_stack = client.active_stack
            print(f"Current active stack: {active_stack.name}")
        except Exception as stack_error:
            print(f"Warning: Error getting active stack: {stack_error}")
            print("Attempting to activate default stack...")
            try:
                # Try mlflow-stack first, then default as a fallback
                try:
                    client.activate_stack("mlflow-stack")
                    print("MLflow stack activated")
                except Exception as e:
                    print(f"Could not activate mlflow-stack, trying default: {e}")
                    client.activate_stack("default")
                    print("Default stack activated")
            except Exception as activate_error:
                print(f"Error activating default stack: {activate_error}")
                print("Continuing anyway, but pipeline execution may fail")
        
        # Create a unique run name
        import uuid
        run_id = str(uuid.uuid4())[:8]
        run_name = f"lora_training_pipeline_run_{run_id}"
        
        print("\n" + "="*80)
        print(f"RUNNING PIPELINE: {run_name}")
        print("="*80)
        
        # Create pipeline instance
        pipeline_instance = lora_training_pipeline()
        print(f"Pipeline instance created: {type(pipeline_instance)}")
        
        # Check ZenML version to determine correct execution method
        try:
            zenml_version = getattr(zenml, "__version__", "unknown")
            print(f"Detailed ZenML version check: {zenml_version}")
            
            # Try to parse version for numeric comparison
            import re
            version_match = re.search(r'(\d+)\.(\d+)', zenml_version)
            major, minor = 0, 0
            if version_match:
                major = int(version_match.group(1))
                minor = int(version_match.group(2)) 
                print(f"Parsed version: major={major}, minor={minor}")
        except Exception as e:
            print(f"Error parsing version: {e}")
            major, minor = 0, 0
            
        # Different methods based on ZenML version
        if hasattr(client, 'run') and callable(client.run):
            # For ZenML 0.74.0 and newer, we use client.run() method
            print("\nExecuting pipeline with client.run()")
            pipeline_run = client.run(
                pipeline=pipeline_instance,
                run_name=run_name
            )
        elif hasattr(pipeline_instance, 'run') and callable(pipeline_instance.run):
            # Fallback for older ZenML versions
            print("\nExecuting pipeline with instance.run()")
            pipeline_run = pipeline_instance.run(
                run_name=run_name
            )
        else:
            # Last resort fallback
            print("\nNone of the expected run methods found, trying with direct execution")
            try:
                # Direct pipeline execution as a last resort
                pipeline_run = lora_training_pipeline()
            except Exception as exec_error:
                print(f"Error with direct execution: {exec_error}")
                raise RuntimeError(f"Unable to find a working method to run the pipeline: {exec_error}")
        
        print(f"\n✅ Pipeline execution started successfully")
        print(f"Run ID: {pipeline_run.id if hasattr(pipeline_run, 'id') else 'unknown'}")
        print(f"Run Name: {run_name}")
        print("\nPipeline will continue to run in the background.")
        print("You can monitor progress on the ZenML dashboard or in logs.")
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION INITIATED")
        print("="*80 + "\n")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to run pipeline: {e}")
        print("\nFor ZenML 0.74.0, please make sure:")
        print("1. ZenML server is running ('zenml up')")
        print("2. ZenML repository is initialized ('zenml init')")
        print("3. A stack is activated ('zenml stack list' and 'zenml stack set default')")
        print("="*80 + "\n")