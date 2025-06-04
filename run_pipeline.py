# LoRA_Training_Pipeline/run_pipeline.py
print("[DEBUG] LoRA Training Pipeline - Module loading started")
import sys
import os
import traceback
import datetime
from pathlib import Path

# Global exception handler to catch any unhandled exceptions
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Handle any unhandled exceptions with comprehensive logging."""
    error_msg = f"""
CRITICAL UNHANDLED EXCEPTION IN RUN_PIPELINE.PY
==============================================
Time: {datetime.datetime.now().isoformat()}
Exception Type: {exc_type.__name__}
Exception Value: {exc_value}
Process ID: {os.getpid()}
Working Directory: {os.getcwd()}
Python Executable: {sys.executable}
Python Version: {sys.version}

FULL TRACEBACK:
{traceback.format_exception(exc_type, exc_value, exc_traceback)}
"""
    
    # Print to stderr for immediate visibility
    print(error_msg, file=sys.stderr)
    
    # Also try to write to error log file
    try:
        error_file = Path("critical_errors.log")
        with open(error_file, "a", encoding="utf-8") as f:
            f.write(error_msg + "\n" + "="*80 + "\n")
        print(f"[CRITICAL] Error logged to {error_file}", file=sys.stderr)
    except Exception as log_err:
        print(f"[CRITICAL] Failed to write error log: {log_err}", file=sys.stderr)
    
    # Call the original exception handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Install the global exception handler
sys.excepthook = global_exception_handler
print("[DEBUG] Global exception handler installed")
print(f"[DEBUG] Basic imports completed - Python {sys.version}")
print(f"[DEBUG] Working directory: {os.getcwd()}")
print(f"[DEBUG] Script path: {__file__ if '__file__' in globals() else 'unknown'}")
print(f"[DEBUG] Python path: {sys.executable}")
print(f"[DEBUG] Platform: {sys.platform}")
print(f"[DEBUG] Current PID: {os.getpid()}")

# Import enhanced process management
print("[DEBUG] Attempting to import enhanced process management...")
try:
    from src.lora_training_pipeline.utils.process_patches import run_pipeline_with_enhanced_process_management
    ENHANCED_PROCESS_MANAGEMENT = True
    print("[DEBUG] Enhanced process management import successful")
    print("✅ Enhanced process management loaded successfully")
except ImportError as imp_err:
    ENHANCED_PROCESS_MANAGEMENT = False
    print(f"[DEBUG] Enhanced process management import failed: {imp_err}")
    print("⚠️ Enhanced process management not available - using standard process management")
except Exception as proc_err:
    ENHANCED_PROCESS_MANAGEMENT = False
    print(f"[DEBUG] Unexpected error loading enhanced process management: {type(proc_err).__name__}: {proc_err}")
    print("⚠️ Enhanced process management not available - using standard process management")

# Import psutil at the module level for process management
print("[DEBUG] Attempting to import psutil...")
try:
    import psutil
    PSUTIL_AVAILABLE = True
    print("[DEBUG] psutil import successful")
    print("✓ Successfully imported psutil for process management")
    # Test basic functionality to ensure it works properly
    current_pid = psutil.Process().pid  # Get current process PID as a simple test
    print(f"[DEBUG] psutil functionality test passed - current PID: {current_pid}")
    print("✓ Verified psutil functionality")
except ImportError as psutil_imp_err:
    PSUTIL_AVAILABLE = False
    print(f"[DEBUG] psutil import failed: {psutil_imp_err}")
    print("CRITICAL ERROR: Failed to import process management utilities: No module named 'psutil'")
    print("ERROR: IMPORT_ERROR - Failed to import process management modules - No module named 'psutil'")
    print("⚠️ WARNING: psutil not installed. Cannot check for stale processes.")
    print("For better crash recovery, install with: uv pip install psutil")
except Exception as e:
    # This could happen if psutil is installed but encounters platform-specific issues
    PSUTIL_AVAILABLE = False
    print(f"[DEBUG] psutil import succeeded but functionality test failed: {type(e).__name__}: {e}")
    print(f"⚠️ WARNING: psutil imported but not working correctly: {e}")
    print("Process management features will be limited")

# Load environment variables from .env file if it exists
try:
    env_path = Path('.env')
    if env_path.exists():
        print(f"Loading environment variables from {env_path}")
        import re
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = re.split(r'=', line, 1)
                    os.environ[key] = value
                    if 'TOKEN' in key or 'SECRET' in key:
                        # Don't print tokens or secrets
                        print(f"  Set {key}=********")
                    else:
                        print(f"  Set {key}={value}")
except Exception as e:
    print(f"Error loading .env file: {e}")

# Add debugging prints to diagnose module loading issues
print("\n" + "="*80)
print("PYTHON ENVIRONMENT DEBUGGING INFORMATION")
print("="*80)

# Override sys.executable with the correct Windows Python path
correct_python_path = r"C:\Users\arnon\Documents\dev\projects\incoming\LoRA_Training_Pipeline\.venv\Scripts\python.exe"
if os.path.exists('/proc/version'):
    # Check if running in WSL
    with open('/proc/version', 'r') as f:
        if 'microsoft' in f.read().lower():
            print(f"Running in WSL, fixing Python path: {sys.executable} -> {correct_python_path}")
            sys.executable = correct_python_path

print(f"Using Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"PATH: {os.environ.get('PATH', 'Not set')}")
print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not set')}")
print(f"sys.path: {sys.path}")

try:
    import site
    print(f"Site packages: {site.getsitepackages()}")
except ImportError:
    print("Could not import site module")

print("\nAttempting to import requests...")
try:
    import requests
    print(f"✓ Requests successfully imported")
    print(f"  Version: {requests.__version__}")
    print(f"  Location: {requests.__file__}")
except ImportError as e:
    print(f"✗ Failed to import requests: {e}")

print("\nAttempting to import pandas...")
try:
    import pandas
    print(f"✓ Pandas successfully imported")
    print(f"  Version: {pandas.__version__}")
    print(f"  Location: {pandas.__file__}")
except ImportError as e:
    print(f"✗ Failed to import pandas: {e}")

print("\nAttempting to import tenacity...")
try:
    import tenacity
    print(f"✓ Tenacity successfully imported")
    try:
        # Try different ways to get version
        if hasattr(tenacity, '__version__'):
            version = tenacity.__version__
        elif hasattr(tenacity, 'version'):
            version = tenacity.version
        else:
            # Try to get version from package metadata
            import pkg_resources
            version = pkg_resources.get_distribution("tenacity").version
    except Exception as ve:
        version = "Unknown (version attribute not found)"
    
    print(f"  Version: {version}")
    print(f"  Location: {tenacity.__file__}")
    print(f"  Dir contents: {dir(tenacity)[:10]}...")  # Show first 10 attributes
except ImportError as e:
    print(f"✗ Failed to import tenacity: {e}")
    print(f"  sys.modules keys: {list(sys.modules.keys())[:20]}...")  # Show first 20 keys
print("="*80 + "\n")

# Import basic Python modules needed for startup
try:
    import subprocess
    import time  # Used throughout script for sleep and timing
    import atexit
    import signal
    import threading
    import queue
    import logging
    import io
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    print("✓ Basic modules imported successfully")
except ImportError as e:
    print(f"ERROR: Missing critical Python dependency: {e}")
    print("Please make sure all dependencies are installed with: uv pip install -e .")
    sys.exit(1)

# Configure logging to avoid stdout conflicts
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_run.log'),
        logging.StreamHandler(io.StringIO())  # Avoid direct console output from logging
    ]
)
logger = logging.getLogger("pipeline")

# Signal handling queue for thread-safe signal processing
signal_queue = queue.Queue()
signal_handler_lock = threading.Lock()

# Global flag to indicate a signal was received - used to minimize operations in signal handler
signal_received = False
# Global flag to indicate immediate shutdown is requested
shutdown_requested = False

# Check for tenacity specifically since it's a common missing dependency
try:
    # Reimport tenacity to ensure it's checked in the same context as the rest of the code
    import tenacity
    
    # Don't try to access __version__ since it might not exist
    print(f"✓ Tenacity check passed - module successfully imported")
except ImportError as e:
    print(f"[DEBUG] Tenacity import failed: {e}")
    print("\n" + "="*80)
    print("ERROR: Missing required dependency: tenacity")
    print("="*80)
    print("This is a common issue. To fix it, run:")
    print("\nuv pip install tenacity~=9.0.0\n")
    print("Or reinstall all dependencies with:")
    print("\nuv pip install -e .\n")
    print("Make sure you've activated your virtual environment first.")
    print("="*80)
    sys.exit(1)

# --- Helper Functions ---

def is_port_in_use(port: int) -> bool:
    """
    Check if a port is in use using socket connection.
    
    Args:
        port: Port number to check
        
    Returns:
        bool: True if port is in use, False otherwise
    """
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)  # 1 second timeout
            result = sock.connect_ex(('localhost', port))
            return result == 0  # 0 means connection successful (port in use)
    except Exception as e:
        print(f"[DEBUG] Error checking port {port}: {e}")
        # If we can't check, assume port is available
        return False

def kill_process_on_port(port: int) -> bool:
    """
    Kill any process using the specified port.
    
    Args:
        port: Port number to clear
        
    Returns:
        bool: True if process was killed or port was free, False if failed
    """
    import socket
    if not PSUTIL_AVAILABLE:
        print(f"[DEBUG] Cannot kill process on port {port} - psutil not available")
        return False
        
    try:
        killed_any = False
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                connections = proc.info['connections'] or []
                for conn in connections:
                    if hasattr(conn, 'laddr') and conn.laddr.port == port:
                        print(f"[DEBUG] Killing process {proc.info['pid']} ({proc.info['name']}) using port {port}")
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            proc.kill()
                        killed_any = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                print(f"[DEBUG] Process access issue while killing port {port}: {type(e).__name__}: {e}")
                continue
            except Exception as e:
                print(f"[DEBUG] Error checking process: {e}")
                continue
        
        if killed_any:
            import time
            time.sleep(1)  # Give time for port to be released
            
        return True
    except Exception as e:
        print(f"[DEBUG] Error killing process on port {port}: {e}")
        return False

def clear_pipeline_ports():
    """
    Clear all ports used by the pipeline by killing any processes using them.
    This prevents port conflicts at startup.
    """
    print("[DEBUG] Clearing pipeline ports to prevent conflicts...")
    
    # Define all ports used by the pipeline
    pipeline_ports = [
        8001,  # FastAPI inference server
        7861,  # Gradio inference UI
        7862,  # Gradio data collection UI  
        7863,  # Dashboard
    ]
    
    ports_cleared = []
    ports_failed = []
    
    for port in pipeline_ports:
        print(f"[DEBUG] Checking port {port}...")
        if is_port_in_use(port):
            print(f"[DEBUG] Port {port} is in use, attempting to clear...")
            if kill_process_on_port(port):
                ports_cleared.append(port)
                print(f"✅ Cleared port {port}")
            else:
                ports_failed.append(port)
                print(f"❌ Failed to clear port {port}")
        else:
            print(f"✅ Port {port} is free")
    
    if ports_cleared:
        print(f"[DEBUG] Successfully cleared ports: {ports_cleared}")
        
    if ports_failed:
        print(f"⚠️ WARNING: Failed to clear ports: {ports_failed}")
        print("These ports may cause conflicts during startup.")
        
        # Show what's using the failed ports
        for port in ports_failed:
            if PSUTIL_AVAILABLE:
                try:
                    for proc in psutil.process_iter(['pid', 'name', 'connections']):
                        try:
                            connections = proc.info['connections'] or []
                            for conn in connections:
                                if hasattr(conn, 'laddr') and conn.laddr.port == port:
                                    print(f"  Port {port} used by: PID {proc.info['pid']} ({proc.info['name']})")
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                            print(f"[DEBUG] Process access issue during port inspection: {type(e).__name__}: {e}")
                            continue
                except Exception as e:
                    print(f"[DEBUG] Error inspecting port usage: {e}")
    
    return len(ports_failed) == 0  # Return True if all ports were cleared successfully

def check_and_ensure_adapter_files():
    print("[DEBUG] check_and_ensure_adapter_files() - FUNCTION ENTRY")
    """
    Check if LoRA adapter files exist in the output/best_model directory.
    If they don't exist, try to copy them from training directory or create placeholders.
    
    Returns:
        bool: True if adapter files are available, False otherwise
    """
    print("[DEBUG] Checking LoRA adapter files...")
    
    # Define paths
    best_model_dir = Path("output/best_model")
    adapter_config_file = best_model_dir / "adapter_config.json"
    adapter_model_file = best_model_dir / "adapter_model.safetensors"
    
    # Ensure output directory exists
    best_model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Best model directory: {best_model_dir}")
    
    # Check if required files exist
    files_exist = {
        "adapter_config.json": adapter_config_file.exists(),
        "adapter_model.safetensors": adapter_model_file.exists()
    }
    
    print(f"[DEBUG] Adapter file status: {files_exist}")
    
    # If files don't exist, try to find them in training output
    missing_files = [name for name, exists in files_exist.items() if not exists]
    
    if missing_files:
        print(f"[DEBUG] Missing adapter files: {missing_files}")
        
        # Look for adapter files in training directory
        training_dir = Path("data/training")
        if training_dir.exists():
            print(f"[DEBUG] Searching for adapter files in: {training_dir}")
            
            # Search for adapter files recursively
            found_files = {}
            for missing_file in missing_files:
                for found_file in training_dir.rglob(missing_file):
                    print(f"[DEBUG] Found {missing_file} at: {found_file}")
                    found_files[missing_file] = found_file
                    break
            
            # Copy found files to best_model directory
            copied_files = []
            for file_name, source_path in found_files.items():
                dest_path = best_model_dir / file_name
                try:
                    import shutil
                    shutil.copy2(source_path, dest_path)
                    print(f"✅ Copied {file_name} from {source_path} to {dest_path}")
                    copied_files.append(file_name)
                except Exception as e:
                    print(f"❌ Failed to copy {file_name}: {e}")
            
            # Update files_exist status
            for file_name in copied_files:
                files_exist[file_name] = True
    
    # Create placeholder files if still missing (for development/testing)
    if not files_exist["adapter_config.json"]:
        print("[DEBUG] Creating placeholder adapter_config.json")
        try:
            placeholder_config = {
                "peft_type": "LORA",
                "auto_mapping": None,
                "base_model_name_or_path": "microsoft/DialoGPT-medium",
                "revision": None,
                "task_type": "CAUSAL_LM",
                "inference_mode": True,
                "r": 16,
                "target_modules": ["c_attn"],
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "bias": "none",
                "modules_to_save": None
            }
            
            with open(adapter_config_file, 'w') as f:
                import json
                json.dump(placeholder_config, f, indent=2)
            
            print(f"✅ Created placeholder: {adapter_config_file}")
            files_exist["adapter_config.json"] = True
            
        except Exception as e:
            print(f"❌ Failed to create placeholder adapter_config.json: {e}")
    
    # Create placeholder weights file if missing
    if not files_exist["adapter_model.safetensors"]:
        print("[DEBUG] Creating placeholder adapter weights file")
        try:
            # Create a minimal placeholder file
            placeholder_path = best_model_dir / "placeholder.txt"
            placeholder_path.write_text("No trained model available yet. Please run training first.")
            print(f"✅ Created model placeholder: {placeholder_path}")
            
        except Exception as e:
            print(f"❌ Failed to create placeholder: {e}")
    
    # Final status check
    adapter_config_exists = adapter_config_file.exists()
    adapter_weights_exist = adapter_model_file.exists() or (best_model_dir / "placeholder.txt").exists()
    
    print(f"[DEBUG] Final adapter status:")
    print(f"  - adapter_config.json: {'✅' if adapter_config_exists else '❌'}")
    print(f"  - adapter weights: {'✅' if adapter_weights_exist else '❌'}")
    
    print(f"[DEBUG] check_and_ensure_adapter_files() - FUNCTION EXIT - Result: {adapter_config_exists}")
    return adapter_config_exists

def check_mlflow_availability():
    """
    Check if MLflow is properly configured and available.
    Returns True if MLflow is available, False otherwise.
    """
    print("[DEBUG] check_mlflow_availability() - FUNCTION ENTRY")
    mlflow_available = False
    debug_info = {
        "mlflow_installed": False,
        "mlflow_version": None,
        "mlflow_path": None,
        "zenml_integration": False,
        "tracking_uri": None,
        "tracking_uri_source": None,
        "errors": []
    }

    print("\n[MLFLOW-DEBUG] Starting MLflow availability check...")

    try:
        # First try to import MLflow
        import mlflow
        print(f"✓ MLflow is installed (version: {mlflow.__version__})")

        # Next, try to check ZenML integration
        try:
            from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
            tracking_uri = get_tracking_uri()
            if tracking_uri:
                print(f"✓ MLflow tracking URI from ZenML: {tracking_uri}")
                # Set the environment variable to ensure other processes can access it
                os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
                mlflow_available = True
        except ImportError as e:
            print(f"⚠️ ZenML MLflow integration not available: {e}")
            # Fall back to environment variable or default
            if "MLFLOW_TRACKING_URI" in os.environ:
                print(f"✓ Using MLFLOW_TRACKING_URI from environment: {os.environ['MLFLOW_TRACKING_URI']}")
                mlflow_available = True
            else:
                # Set a default tracking URI as fallback
                mlruns_dir = os.path.join(os.getcwd(), ".zen", ".mlruns")
                os.makedirs(mlruns_dir, exist_ok=True)
                default_uri = f"file://{mlruns_dir}"
                os.environ["MLFLOW_TRACKING_URI"] = default_uri
                print(f"⚠️ Setting default MLflow tracking URI: {default_uri}")
                mlflow_available = True
    except ImportError:
        print("⚠️ MLflow is not installed, some features may not work correctly")
    except Exception as e:
        print(f"⚠️ Error checking MLflow availability: {e}")

    print(f"[DEBUG] check_mlflow_availability() - FUNCTION EXIT - Result: {mlflow_available}")
    return mlflow_available

# --- Process Storage ---
processes = []

# --- Configuration ---
DATA_COLLECTION_PORT = 7862  # Changed from 7860 to avoid conflicts
INFERENCE_UI_PORT = 7861
FASTAPI_INFERENCE_PORT = 8001
DASHBOARD_PORT = 7863  # Add explicit dashboard port

# Ensure each component has its own URL config
DATA_COLLECTION_API_URL = f"http://localhost:{DATA_COLLECTION_PORT}"
INFERENCE_API_URL = f"http://localhost:{FASTAPI_INFERENCE_PORT}" 
DASHBOARD_API_URL = f"http://localhost:{DASHBOARD_PORT}"

# --- Session ID to track this run ---
import uuid
import datetime
import json

SESSION_ID = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
print(f"Session ID: {SESSION_ID}")

# Make sure file paths work on both Windows and Linux
def get_file_path(name):
    """Creates a platform-independent file path that works on both Windows and Linux."""
    return Path(os.path.join(".", name))

# Create file paths
FIRST_RUN_INDICATOR_FILE = get_file_path(".first_run")  # File to track if it's the first run
MODEL_UPDATE_SIGNAL_FILE = get_file_path(".model_update") # Path to the update signal file
INFERENCE_PROCESS_PID_FILE = get_file_path('inference_process.pid')  # FastAPI server PID
DATA_COLLECTION_PID_FILE = get_file_path('data_collection_ui.pid')  # Data Collection UI PID
INFERENCE_UI_PID_FILE = get_file_path('inference_ui.pid')  # Inference UI PID
ACTIVE_SESSION_FILE = get_file_path('.active_session')  # Track the currently running session
LAST_SESSION_FILE = get_file_path('.last_session')  # Track the last completed session
CRASH_RECOVERY_LOCKFILE = get_file_path('.crash_recovery.lock')  # Lock file for crash recovery

# --- Session Management Functions ---
def write_session_info():
    """
    Writes the current session information to the active session file.
    This helps with recovery if the process crashes unexpectedly.
    """
    print("[DEBUG] write_session_info() - FUNCTION ENTRY")
    print("\n" + "="*80)
    print("SESSION TRACKING: INITIALIZING")
    print("="*80)
    print(f"Creating session tracking for Session ID: {SESSION_ID}")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"[DEBUG] Script process PID: {os.getpid()}")
    
    # Get operating system info for diagnostics
    print("[DEBUG] Importing platform module...")
    import platform
    print(f"[DEBUG] Platform system: {platform.system()}")
    print(f"[DEBUG] Platform version: {platform.version()}")
    
    # Get training info if available
    print("[DEBUG] Loading training cycle information...")
    training_info = {}
    try:
        TRAINING_CYCLE_FILE = Path("./training_cycle_info.json")
        print(f"[DEBUG] Training cycle file path: {TRAINING_CYCLE_FILE.absolute()}")
        if TRAINING_CYCLE_FILE.exists():
            file_size = TRAINING_CYCLE_FILE.stat().st_size
            print(f"[DEBUG] Training cycle file exists, size: {file_size} bytes")
            with open(TRAINING_CYCLE_FILE, 'r') as f:
                training_info = json.load(f)
            print(f"[DEBUG] Loaded training cycle info: {len(training_info)} entries")
            print(f"[DEBUG] Training cycle keys: {list(training_info.keys()) if training_info else 'empty'}")
        else:
            print("[DEBUG] Training cycle file does not exist - will create new session")
    except Exception as e:
        print(f"[DEBUG] Warning: Could not load training cycle info: {e}")
        print(f"[DEBUG] Exception type: {type(e).__name__}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        logger.warning(f"Failed to load training cycle info: {e}")
    
    # Create comprehensive session data
    session_data = {
        "session_id": SESSION_ID,
        "pid": os.getpid(),
        "start_time": datetime.datetime.now().isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "training_cycle": training_info.get("cycle_count", 0)
    }
    
    # Add environment variables that might be relevant (but no secrets)
    env_vars = {}
    for key in ["PYTHONPATH", "PATH", "VIRTUAL_ENV", "STOP_INFERENCE_ON_EXIT"]:
        if key in os.environ:
            env_vars[key] = os.environ.get(key)
    session_data["environment"] = env_vars
    
    try:
        print(f"[DEBUG] Writing session data to: {ACTIVE_SESSION_FILE}")
        print(f"[DEBUG] Session data keys: {list(session_data.keys())}")
        with open(ACTIVE_SESSION_FILE, 'w') as f:
            json.dump(session_data, f, indent=2)
        print(f"✓ Session information written successfully")
        print(f"[DEBUG] Session file size: {ACTIVE_SESSION_FILE.stat().st_size} bytes")
        
        # Print key session info for logging
        print(f"Session info summary:")
        print(f"- Session ID: {SESSION_ID}")
        print(f"- Process ID: {os.getpid()}")
        print(f"- Start time: {session_data['start_time']}")
        print(f"- Python: {platform.python_version()} ({sys.executable})")
        print(f"- System: {platform.system()} {platform.release()}")
        print(f"- Training cycle: {session_data['training_cycle']}")
        print("="*80)
    except Exception as e:
        print(f"⚠️ WARNING: Unable to write session information: {e}")
        import traceback
        print(traceback.format_exc())

def read_session_info(file_path):
    """
    Reads session information from the specified file.
    Returns None if the file doesn't exist or can't be read.
    """
    try:
        if not file_path.exists():
            return None
            
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[DEBUG] Session info read error type: {type(e).__name__}")
        print(f"[DEBUG] Session info read error details: {e}")
        print(f"Warning: Unable to read session information from {file_path}: {e}")
        return None

def check_for_stale_processes():
    """
    Checks for stale processes from previous sessions that may have crashed.
    Attempts to clean them up before starting a new session.
    Preserves active training processes during crash recovery.
    """
    print("[DEBUG] check_for_stale_processes() - FUNCTION ENTRY")
    print("\n" + "="*80)
    print("SESSION RECOVERY: STARTING STALE PROCESS CHECK")
    print("="*80)
    print(f"Current Session ID: {SESSION_ID}")
    print(f"Current Process ID: {os.getpid()}")
    print(f"Recovery lock file: {CRASH_RECOVERY_LOCKFILE}")
    print(f"Active session file: {ACTIVE_SESSION_FILE}")
    print(f"[DEBUG] PSUTIL_AVAILABLE: {PSUTIL_AVAILABLE}")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    
    # Check if psutil is available (imported globally)
    if not PSUTIL_AVAILABLE:
        print("⚠️ WARNING: psutil not installed. Cannot check for stale processes.")
        print("For better crash recovery, install with: uv pip install psutil")
        return
    
    # Gather information about running processes
    process_counts = {
        "total": 0,
        "python": 0,
        "gradio": 0,
        "uvicorn": 0,
        "zenml": 0
    }
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            process_counts["total"] += 1
            if "python" in proc.info['name'].lower():
                process_counts["python"] += 1
    except Exception as e:
        print(f"⚠️ Error getting initial process counts: {e}")
    
    print(f"Process Environment: {process_counts['total']} total processes, {process_counts['python']} Python processes")
        
    # If recovery is already in progress, wait a bit and then continue
    if CRASH_RECOVERY_LOCKFILE.exists():
        lockfile_age = datetime.datetime.now().timestamp() - CRASH_RECOVERY_LOCKFILE.stat().st_mtime
        print(f"Found existing recovery lock file (age: {lockfile_age:.1f} seconds)")
        
        # Check if the lock file is stale (older than 5 minutes)
        if lockfile_age > 300:
            print("⚠️ Lock file is stale (older than 5 minutes). Removing it.")
            CRASH_RECOVERY_LOCKFILE.unlink(missing_ok=True)
        else:
            print("⚠️ Another recovery process is already running. Continuing anyway.")
            print("="*80)
            return
            
    # Create lock file
    try:
        print("Creating recovery lock file...")
        lock_data = {
            "session_id": SESSION_ID,
            "pid": os.getpid(),
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(CRASH_RECOVERY_LOCKFILE, 'w') as f:
            json.dump(lock_data, f, indent=2)
        print("✓ Recovery lock file created")
    except Exception as e:
        print(f"⚠️ WARNING: Unable to create recovery lock file: {e}")
    
    # Load training cycle info to know what to preserve
    training_info = {}
    try:
        TRAINING_CYCLE_FILE = Path("./training_cycle_info.json")
        if TRAINING_CYCLE_FILE.exists():
            with open(TRAINING_CYCLE_FILE, 'r') as f:
                training_info = json.load(f)
            print(f"✓ Loaded training cycle info: Cycle {training_info.get('cycle_count', 0)}, Last run: {training_info.get('last_training_time', 'never')}")
        else:
            print("No training cycle info found")
    except Exception as e:
        print(f"⚠️ Error loading training cycle info: {e}")
        training_info = {}
    
    # Track if we have an active training process
    active_training = False
    active_training_pid = None
    
    try:
        # Read previous session info
        print(f"[DEBUG] Reading session info from {ACTIVE_SESSION_FILE}")
        previous_session = read_session_info(ACTIVE_SESSION_FILE)
        
        if previous_session:
            print(f"[DEBUG] Found previous session data: {len(previous_session)} entries")
            print("\n" + "="*80)
            print("CRASH RECOVERY: PREVIOUS SESSION DETECTED")
            print("="*80)
            print(f"Previous Session ID: {previous_session.get('session_id', 'unknown')}")
            print(f"Previous Session Start: {previous_session.get('start_time', 'unknown')}")
            
            # Check if there was a clean shutdown
            status = previous_session.get('status', 'unknown')
            print(f"Previous Session Status: {status}")
            if status in ['completed', 'clean_shutdown', 'user_interrupted']:
                print("✓ Previous session ended cleanly. No recovery needed.")
                print("="*80)
                # Clean up the session files since they're from a completed session
                try:
                    ACTIVE_SESSION_FILE.unlink(missing_ok=True)
                    print(f"✓ Removed completed session file: {ACTIVE_SESSION_FILE}")
                except Exception as e:
                    print(f"⚠️ Error removing completed session file: {e}")
                return
            
            # Check if the previous main process is still running
            previous_pid = previous_session.get('pid')
            if previous_pid and psutil.pid_exists(previous_pid):
                print(f"⚠️ WARNING: Previous session main process (PID: {previous_pid}) is still running!")
                print("This could indicate a duplicate run. Continuing with caution.")
            else:
                print(f"✓ Previous session main process (PID: {previous_pid}) is not running.")
                
            # Look for ZenML training processes that we might want to preserve
            active_processes = []
            training_processes = []
            inference_processes = []
            data_collection_processes = []
            inference_ui_processes = []
            other_processes = []
            
            print("\nSCANNING FOR PREVIOUS SESSION PROCESSES...")
            for proc in psutil.process_iter(['pid', 'cmdline', 'name', 'create_time', 'environ']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    cmdline_str = ' '.join([str(c) for c in cmdline]) if cmdline else ''
                    proc_name = proc.info.get('name', '')
                    
                    # Get process environment if available
                    try:
                        proc_env = proc.environ()
                    except (psutil.AccessDenied, Exception):
                        proc_env = {}
                    
                    process_type = None
                    
                    if not cmdline_str:
                        continue
                        
                    # First check the PROCESS_NAME environment variable (most reliable)
                    if proc_env and 'PROCESS_NAME' in proc_env:
                        process_type = proc_env['PROCESS_NAME']
                        print(f"  Found process with explicit PROCESS_NAME={process_type}, PID={proc.pid}")
                    
                    # Then analyze command line for specific patterns
                    if not process_type:
                        if 'zenml_pipeline.py' in cmdline_str:
                            process_type = 'ZenMLTrainingPipeline'
                        elif 'gradio_app.py' in cmdline_str:
                            process_type = 'GradioDataCollection'
                        elif 'gradio_inference.py' in cmdline_str:
                            process_type = 'GradioInferenceUI'
                        elif 'uvicorn' in cmdline_str and 'fastapi_inference' in cmdline_str:
                            process_type = 'FastAPIInferenceServer'
                        elif ('python' in proc_name.lower() and
                              ('lora_training_pipeline' in cmdline_str or 
                               'src/lora_training_pipeline' in cmdline_str)):
                            process_type = 'UnknownPipelineProcess'
                    
                    if process_type:
                        proc_age = datetime.datetime.now().timestamp() - proc.info.get('create_time', 0)
                        
                        # Track specific process types
                        if process_type == 'ZenMLTrainingPipeline':
                            # Check if the process was created after the last training time in training_info
                            last_training_time = training_info.get('last_training_time', 0)
                            
                            if isinstance(last_training_time, str):
                                try:
                                    # Convert ISO timestamp to epoch seconds
                                    last_training_time = datetime.datetime.fromisoformat(last_training_time).timestamp()
                                except Exception as timestamp_err:
                                    print(f"[DEBUG] Timestamp conversion error type: {type(timestamp_err).__name__}")
                                    print(f"[DEBUG] Timestamp conversion error details: {timestamp_err}")
                                    print(f"[DEBUG] Invalid timestamp value: {last_training_time}")
                                    last_training_time = 0
                                    
                            is_recent = (proc.info.get('create_time', 0) > last_training_time - 60)  # 1-minute buffer
                            
                            training_processes.append({
                                'pid': proc.pid,
                                'cmd': cmdline_str[:100],
                                'age_seconds': proc_age,
                                'is_recent': is_recent,
                                'type': process_type
                            })
                            
                            # If this process was created around the time of the last training run,
                            # it's likely our active training process that we want to preserve
                            if is_recent:
                                active_training = True
                                active_training_pid = proc.pid
                                print(f"  ✓ Found active training process: PID={proc.pid}, Age={proc_age:.1f}s")
                        
                        # Data Collection UI processes
                        elif process_type == 'GradioDataCollection':
                            data_collection_processes.append({
                                'pid': proc.pid, 
                                'cmd': cmdline_str[:100],
                                'age_seconds': proc_age,
                                'type': process_type
                            })
                            print(f"  Found data collection UI process: PID={proc.pid}, Age={proc_age:.1f}s")
                        
                        # Inference UI processes
                        elif process_type == 'GradioInferenceUI':
                            inference_ui_processes.append({
                                'pid': proc.pid, 
                                'cmd': cmdline_str[:100],
                                'age_seconds': proc_age,
                                'type': process_type
                            })
                            print(f"  Found inference UI process: PID={proc.pid}, Age={proc_age:.1f}s")
                        
                        # FastAPI processes
                        elif process_type == 'FastAPIInferenceServer':
                            inference_processes.append({
                                'pid': proc.pid, 
                                'cmd': cmdline_str[:100],
                                'age_seconds': proc_age,
                                'type': process_type
                            })
                            print(f"  Found FastAPI server process: PID={proc.pid}, Age={proc_age:.1f}s")
                        
                        # Any other relevant processes
                        else:
                            other_processes.append({
                                'pid': proc.pid, 
                                'cmd': cmdline_str[:100],
                                'age_seconds': proc_age,
                                'type': process_type
                            })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Print detailed information about what we found
            print(f"\nProcess Analysis Results:")
            print(f"- Training processes: {len(training_processes)}")
            print(f"- Data Collection UIs: {len(data_collection_processes)}")
            print(f"- Inference UIs: {len(inference_ui_processes)}")
            print(f"- FastAPI processes: {len(inference_processes)}")
            print(f"- Other related processes: {len(other_processes)}")
            
            if active_training:
                print(f"\n🔄 ACTIVE TRAINING DETECTED: PID={active_training_pid}")
                print(f"The training process appears to be active from a previous session.")
                print(f"This process will be preserved to allow training to complete.")
                
                # Print more detailed information about the training process
                if training_info:
                    print(f"Training cycle info:")
                    print(f"- Cycle count: {training_info.get('cycle_count', 0)}")
                    print(f"- Last training time: {training_info.get('last_training_time', 'unknown')}")
                    print(f"- Valid data points: {training_info.get('total_valid_data_points', 0)}")
            
            # Lists to track what to terminate and what to preserve
            processes_to_terminate = []
            processes_to_preserve = []
            
            # Handle excess Data Collection UI processes - keep only the newest one
            if len(data_collection_processes) > 1:
                # Sort by creation time (newest first)
                data_collection_processes.sort(key=lambda p: p['age_seconds'])
                # Keep the newest one
                newest = data_collection_processes[0]
                processes_to_preserve.append((newest['pid'], newest['cmd'], 'Data Collection UI', newest['age_seconds']))
                print(f"\n✅ Preserving newest Data Collection UI: PID={newest['pid']}, Age={newest['age_seconds']:.1f}s")
                # Mark others for termination
                for p in data_collection_processes[1:]:
                    processes_to_terminate.append((p['pid'], p['cmd'], 'Data Collection UI (duplicate)'))
            elif len(data_collection_processes) == 1:
                # If only one, preserve it
                p = data_collection_processes[0]
                processes_to_preserve.append((p['pid'], p['cmd'], 'Data Collection UI', p['age_seconds']))
            
            # Handle excess Inference UI processes - keep only the oldest/first one with health check
            if len(inference_ui_processes) > 1:
                # Sort by creation time (oldest first based on age_seconds - higher age = older)
                inference_ui_processes.sort(key=lambda p: p['age_seconds'], reverse=True)
                
                # Find the oldest UI that is still healthy/responding
                preserved_ui = None
                for ui_proc in inference_ui_processes:
                    # Check if the process is still running and healthy
                    try:
                        proc = psutil.Process(ui_proc['pid'])
                        if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                            # Additional health check - test if process responds to signals
                            try:
                                # Send signal 0 (null signal) to test if process can receive signals
                                proc.send_signal(0)
                                preserved_ui = ui_proc
                                print(f"🔍 Health check PASSED for Inference UI PID {ui_proc['pid']}")
                                break
                            except (psutil.AccessDenied, OSError) as signal_err:
                                print(f"⚠️ Health check FAILED for Inference UI PID {ui_proc['pid']}: {signal_err}")
                                continue
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as proc_err:
                        print(f"⚠️ Process check FAILED for Inference UI PID {ui_proc['pid']}: {proc_err}")
                        continue
                
                if preserved_ui:
                    processes_to_preserve.append((preserved_ui['pid'], preserved_ui['cmd'], 'Inference UI', preserved_ui['age_seconds']))
                    print(f"✅ Preserving oldest healthy Inference UI: PID={preserved_ui['pid']}, Age={preserved_ui['age_seconds']:.1f}s")
                    
                    # Mark others for termination (excluding the preserved one)
                    for p in inference_ui_processes:
                        if p['pid'] != preserved_ui['pid']:
                            processes_to_terminate.append((p['pid'], p['cmd'], 'Inference UI (duplicate)'))
                            print(f"🛑 Marking duplicate Inference UI for termination: PID={p['pid']}, Age={p['age_seconds']:.1f}s")
                else:
                    # If none are healthy, preserve the first one anyway to avoid killing all UIs
                    fallback_ui = inference_ui_processes[0]
                    processes_to_preserve.append((fallback_ui['pid'], fallback_ui['cmd'], 'Inference UI (fallback)', fallback_ui['age_seconds']))
                    print(f"⚠️ No healthy Inference UIs found, preserving first as fallback: PID={fallback_ui['pid']}")
                    
                    # Mark others for termination
                    for p in inference_ui_processes[1:]:
                        processes_to_terminate.append((p['pid'], p['cmd'], 'Inference UI (duplicate)'))
            elif len(inference_ui_processes) == 1:
                # If only one, preserve it
                p = inference_ui_processes[0]
                processes_to_preserve.append((p['pid'], p['cmd'], 'Inference UI', p['age_seconds']))
                print(f"✅ Preserving single Inference UI: PID={p['pid']}, Age={p['age_seconds']:.1f}s")
            else:
                print("ℹ️ No Inference UI processes found - none to preserve or terminate")
            
            # FastAPI processes - CRITICAL: These are dependencies for Inference UI!
            # We must preserve FastAPI processes if we have Inference UI processes
            inference_ui_count = len([p for p in processes_to_preserve if 'Inference UI' in p[2]])
            
            if inference_processes:
                if inference_ui_count > 0:
                    print(f"\n🔗 DEPENDENCY CHECK: Found {inference_ui_count} Inference UIs - preserving ALL FastAPI servers")
                    for p in inference_processes:
                        processes_to_preserve.append((p['pid'], p['cmd'], 'FastAPI Server (UI dependency)', p['age_seconds']))
                        print(f"🔗 Preserving FastAPI dependency: PID={p['pid']}")
                else:
                    print(f"\n🔍 No Inference UIs found - applying normal FastAPI cleanup rules")
                    # Keep only the newest FastAPI process if multiple exist
                    if len(inference_processes) > 1:
                        inference_processes.sort(key=lambda p: p['age_seconds'])  # Newest first (lowest age)
                        newest_fastapi = inference_processes[0]
                        processes_to_preserve.append((newest_fastapi['pid'], newest_fastapi['cmd'], 'FastAPI Server', newest_fastapi['age_seconds']))
                        print(f"✅ Preserving newest FastAPI server: PID={newest_fastapi['pid']}")
                        
                        # Mark others for termination
                        for p in inference_processes[1:]:
                            processes_to_terminate.append((p['pid'], p['cmd'], 'FastAPI Server (duplicate)'))
                    else:
                        # Single FastAPI process, preserve it
                        p = inference_processes[0]
                        processes_to_preserve.append((p['pid'], p['cmd'], 'FastAPI Server', p['age_seconds']))
            else:
                print("\n⚠️ No FastAPI processes found - this may cause Inference UI failures!")
            
            # Other processes - terminate by default
            for p in other_processes:
                processes_to_terminate.append((p['pid'], p['cmd'], 'Other'))
            
            # Handle training processes specially
            for p in training_processes:
                # If it's our active training process, preserve it
                if active_training and p['pid'] == active_training_pid:
                    processes_to_preserve.append((p['pid'], p['cmd'], 'Training', p['age_seconds']))
                else:
                    processes_to_terminate.append((p['pid'], p['cmd'], 'Training (inactive)'))
            
            # Print what we're going to do
            if processes_to_preserve:
                print(f"\n✅ Preserving {len(processes_to_preserve)} processes:")
                for pid, cmd, type_name, age in processes_to_preserve:
                    print(f"  [{type_name}] PID={pid}, Age={age:.1f}s, CMD={cmd[:80]}...")
            
            # SAFEGUARD: Verify we're not about to kill all UIs
            preserved_inference_uis = [p for p in processes_to_preserve if 'Inference UI' in p[2]]
            preserved_data_uis = [p for p in processes_to_preserve if 'Data Collection UI' in p[2]]
            ui_types_to_terminate = [t for _, _, t in processes_to_terminate if 'UI' in t]
            
            print(f"\n🔍 SAFEGUARD CHECK:")
            print(f"  - Preserved Inference UIs: {len(preserved_inference_uis)}")
            print(f"  - Preserved Data Collection UIs: {len(preserved_data_uis)}")
            print(f"  - UI processes to terminate: {len(ui_types_to_terminate)}")
            
            # Check if we're about to terminate ALL UIs of a type
            if len(preserved_inference_uis) == 0 and any('Inference UI' in t for t in ui_types_to_terminate):
                print("⚠️ WARNING: Termination would leave NO Inference UIs running!")
                print("⚠️ Adjusting to preserve at least one Inference UI...")
                
                # Find the first Inference UI in termination list and move it to preservation
                for i, (pid, cmd, type_name) in enumerate(processes_to_terminate):
                    if 'Inference UI' in type_name:
                        # Remove from termination list
                        removed = processes_to_terminate.pop(i)
                        # Add to preservation list
                        processes_to_preserve.append((removed[0], removed[1], 'Inference UI (emergency preserve)', 0))
                        print(f"✅ Emergency preservation of Inference UI: PID={removed[0]}")
                        break
                        
            # Terminate excess processes automatically - without asking
            if processes_to_terminate:
                print(f"\n🛑 Automatically terminating {len(processes_to_terminate)} excess processes:")
                
                # CRITICAL FIX: Delay termination to allow preserved processes to stabilize
                if any('Inference UI' in p[2] for p in processes_to_preserve):
                    print("⏱️ Delaying termination for 3 seconds to allow preserved Inference UIs to stabilize...")
                    time.sleep(3)
                
                for pid, cmd, type_name in processes_to_terminate:
                    try:
                        print(f"  [{type_name}] Terminating: PID={pid}, CMD={cmd[:80]}...")
                        proc = psutil.Process(pid)
                        proc.terminate()
                        
                        # Wait briefly for process to terminate
                        try:
                            proc.wait(timeout=2)
                            print(f"  ✓ Process {pid} terminated successfully")
                        except psutil.TimeoutExpired:
                            # Force kill if termination takes too long
                            print(f"  ⚠️ Process {pid} didn't terminate gracefully, force killing...")
                            proc.kill()
                            try:
                                proc.wait(timeout=1)
                                print(f"  ✓ Process {pid} force-killed successfully")
                            except psutil.TimeoutExpired:
                                print(f"  ❌ Failed to kill process {pid} - may require manual intervention")
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                        print(f"  ⚠️ Error terminating process {pid}: {e}")
            else:
                print("\n✓ No excess processes to terminate")
            
            # IMMEDIATE VERIFICATION: Check if preserved processes are still alive
            print("\n" + "="*80)
            print("🔍 IMMEDIATE POST-CLEANUP VERIFICATION")
            print("="*80)
            
            preserved_inference_count = 0
            preserved_data_count = 0
            
            for pid, cmd, type_name, age in processes_to_preserve:
                try:
                    proc = psutil.Process(pid)
                    if proc.is_running():
                        status = proc.status()
                        print(f"✅ [{type_name}] PID {pid} is ALIVE (status: {status})")
                        
                        if 'Inference UI' in type_name:
                            preserved_inference_count += 1
                        elif 'Data Collection UI' in type_name:
                            preserved_data_count += 1
                    else:
                        print(f"❌ [{type_name}] PID {pid} is NOT RUNNING after preservation!")
                        print(f"🐛 BUG DETECTED: Process was marked for preservation but died!")
                except (psutil.NoSuchProcess, psutil.AccessDenied) as proc_err:
                    print(f"❌ [{type_name}] PID {pid} NOT FOUND after preservation: {proc_err}")
                    print(f"🐛 BUG DETECTED: Process was marked for preservation but disappeared!")
                except Exception as verify_err:
                    print(f"❌ [{type_name}] PID {pid} verification failed: {type(verify_err).__name__}: {verify_err}")
            
            print(f"\n📊 POST-CLEANUP SUMMARY:")
            print(f"   - Inference UIs preserved and alive: {preserved_inference_count}")
            print(f"   - Data Collection UIs preserved and alive: {preserved_data_count}")
            
            if preserved_inference_count == 0 and any('Inference UI' in p[2] for p in processes_to_preserve):
                print("🚨 CRITICAL BUG: Inference UI was preserved but is not alive!")
                print("🔧 This explains why http://localhost:7861/ is not loading")
                
                # Try to find any surviving Gradio inference processes
                print("\n🔍 Searching for any surviving Gradio inference processes...")
                surviving_inference = []
                for proc in psutil.process_iter(['pid', 'cmdline', 'name']):
                    try:
                        if 'python' in proc.name().lower():
                            cmd = ' '.join([str(c) for c in proc.cmdline()]) if proc.cmdline() else ''
                            if 'gradio_inference.py' in cmd:
                                surviving_inference.append((proc.pid, cmd))
                                print(f"🔍 Found surviving inference process: PID {proc.pid}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if not surviving_inference:
                    print("💀 NO SURVIVING INFERENCE UI PROCESSES FOUND")
                    print("🎯 ROOT CAUSE: All Inference UI processes died after cleanup")
            
            print("="*80)
                
            # Clean up PID files if they exist and no active inference server
            if INFERENCE_PROCESS_PID_FILE.exists():
                if len(inference_processes) == 0:
                    try:
                        INFERENCE_PROCESS_PID_FILE.unlink()
                        print(f"✓ Removed stale PID file: {INFERENCE_PROCESS_PID_FILE}")
                    except Exception as e:
                        print(f"⚠️ WARNING: Unable to remove stale PID file: {e}")
                else:
                    print(f"ℹ️ Keeping inference PID file as active processes exist")
            
            # Clean up Data Collection PID files if they exist and no active data collection processes
            if DATA_COLLECTION_PID_FILE.exists():
                if len(data_collection_processes) == 0:
                    try:
                        DATA_COLLECTION_PID_FILE.unlink()
                        print(f"✓ Removed stale PID file: {DATA_COLLECTION_PID_FILE}")
                    except Exception as e:
                        print(f"⚠️ WARNING: Unable to remove stale PID file: {e}")
                else:
                    print(f"ℹ️ Keeping data collection PID file as active processes exist")
            
            # Clean up Inference UI PID files if they exist and no active inference UI processes
            if INFERENCE_UI_PID_FILE.exists():
                if len(inference_ui_processes) == 0:
                    try:
                        INFERENCE_UI_PID_FILE.unlink()
                        print(f"✓ Removed stale PID file: {INFERENCE_UI_PID_FILE}")
                    except Exception as e:
                        print(f"⚠️ WARNING: Unable to remove stale PID file: {e}")
                else:
                    print(f"ℹ️ Keeping inference UI PID file as active processes exist")
            
            print("="*80)
    except Exception as e:
        print(f"❌ ERROR during crash recovery: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        # Always remove the lock file
        try:
            CRASH_RECOVERY_LOCKFILE.unlink(missing_ok=True)
            print(f"✓ Removed recovery lock file")
        except Exception as e:
            print(f"⚠️ WARNING: Unable to remove recovery lock file: {e}")
        
        print("\nRECOVERY PROCESS COMPLETE")
        print("="*80)

def complete_session():
    """
    Marks the current session as completed successfully.
    Moves the active session file to the last session file.
    """
    print("\n" + "="*80)
    print("SESSION TRACKING: COMPLETING SESSION")
    print("="*80)
    
    # Get training info to record final state
    training_info = {}
    try:
        TRAINING_CYCLE_FILE = Path("./training_cycle_info.json")
        if TRAINING_CYCLE_FILE.exists():
            with open(TRAINING_CYCLE_FILE, 'r') as f:
                training_info = json.load(f)
            print(f"Training info at session completion:")
            print(f"- Cycle count: {training_info.get('cycle_count', 0)}")
            print(f"- Last training time: {training_info.get('last_training_time', 'unknown')}")
            print(f"- Valid data points: {training_info.get('total_valid_data_points', 0)}")
    except Exception as e:
        print(f"⚠️ Unable to read training info: {e}")
    
    try:
        if ACTIVE_SESSION_FILE.exists():
            print(f"Reading current session data from: {ACTIVE_SESSION_FILE}")
            # Update the session data with end time
            session_data = read_session_info(ACTIVE_SESSION_FILE)
            if session_data:
                # Calculate duration
                start_time = None
                try:
                    if "start_time" in session_data:
                        start_time = datetime.datetime.fromisoformat(session_data["start_time"])
                except Exception as e:
                    print(f"⚠️ Error parsing start time: {e}")
                
                # Add completion info
                session_data["end_time"] = datetime.datetime.now().isoformat()
                session_data["status"] = "completed"
                
                # Add duration if we have a valid start time
                if start_time:
                    end_time = datetime.datetime.now()
                    duration = end_time - start_time
                    session_data["duration_seconds"] = duration.total_seconds()
                    print(f"Session duration: {duration.total_seconds():.1f} seconds ({duration})")
                
                # Add final training cycle info
                if training_info:
                    session_data["final_training_cycle"] = training_info.get("cycle_count", 0)
                    session_data["final_valid_data_points"] = training_info.get("total_valid_data_points", 0)
                
                print(f"Writing completed session data to: {LAST_SESSION_FILE}")
                # Write updated data to the last session file
                with open(LAST_SESSION_FILE, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                # Remove the active session file
                ACTIVE_SESSION_FILE.unlink()
                print(f"✅ Session {SESSION_ID} marked as completed successfully")
                print(f"Session summary:")
                print(f"- Session ID: {session_data.get('session_id', 'unknown')}")
                print(f"- Start time: {session_data.get('start_time', 'unknown')}")
                print(f"- End time: {session_data.get('end_time', 'unknown')}")
                print(f"- Status: {session_data.get('status', 'unknown')}")
                if "duration_seconds" in session_data:
                    print(f"- Duration: {session_data['duration_seconds']:.1f} seconds")
                print(f"- Initial training cycle: {session_data.get('training_cycle', 0)}")
                print(f"- Final training cycle: {session_data.get('final_training_cycle', 0)}")
                
                # Check if we completed training cycles
                cycles_completed = session_data.get('final_training_cycle', 0) - session_data.get('training_cycle', 0)
                if cycles_completed > 0:
                    print(f"✅ Completed {cycles_completed} training cycle(s) during this session")
                else:
                    print(f"ℹ️ No new training cycles completed during this session")
            else:
                print(f"⚠️ No session data found in {ACTIVE_SESSION_FILE}")
        else:
            print(f"⚠️ No active session file found at {ACTIVE_SESSION_FILE}")
    except Exception as e:
        print(f"⚠️ WARNING: Unable to complete session: {e}")
        import traceback
        print(traceback.format_exc())
    
    print("="*80)

# --- REMOVED SECRETS ---
# os.environ["HUGGING_FACE_HUB_TOKEN"] = "your_hugging_face_token" # Removed
# os.environ["LLAMA3_70B_URL"] = "your_llama_70b_endpoint" # Removed

def terminate_process_group(process):
    """
    Terminate a process and all its children.
    
    Args:
        process: subprocess.Popen object
    """
    if not PSUTIL_AVAILABLE:
        print(f"[DEBUG] psutil not available, using basic termination")
        return False
        
    try:
        import psutil
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        
        print(f"[DEBUG] Terminating process group: parent {process.pid}, {len(children)} children")
        
        # Terminate children first
        for child in children:
            try:
                print(f"[DEBUG] Terminating child process {child.pid}")
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"[DEBUG] Process access issue terminating child {child.pid}: {type(e).__name__}: {e}")
                pass
        
        # Terminate parent
        try:
            parent.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"[DEBUG] Process access issue terminating parent {process.pid}: {type(e).__name__}: {e}")
            pass
        
        # Wait for processes to terminate
        gone, alive = psutil.wait_procs(children + [parent], timeout=3)
        
        # Kill any that didn't terminate
        for proc in alive:
            try:
                print(f"[DEBUG] Force killing process {proc.pid}")
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"[DEBUG] Process access issue force killing {proc.pid}: {type(e).__name__}: {e}")
                pass
        
        return True
        
    except Exception as e:
        print(f"[DEBUG] Error in process group termination: {e}")
        return False

def start_process(command, env=None, shell=False):
    """Starts a subprocess and adds it to the processes list."""
    print(f"[DEBUG] start_process() - FUNCTION ENTRY")
    print(f"[DEBUG] Command: {command}")
    print(f"[DEBUG] Shell mode: {shell}")
    print(f"[DEBUG] Environment provided: {env is not None}")
    if env:
        print(f"[DEBUG] Environment keys: {list(env.keys())}")
    
    global processes
    print(f"[DEBUG] Current processes list length: {len(processes)}")
    
    # Determine process name for better logging
    process_name = None
    if env and "PROCESS_NAME" in env:
        process_name = env["PROCESS_NAME"]
        print(f"[DEBUG] Process name from env: {process_name}")
    else:
        # Try to extract a process name from the command
        cmd_str = str(command)
        print(f"[DEBUG] Analyzing command string: {cmd_str}")
        if 'gradio_app.py' in cmd_str or 'GradioDataCollection' in cmd_str:
            process_name = 'GradioDataCollection'
        elif 'zenml_pipeline.py' in cmd_str or 'ZenMLTrainingPipeline' in cmd_str:
            process_name = 'ZenMLTrainingPipeline'
        elif 'gradio_inference.py' in cmd_str or 'GradioInferenceUI' in cmd_str:
            process_name = 'GradioInferenceUI'
        elif 'fastapi_inference' in cmd_str:
            process_name = 'FastAPIInferenceServer'
        print(f"[DEBUG] Detected process name: {process_name}")
    
    # Make sure we use the correct Python executable
    if isinstance(command, list) and command[0] == "python":
        # Use the corrected sys.executable
        command[0] = sys.executable
        print(f"Using Python executable: {sys.executable}")
    elif shell and command.startswith("python "):
        # Use the corrected sys.executable in shell commands
        command = command.replace("python ", f'"{sys.executable}" ', 1)
        print(f"Using Python executable in shell command: {sys.executable}")
    
    print(f"Starting process: {process_name if process_name else command}")
    
    # If on Windows, log the process start
    if os.name == 'nt':
        try:
            # Import the logging function if possible
            try:
                from src.lora_training_pipeline.utils.helpers import log_process_activity
                
                # Log process before starting
                log_info = {
                    "command": str(command),
                    "shell": shell,
                    "env_vars": str(list(env.keys())) if env else "default"
                }
                
                # Add the process name if we have it
                if process_name:
                    log_info["process_name"] = process_name
                    
                log_process_activity("START", log_info)
            except ImportError:
                # If we can't import, continue anyway but print a warning
                print("⚠️ Warning: Process logging utility not available")
        except Exception as e:
            print(f"⚠️ Warning: Could not log process start: {e}")
    
    try:
        # Clone environment if needed
        if env is None:
            env = os.environ.copy()
        else:
            # Start with current environment and update with provided values
            new_env = os.environ.copy()
            new_env.update(env)
            env = new_env
        
        # Make sure PYTHONPATH is properly set to include our project root and virtual environment
        # This is critical for module resolution in subprocesses
        python_path_parts = []
        
        # Add the project root directory
        root_dir = os.path.dirname(os.path.abspath(__file__))
        python_path_parts.append(root_dir)
        
        # Add the src directory
        src_dir = os.path.join(root_dir, 'src')
        if os.path.exists(src_dir):
            python_path_parts.append(src_dir)
        
        # Add the virtual environment site-packages
        if os.environ.get("VIRTUAL_ENV"):
            if os.name == 'nt':  # Windows
                site_packages = os.path.join(os.environ["VIRTUAL_ENV"], "Lib", "site-packages")
            else:  # Unix/Linux
                site_packages = os.path.join(os.environ["VIRTUAL_ENV"], "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
            
            if os.path.exists(site_packages):
                python_path_parts.append(site_packages)
        
        # Combine paths and set the PYTHONPATH environment variable
        if python_path_parts:
            env["PYTHONPATH"] = os.pathsep.join(python_path_parts)
            print(f"Setting PYTHONPATH for subprocess: {env['PYTHONPATH']}")
            
        # Set process name in environment if we have one
        if process_name and "PROCESS_NAME" not in env:
            env["PROCESS_NAME"] = process_name
        
        # Configure signal handling for child processes
        import signal
        
        def setup_child_signal_handling():
            """Setup proper signal handling in child processes"""
            # Restore default signal handlers in child processes
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        
        try:
            if shell:
                process = subprocess.Popen(
                    command, 
                    env=env, 
                    shell=True,
                    preexec_fn=setup_child_signal_handling if os.name != 'nt' else None
                )
            else:
                process = subprocess.Popen(
                    command, 
                    env=env,
                    preexec_fn=setup_child_signal_handling if os.name != 'nt' else None
                )
            
            print(f"[DEBUG] Started process with PID {process.pid}")
            
        except Exception as e:
            print(f"[DEBUG] Error starting process: {e}")
            raise
            
        # If on Windows, log the successful process start with PID
        if os.name == 'nt':
            try:
                from src.lora_training_pipeline.utils.helpers import log_process_activity
                
                log_info = {
                    "pid": process.pid,
                    "command": str(command),
                    "status": "running"
                }
                
                # Add the process name if we have it
                if process_name:
                    log_info["process_name"] = process_name
                    
                log_process_activity("STARTED", log_info)
            except Exception as e:
                print(f"[DEBUG] Warning: Failed to log process activity: {e}")
                logger.warning(f"Failed to log process activity: {e}")
                
        processes.append(process)
        return process
    except Exception as e:
        print(f"Error starting process: {e}")
        
        # If on Windows, log the process start failure
        if os.name == 'nt':
            try:
                from src.lora_training_pipeline.utils.helpers import log_process_activity
                
                log_info = {
                    "command": str(command),
                    "error": str(e)
                }
                
                # Add the process name if we have it
                if process_name:
                    log_info["process_name"] = process_name
                    
                log_process_activity("START_FAILED", log_info)
            except Exception as e:
                print(f"[DEBUG] Warning: Failed to log process start failure: {e}")
                logger.warning(f"Failed to log process start failure: {e}")
                
        sys.exit(1)  # Exit if a process fails to start

def stop_processes():
    """Stops all running subprocesses."""
    global processes
    print("[DEBUG] stop_processes() called")
    print(f"[DEBUG] Current process list has {len(processes)} entries")
    print("Stopping all processes...")
    
    print("\n" + "="*80)
    print("STOPPING PROCESSES - DEBUG INFO")
    print("="*80)
    print(f"Total processes to check: {len(processes)}")
    
    # Check if psutil is available for better process management
    psutil_available = False
    try:
        import psutil
        psutil_available = True
        print("[DEBUG] psutil available for enhanced process management")
    except ImportError:
        print("[DEBUG] psutil not available, using basic process management")
    except Exception as psutil_err:
        print(f"[DEBUG] psutil import error: {psutil_err}")
    
    # Log each process before attempting to stop
    for i, process in enumerate(processes):
        try:
            pid = process.pid if hasattr(process, 'pid') else 'unknown'
            poll_status = process.poll() if hasattr(process, 'poll') else 'unknown'
            print(f"[DEBUG] Process {i}: PID={pid}, poll_status={poll_status}")
        except Exception as e:
            print(f"[DEBUG] Error inspecting process {i}: {e}")
    
    # Print list of processes for debugging with improved clarity
    fastapi_processes = []
    for i, process in enumerate(processes):
        try:
            is_running = process.poll() is None
            cmd = str(process.args) if hasattr(process, 'args') else "unknown"
            pid = process.pid if hasattr(process, 'pid') else "unknown"
            
            # Identify process type for clearer output
            process_type = "Unknown"
            if "gradio_app.py" in cmd:
                process_type = "Data Collection UI"
            elif "gradio_inference.py" in cmd:
                process_type = "Inference UI"
            elif "fastapi_inference" in cmd and "uvicorn" in cmd:
                process_type = "FastAPI Inference Server"
            elif "zenml_pipeline.py" in cmd:
                process_type = "ZenML Training Pipeline"
            elif "clean_filter.py" in cmd:
                process_type = "Data Cleaning Process"
                
            print(f"Process {i+1}: Type={process_type}, PID={pid}, Running={is_running}, Command={cmd[:50]}...")
            
            # Check if it's the FastAPI inference server
            if is_running and ("uvicorn" in cmd and "fastapi_inference" in cmd):
                fastapi_processes.append((i, process))
        except Exception as e:
            print(f"Error checking process {i}: {e}")
    
    print(f"Found {len(fastapi_processes)} FastAPI inference server processes running")
    
    # Handle FastAPI processes specially
    if fastapi_processes:
        print("IMPORTANT: Will NOT stop FastAPI inference server processes")
        print("These will be handled separately to maintain the inference API")
    
    # Process all non-FastAPI processes
    for process in processes:
        try:
            # Get basic process info with error handling
            try:
                pid = process.pid if hasattr(process, 'pid') else 'unknown'
                poll_status = process.poll() if hasattr(process, 'poll') else None
                args = process.args if hasattr(process, 'args') else []
                cmd = str(args) if args else "unknown"
                print(f"[DEBUG] Checking process PID={pid}, poll_status={poll_status}")
            except Exception as info_err:
                print(f"[DEBUG] Error getting process info: {type(info_err).__name__}: {info_err}")
                continue
            
            # Skip if not running
            if poll_status is not None:
                print(f"[DEBUG] Process {pid} already terminated (poll_status={poll_status})")
                continue
                
            # Check if this is a FastAPI process we want to keep
            if "uvicorn" in cmd and "fastapi_inference" in cmd:
                print(f"[DEBUG] Skipping FastAPI inference server (PID: {pid})")
                continue
                
            # Get process info before terminating
            command = cmd
            
            # Log process termination on Windows
            if os.name == 'nt':
                try:
                    from src.lora_training_pipeline.utils.helpers import log_process_activity
                    log_process_activity("STOP", {
                        "pid": pid,
                        "command": command[:100] if isinstance(command, str) else str(command)[:100],
                        "method": "terminate"
                    })
                except Exception as e:
                    print(f"[DEBUG] Warning: Failed to log process termination: {e}")
                    logger.warning(f"Failed to log process termination: {e}")
            
            # Enhanced process termination with process group handling
            print(f"[DEBUG] Attempting to terminate process {pid} and its children...")
            
            # Try process group termination first (if psutil available)
            try:
                if terminate_process_group(process):
                    print(f"[DEBUG] Process group {pid} terminated successfully")
                    continue
                else:
                    print(f"[DEBUG] Process group termination failed, trying basic termination for {pid}")
            except Exception as group_err:
                print(f"[DEBUG] Error in process group termination: {type(group_err).__name__}: {group_err}")
            
            # Fallback to basic termination
            try:
                print(f"[DEBUG] Using fallback termination for process {pid}")
                
                # First attempt: graceful termination
                process.terminate()
                print(f"[DEBUG] Sent terminate signal to process {pid}")
                
                # Wait with timeout for process to end
                try:
                    process.wait(timeout=3)
                    print(f"[DEBUG] Process {pid} terminated gracefully")
                    continue
                except subprocess.TimeoutExpired:
                    print(f"[DEBUG] Process {pid} did not terminate gracefully within 3 seconds")
                    
                    # Force kill if graceful termination failed
                    try:
                        process.kill()
                        print(f"[DEBUG] Force killed process {pid}")
                        process.wait(timeout=2)
                        print(f"[DEBUG] Process {pid} force terminated successfully")
                    except Exception as kill_err:
                        print(f"[DEBUG] Error force killing process {pid}: {type(kill_err).__name__}: {kill_err}")
                        
            except Exception as term_err:
                print(f"[DEBUG] Error terminating process {pid}: {type(term_err).__name__}: {term_err}")
                # Try to at least kill it
                try:
                    process.kill()
                    print(f"[DEBUG] Force killed process {pid} after termination error")
                except Exception as kill_err:
                    print(f"[DEBUG] Failed to force kill process {pid}: {type(kill_err).__name__}: {kill_err}")
                
                # Log successful termination
                if os.name == 'nt':
                    try:
                        from src.lora_training_pipeline.utils.helpers import log_process_activity
                        log_process_activity("STOPPED", {
                            "pid": pid,
                            "command": command[:100] if isinstance(command, str) else str(command)[:100],
                            "method": "terminate",
                            "status": "success"
                        })
                    except Exception as e:
                        print(f"[DEBUG] Warning: Failed to log process termination success: {e}")
                        logger.warning(f"Failed to log process termination success: {e}")
        except Exception as e:
            print(f"Error stopping process: {e}")
            # Log the error on Windows
            if os.name == 'nt':
                try:
                    pid = process.pid if hasattr(process, 'pid') else "unknown"
                    from src.lora_training_pipeline.utils.helpers import log_process_activity
                    log_process_activity("STOP_ERROR", {
                        "pid": pid,
                        "error": str(e)
                    })
                except Exception as log_e:
                    print(f"[DEBUG] Warning: Failed to log stop error: {log_e}")
                    logger.warning(f"Failed to log stop error: {log_e}")
    
    # Rebuild the processes list to include only FastAPI processes
    updated_processes = []
    for i, proc in enumerate(processes):
        try:
            if proc.poll() is None:  # Still running
                cmd = str(proc.args) if hasattr(proc, 'args') else ""
                if "uvicorn" in cmd and "fastapi_inference" in cmd:
                    updated_processes.append(proc)
                    print(f"Keeping FastAPI process (PID: {proc.pid})")
        except Exception as e:
            print(f"[DEBUG] Warning: Error checking process for FastAPI: {e}")
            logger.warning(f"Error checking process for FastAPI: {e}")
    
    processes = updated_processes
    remaining = len(processes)
    if remaining > 0:
        print(f"Processes remaining after cleanup: {remaining}")
        print("The following processes are still running:")
        for i, process in enumerate(processes):
            try:
                cmd = str(process.args) if hasattr(process, 'args') else "unknown"
                pid = process.pid if hasattr(process, 'pid') else "unknown"
                
                # Identify process type for clearer output
                process_type = "Unknown"
                if "gradio_app.py" in cmd:
                    process_type = "Data Collection UI"
                elif "gradio_inference.py" in cmd:
                    process_type = "Inference UI"
                elif "fastapi_inference" in cmd and "uvicorn" in cmd:
                    process_type = "FastAPI Inference Server"
                elif "zenml_pipeline.py" in cmd:
                    process_type = "ZenML Training Pipeline"
                elif "clean_filter.py" in cmd:
                    process_type = "Data Cleaning Process"
                    
                print(f"  - {process_type} (PID={pid})")
            except Exception as proc_detail_err:
                print(f"[DEBUG] Error obtaining process details: {type(proc_detail_err).__name__}: {proc_detail_err}")
                print(f"  - Unknown process (Error obtaining details)")
    else:
        print("All processes have been stopped")
    print("="*80)

def get_fastapi_pids():
    """
    Gets the PIDs of all running FastAPI inference server processes.
    Returns a list of PIDs or an empty list if none are found.
    """
    fastapi_pids = []
    
    # Only attempt to find processes if psutil is available
    if not PSUTIL_AVAILABLE:
        print("Cannot get FastAPI processes: psutil not available")
        return fastapi_pids
        
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                cmdline_str = ' '.join([str(c) for c in cmdline]) if cmdline else ''
                
                # Check if this is a FastAPI inference server process
                if ('fastapi_inference:app' in cmdline_str or 
                    'uvicorn' in cmdline_str and 'fastapi_inference' in cmdline_str):
                    fastapi_pids.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        print(f"Error searching for FastAPI processes: {e}")
    
    return fastapi_pids

def stop_inference_server():
    """Stops the FastAPI inference server."""
    # First, try to find and stop the inference server process by looking through our processes list
    global processes
    inference_server_stopped = False
    
    # Add stack trace for debugging to see who's calling this function
    import traceback
    stack_trace = traceback.format_stack()
    print("\n" + "="*80)
    print("STOP_INFERENCE_SERVER CALLED - DEBUG STACK TRACE")
    print("="*80)
    print("Call stack (most recent last):")
    for i, frame in enumerate(stack_trace[:-1]):  # Skip the last frame which is this function
        print(f"Frame {i+1}:\n{frame.strip()}")
    print("="*80)
    
    # Print detailed debugging info
    print("\n" + "="*80)
    print("STOPPING INFERENCE SERVER - DEBUG INFO")
    print("="*80)
    print(f"Total processes tracked: {len(processes)}")
    print(f"PID file exists: {INFERENCE_PROCESS_PID_FILE.exists()}")
    
    # Log all running processes for debugging
    running_count = 0
    for i, proc in enumerate(processes):
        if proc.poll() is None:  # Process is running
            running_count += 1
            cmd = str(proc.args) if hasattr(proc, 'args') else "Unknown"
            print(f"Process {i}: PID={proc.pid}, Running={proc.poll() is None}, Command={cmd[:100]}...")

    print(f"Total running processes: {running_count}")
    print("="*80)
    
    # Look for FastAPI inference server in our processes list
    for process in processes:
        try:
            if process.poll() is None:  # Process is running
                process_cmd = str(process.args) if hasattr(process, 'args') else ""
                if ("FastAPIInferenceServer" in process_cmd or 
                    "uvicorn" in process_cmd and "fastapi_inference" in process_cmd):
                    print(f"Found FastAPI inference server process (PID: {process.pid}), stopping...")
                    
                    # Log that we're stopping the server on Windows
                    if os.name == 'nt':
                        try:
                            from src.lora_training_pipeline.utils.helpers import log_process_activity
                            log_process_activity("STOP_INFERENCE", {
                                "pid": process.pid,
                                "type": "fastapi_inference_server"
                            })
                        except Exception as log_err:
                            print(f"[DEBUG] Process activity logging error type: {type(log_err).__name__}")
                            print(f"[DEBUG] Process activity logging error details: {log_err}")
                            print(f"[DEBUG] Failed to log STOP_INFERENCE for PID: {process.pid}")
                    
                    # Terminate the process
                    process.terminate()
                    try:
                        # Wait with timeout
                        process.wait(timeout=5)
                        inference_server_stopped = True
                        print(f"FastAPI inference server stopped (PID: {process.pid})")
                    except subprocess.TimeoutExpired:
                        # Force kill if normal termination fails
                        if os.name == 'nt':
                            subprocess.run(['taskkill', '/F', '/PID', str(process.pid)], check=False)
                        else:
                            process.kill()
                        inference_server_stopped = True
                        print(f"FastAPI inference server force-killed (PID: {process.pid})")
        except Exception as e:
            print(f"Error checking process: {e}")
    
    # Now also check for the PID file as a backup method
    if INFERENCE_PROCESS_PID_FILE.exists():
        try:
            pid = int(INFERENCE_PROCESS_PID_FILE.read_text().strip())
            print(f"Found inference server PID file, stopping server (PID: {pid})...")
            
            if os.name == 'nt':  # Windows
                # Log that we're attempting to stop the inference server
                try:
                    from src.lora_training_pipeline.utils.helpers import log_process_activity
                    log_process_activity("STOP_INFERENCE", {
                        "pid": pid,
                        "type": "inference_server"
                    })
                except Exception as log_err:
                    print(f"[DEBUG] Windows process activity logging error type: {type(log_err).__name__}")
                    print(f"[DEBUG] Windows process activity logging error details: {log_err}")
                    print(f"[DEBUG] Failed to log STOP_INFERENCE for Windows PID: {pid}")
                    
                # On Windows, use taskkill for more reliable process termination
                try:
                    result = subprocess.run(['taskkill', '/F', '/PID', str(pid)], 
                                          check=False, capture_output=True)
                                          
                    # Log the result
                    try:
                        from src.lora_training_pipeline.utils.helpers import log_process_activity
                        log_process_activity("STOPPED_INFERENCE", {
                            "pid": pid,
                            "success": result.returncode == 0,
                            "return_code": result.returncode,
                            "output": result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
                        })
                    except Exception as log_err:
                        print(f"[DEBUG] Error logging stopped inference process: {type(log_err).__name__}: {log_err}")
                        pass
                        
                except Exception as e:
                    print(f"Error running taskkill: {e}")
                    
                    # Log the error
                    try:
                        from src.lora_training_pipeline.utils.helpers import log_process_activity
                        log_process_activity("STOP_INFERENCE_ERROR", {
                            "pid": pid,
                            "error": str(e)
                        })
                    except Exception as log_err2:
                        print(f"[DEBUG] Error logging inference stop error: {type(log_err2).__name__}: {log_err2}")
                        pass
            else:  # Linux/macOS
                # Use SIGTERM signal on Unix-like systems
                try:
                    os.kill(pid, signal.SIGTERM)
                    inference_server_stopped = True
                except ProcessLookupError:
                    # Process doesn't exist
                    print(f"[DEBUG] Process {pid} not found when trying to send SIGTERM (already terminated)")
                    pass
                
            time.sleep(2) # Allow time for shutdown
            
            # Now remove the PID file to allow a new server to start
            if INFERENCE_PROCESS_PID_FILE.exists():
                try:
                    INFERENCE_PROCESS_PID_FILE.unlink() # Remove the PID file
                    print(f"Removed PID file: {INFERENCE_PROCESS_PID_FILE}")
                except Exception as e:
                    print(f"Error removing PID file: {e}")
            
            if not inference_server_stopped:
                print("Inference server stopped via PID file.")
                inference_server_stopped = True

        except ProcessLookupError:
            print("Inference server process not found.")
            # Clean up if process doesn't exist
            if INFERENCE_PROCESS_PID_FILE.exists():
                try:
                    INFERENCE_PROCESS_PID_FILE.unlink()
                    print(f"Removed stale PID file: {INFERENCE_PROCESS_PID_FILE}")
                except Exception as unlink_err:
                    print(f"[DEBUG] PID file unlink error type: {type(unlink_err).__name__}")
                    print(f"[DEBUG] PID file unlink error details: {unlink_err}")
                    print(f"[DEBUG] Failed to remove PID file: {INFERENCE_PROCESS_PID_FILE}")
            
            # Log the error on Windows
            if os.name == 'nt':
                try:
                    from src.lora_training_pipeline.utils.helpers import log_process_activity
                    log_process_activity("STOP_INFERENCE_NOT_FOUND", {
                        "pid": pid if 'pid' in locals() else "unknown"
                    })
                except Exception as log_err:
                    print(f"[DEBUG] STOP_INFERENCE_NOT_FOUND logging error type: {type(log_err).__name__}")
                    print(f"[DEBUG] STOP_INFERENCE_NOT_FOUND logging error details: {log_err}")
                    print(f"[DEBUG] Failed to log STOP_INFERENCE_NOT_FOUND event")
        except Exception as e:
            print(f"Error stopping inference server from PID file: {e}")
            
            # Log the error on Windows
            if os.name == 'nt':
                try:
                    from src.lora_training_pipeline.utils.helpers import log_process_activity
                    log_process_activity("STOP_INFERENCE_ERROR", {
                        "pid": pid if 'pid' in locals() else "unknown",
                        "error": str(e)
                    })
                except Exception as log_err:
                    print(f"[DEBUG] STOP_INFERENCE_ERROR logging error type: {type(log_err).__name__}")
                    print(f"[DEBUG] STOP_INFERENCE_ERROR logging error details: {log_err}")
                    print(f"[DEBUG] Failed to log STOP_INFERENCE_ERROR event for error: {str(e)}")
    else:
        print("No inference server PID file found.")
    
    if not inference_server_stopped:
        print("No running inference server found.")
    else:
        # Ensure the PID file is removed so we can start a new server
        if INFERENCE_PROCESS_PID_FILE.exists():
            try:
                INFERENCE_PROCESS_PID_FILE.unlink()
                print("Removed PID file to allow new server to start")
            except Exception as e:
                print(f"Error removing PID file: {e}")
    
    print("="*80)

# Function to clean up batch files
# Function removed as we're no longer using batch files

# --- Signal Handling ---
# Non-blocking signal handler to avoid reentrant call issues
def non_blocking_signal_handler(sig, frame):
    """
    This is a truly non-blocking signal handler that only places the signal in a queue
    without performing any I/O operations to avoid reentrant call issues.
    """
    try:
        # Queue the signal and additional info for later processing
        # DO NOT perform any I/O operations (no print, no logger, no stderr/stdout)
        signal_data = {
            "signal": sig,
            "frame": frame,
            "time": time.time(),
            "pid": os.getpid()
        }
        signal_queue.put(signal_data)
        
        # Set a global flag to indicate a signal was received
        # This is a simple atomic operation that won't cause reentrant issues
        global signal_received
        signal_received = True
        
    except Exception as e:
        # Write to stderr only - safer than other I/O in signal handler
        import sys
        sys.stderr.write(f"[DEBUG] Signal handler error: {e}\n")
        sys.stderr.flush()

# The actual signal processing function that runs in the main thread
def process_signal():
    """
    Process any signals in the queue. This should be called from the main thread
    in a safe context where stdout/stderr are not being written to.
    """
    global signal_received
    
    # Quick check using the flag to avoid unnecessary queue operations
    if not signal_received and signal_queue.empty():
        return False
    
    print("[DEBUG] process_signal() called - signal detected")
    print(f"[DEBUG] Signal queue size: {signal_queue.qsize()}")
    print(f"[DEBUG] Global signal_received flag: {signal_received}")

    # Get the signal from the queue
    try:
        signal_data = signal_queue.get_nowait()
        print(f"[DEBUG] Retrieved signal data from queue: {signal_data}")
        
        # Reset flag if queue is now empty
        if signal_queue.empty():
            signal_received = False
            print("[DEBUG] Signal queue now empty, reset signal_received flag")
            
    except queue.Empty:
        # Queue is empty, reset the flag
        signal_received = False
        return False

    # Extract signal information
    sig = signal_data.get("signal")
    
    # Get signal name for better logs
    signal_name = "UNKNOWN"
    if sig == signal.SIGINT:
        signal_name = "SIGINT (Ctrl+C)"
    elif sig == signal.SIGTERM:
        signal_name = "SIGTERM"

    # Acquire the lock to ensure we don't have multiple signal handlers running
    with signal_handler_lock:
        # Set global shutdown flag immediately
        global shutdown_requested
        shutdown_requested = True
        
        # Now we can safely do logging and I/O since we're in the main thread
        logger.info(f"Processing signal: {signal_name}")

        # Safe stdout writes with clear separation to avoid conflicts
        sys.stdout.write("\n\n")
        sys.stdout.flush()
        sys.stdout.write("="*80 + "\n")
        sys.stdout.write(f"SIGNAL RECEIVED: {signal_name} - Starting graceful shutdown\n")
        sys.stdout.write("="*80 + "\n\n")
        sys.stdout.flush()
        
        # Start graceful shutdown
        try:
            # Set up immediate exit for subsequent signals
            signal.signal(signal.SIGINT, immediate_exit_handler)
            signal.signal(signal.SIGTERM, immediate_exit_handler)
            
            # Get termination timestamp from signal data or use current time
            signal_time = signal_data.get("time", time.time())
            current_time = time.time()
            elapsed = current_time - signal_time
            
            # If signal was sent more than 3 seconds ago, use forced shutdown
            # This helps users who press Ctrl+C multiple times to force quit
            force_shutdown = elapsed > 3 or sig == signal.SIGTERM
            
            if force_shutdown:
                sys.stdout.write("Using forced shutdown due to timeout or SIGTERM.\n")
                sys.stdout.flush()
                
            # Call the appropriate shutdown functions to clean up
            print(f"[DEBUG] Calling shutdown functions, force={force_shutdown}")
            try:
                # Use stop_processes function (not stop_services)
                print("[DEBUG] Calling stop_processes()...")
                stop_processes()
                print("[DEBUG] stop_processes() completed successfully")
            except Exception as stop_err:
                print(f"[DEBUG] Error in stop_processes(): {type(stop_err).__name__}: {stop_err}")
                # Try to stop inference server specifically
                try:
                    print("[DEBUG] Trying stop_inference_server() as fallback...")
                    stop_inference_server()
                    print("[DEBUG] stop_inference_server() completed successfully")
                except Exception as inference_err:
                    print(f"[DEBUG] Error in stop_inference_server(): {type(inference_err).__name__}: {inference_err}")
                    print("[DEBUG] Manual cleanup of processes...")
                    # Manual cleanup as last resort
                    try:
                        import psutil
                        current_pid = os.getpid()
                        parent = psutil.Process(current_pid)
                        children = parent.children(recursive=True)
                        print(f"[DEBUG] Found {len(children)} child processes to terminate")
                        for child in children:
                            try:
                                print(f"[DEBUG] Terminating child process PID {child.pid}")
                                child.terminate()
                            except Exception as child_err:
                                print(f"[DEBUG] Error terminating child {child.pid}: {child_err}")
                    except Exception as manual_err:
                        print(f"[DEBUG] Manual cleanup failed: {manual_err}")
            
            # Allow some time for the cleanup to complete
            sys.stdout.write("Shutdown completed. Exiting...\n")
            sys.stdout.flush()
            
            # Exit gracefully rather than forcing immediate termination
            print("[DEBUG] Signal processing complete - calling sys.exit(0)")
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(0)
        except Exception as e:
            # If something goes wrong during shutdown, log and exit anyway
            logger.error(f"Error during signal shutdown: {e}")
            sys.stderr.write(f"Error during shutdown: {e}\n")
            sys.stderr.flush()
            
            # Still exit
            os._exit(1)

        # Print warning about inference server
        sys.stdout.write("NOTE: The inference server will be stopped. If you need it to continue\n")
        sys.stdout.write("running, start it manually with: \n")
        sys.stdout.write("python -m uvicorn src.lora_training_pipeline.inference.fastapi_inference:app --reload --port 8001\n")
        sys.stdout.flush()

        # Mark this as a clean shutdown in the session info
        try:
            # Update the session data with shutdown info
            if ACTIVE_SESSION_FILE.exists():
                session_data = read_session_info(ACTIVE_SESSION_FILE)
                if session_data:
                    session_data["end_time"] = datetime.datetime.now().isoformat()
                    session_data["status"] = "clean_shutdown"
                    session_data["shutdown_signal"] = signal_name

                    # Write updated data to the session file
                    with open(ACTIVE_SESSION_FILE, 'w') as f:
                        json.dump(session_data, f, indent=2)

                    # Copy to last session file as well
                    with open(LAST_SESSION_FILE, 'w') as f:
                        json.dump(session_data, f, indent=2)

                    logger.info(f"Session {SESSION_ID} marked as cleanly shut down")
                    sys.stdout.write(f"Session {SESSION_ID} marked as cleanly shut down\n")
                    sys.stdout.flush()
        except Exception as e:
            logger.error(f"Unable to update session shutdown info: {e}")
            sys.stderr.write(f"Warning: Unable to update session shutdown info: {e}\n")
            sys.stderr.flush()

        # Stop all processes except the inference server
        stop_processes()

        # Notify user about the FastAPI server status
        fastapi_pids = get_fastapi_pids()
        if fastapi_pids:
            pid_str = ", ".join(str(pid) for pid in fastapi_pids)
            sys.stdout.write("\n" + "="*80 + "\n")
            sys.stdout.write(f"⚠️ FastAPI inference server (PIDs: {pid_str}) is still running\n")
            sys.stdout.write(f"ℹ️ This allows model inference to continue working even after the pipeline script ends\n")
            sys.stdout.write(f"ℹ️ To manually terminate the server, use your system's process manager\n")
            sys.stdout.write(f"ℹ️ On Windows: taskkill /F /PID {pid_str}\n")
            sys.stdout.write(f"ℹ️ On Linux/Mac: kill {pid_str}\n")
            sys.stdout.write("="*80 + "\n\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("No FastAPI inference server detected\n")
            sys.stdout.flush()

        # Final cleanup of session files
        try:
            if ACTIVE_SESSION_FILE.exists():
                ACTIVE_SESSION_FILE.unlink()
        except Exception as cleanup_err:
            print(f"[DEBUG] Error cleaning up session file: {type(cleanup_err).__name__}: {cleanup_err}")
            pass

        sys.stdout.write("All processes stopped. Exiting.\n")
        sys.stdout.flush()

        # Give a little time for all output to be flushed
        time.sleep(0.2)
        exit(0)

    return True  # Signal was processed

# Simple fallback signal handler for immediate response
def immediate_exit_handler(sig, frame):
    """Simple signal handler that exits immediately on second Ctrl+C"""
    global shutdown_requested
    shutdown_requested = True
    print("\nForced shutdown requested. Exiting immediately...")
    os._exit(1)

# Register the non-blocking signal handler
signal.signal(signal.SIGINT, non_blocking_signal_handler)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, non_blocking_signal_handler)  # Handle termination signals

# Start a background thread to process signals periodically to avoid reentrant issues
def signal_processing_thread():
    """Background thread that periodically checks for and processes queued signals."""
    while True:
        try:
            time.sleep(0.5)  # Check every 500ms
            # If there's a signal to process, the main program will handle it
            # We don't process signals in this thread to avoid threading issues
        except Exception as e:
            logger.error(f"Error in signal processing thread: {e}")

# Start the signal processing thread
signal_thread = threading.Thread(target=signal_processing_thread, daemon=True)
signal_thread.start()

# --- atexit Registration ---
# Explicitly set STOP_INFERENCE_ON_EXIT to "false" to ensure it's preserved
os.environ["STOP_INFERENCE_ON_EXIT"] = "false"

# Register the process stopping but NOT the inference server
atexit.register(stop_processes)

# Define a safer version of the stop_inference_server function for atexit
def stop_inference_server_if_requested():
    """Only stop the inference server if specifically requested by setting an environment variable."""
    if os.environ.get("STOP_INFERENCE_ON_EXIT", "false").lower() == "true":
        print("Environment variable STOP_INFERENCE_ON_EXIT is set to true, stopping inference server...")
        stop_inference_server()
    else:
        print("Preserving FastAPI inference server on exit (set STOP_INFERENCE_ON_EXIT=true to change this behavior)")

# Register the safer version
atexit.register(stop_inference_server_if_requested)

def scan_data_files(progress_tracker=None):
    """
    Scans the data directory and subdirectories to count various types of parquet files,
    updates the checkpoint file, and synchronizes counters.
    
    Args:
        progress_tracker: Optional ProgressTracker instance to use instead of creating a new one
    """
    print("[DEBUG] scan_data_files() - FUNCTION ENTRY")
    print(f"[DEBUG] Progress tracker provided: {progress_tracker is not None}")
    print("\n" + "="*80)
    print("SCANNING DATA DIRECTORY FOR PARQUET FILES")
    print("="*80)
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"[DEBUG] Checking data directory existence...")
    
    # Use provided progress tracker or create a simple version
    if progress_tracker:
        scan_progress = progress_tracker
        scan_progress.update(message="Scanning data directory")
    else:
        # Create a simple version for tracking
        class SimpleTracker:
            def __init__(self, name):
                self.name = name
                print(f"Starting {name}...")
                
            def start(self, message):
                print(f"• {message}")
                
            def update(self, message):
                print(f"• {message}")
                
            def complete(self, message):
                print(f"✅ {message}")
                
        scan_progress = SimpleTracker("Data Scan")
        scan_progress.start("Scanning data directory")
    
    try:
        from src.lora_training_pipeline.config import DATASET_NAME, DATA_VERSION
        from pathlib import Path
        import pandas as pd
        import json
        
        # Define paths
        DATA_DIR = Path("./data")
        VALID_DIR = DATA_DIR / "valid"
        REJECTED_DIR = DATA_DIR / "rejected"
        TRAINING_DIR = DATA_DIR / "training"
        CHECKPOINT_FILE = DATA_DIR / "validation_checkpoint.json"
        
        # Create directories if they don't exist
        DATA_DIR.mkdir(exist_ok=True)
        VALID_DIR.mkdir(exist_ok=True, parents=True)
        REJECTED_DIR.mkdir(exist_ok=True, parents=True)
        TRAINING_DIR.mkdir(exist_ok=True, parents=True)
        
        scan_progress.update(message="Counting original files")
        
        # Count original files
        original_files = list(DATA_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_original_*.parquet"))
        original_count = len(original_files)
        print(f"📊 Found {original_count} original data files")
        
        # Count valid files (both in data dir and valid subdir)
        scan_progress.update(message="Counting valid files")
        valid_files_main = list(DATA_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_valid_*.parquet"))
        valid_files_subdir = list(VALID_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_valid_*.parquet"))
        valid_files = valid_files_main + valid_files_subdir
        valid_count = len(valid_files)
        print(f"📊 Found {valid_count} valid data files")
        
        # Count invalid files
        scan_progress.update(message="Counting invalid files")
        invalid_files_main = list(DATA_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_invalid_*.parquet"))
        invalid_files_subdir = list(REJECTED_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_invalid_*.parquet"))
        invalid_files = invalid_files_main + invalid_files_subdir
        invalid_count = len(invalid_files)
        print(f"📊 Found {invalid_count} invalid data files")
        
        # Count consolidated files
        scan_progress.update(message="Counting consolidated files")
        consolidated_files = list(DATA_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_consolidated_*.parquet"))
        consolidated_count = len(consolidated_files)
        print(f"📊 Found {consolidated_count} consolidated data files")
        
        # Count training files
        scan_progress.update(message="Counting training files")
        training_files = list(TRAINING_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_training_*.parquet"))
        training_count = len(training_files)
        print(f"📊 Found {training_count} training data files")
        
        # Count actual data points
        scan_progress.update(message="Counting actual data points")
        
        # For original files
        total_original_points = 0
        try:
            for file in original_files[:10]:  # Sample first 10 files to estimate
                try:
                    df = pd.read_parquet(file)
                    total_original_points += len(df)
                except Exception as e:
                    print(f"⚠️ Warning: Could not read file {file.name}: {e}")
            
            # If we have more than 10 files, estimate the total
            if len(original_files) > 10:
                avg_points_per_file = total_original_points / min(10, len(original_files))
                total_original_points = int(avg_points_per_file * len(original_files))
                print(f"📊 Estimated {total_original_points} original data points (based on sampling)")
            else:
                print(f"📊 Found {total_original_points} original data points")
        except Exception as e:
            print(f"⚠️ Warning: Error estimating original data points: {e}")
        
        # Update or create checkpoint file
        scan_progress.update(message="Updating checkpoint file")
        
        # Default checkpoint structure
        default_checkpoint = {
            "processed_files": [],
            "last_processed_timestamp": "",
            "valid_files": [],
            "invalid_files": [],
            "checkpoint_version": 1,
            "last_reprocessed": 0
        }
        
        # Try to read existing checkpoint
        checkpoint = default_checkpoint
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE, 'r') as f:
                    checkpoint = json.load(f)
                print(f"📊 Loaded existing checkpoint file")
                
                # Make sure all fields exist (backward compatibility)
                for key, value in default_checkpoint.items():
                    if key not in checkpoint:
                        checkpoint[key] = value
            except Exception as e:
                print(f"[DEBUG] Checkpoint file read error type: {type(e).__name__}")
                print(f"[DEBUG] Checkpoint file read error details: {e}")
                import traceback
                print(f"[DEBUG] Checkpoint file read traceback: {traceback.format_exc()}")
                print(f"⚠️ Warning: Could not read checkpoint file: {e}")
                checkpoint = default_checkpoint
        
        # Compare file lists with checkpoint
        original_file_names = [f.name for f in original_files]
        valid_file_names = [f.name for f in valid_files]
        invalid_file_names = [f.name for f in invalid_files]
        
        # Detect differences
        processed_files_set = set(checkpoint["processed_files"])
        valid_files_set = set(checkpoint["valid_files"])
        invalid_files_set = set(checkpoint["invalid_files"])
        
        # Files that are possibly missing from the checkpoint
        original_not_in_processed = set(original_file_names) - processed_files_set
        valid_not_in_checkpoint = set(valid_file_names) - valid_files_set
        invalid_not_in_checkpoint = set(invalid_file_names) - invalid_files_set
        
        if original_not_in_processed:
            print(f"⚠️ Warning: Found {len(original_not_in_processed)} original files not marked as processed")
            # We don't update processed_files automatically as that could mess up the cleaning logic
            
        if valid_not_in_checkpoint:
            print(f"⚠️ Warning: Found {len(valid_not_in_checkpoint)} valid files not in checkpoint")
            # Update valid files in checkpoint - make sure it's a list, not a set
            valid_files_combined = list(valid_files_set.union(valid_not_in_checkpoint))
            checkpoint["valid_files"] = valid_files_combined
            print(f"✅ Updated valid_files in checkpoint (now {len(valid_files_combined)} files)")
            
        if invalid_not_in_checkpoint:
            print(f"⚠️ Warning: Found {len(invalid_not_in_checkpoint)} invalid files not in checkpoint")
            # Update invalid files in checkpoint - make sure it's a list, not a set
            invalid_files_combined = list(invalid_files_set.union(invalid_not_in_checkpoint))
            checkpoint["invalid_files"] = invalid_files_combined
            print(f"✅ Updated invalid_files in checkpoint (now {len(invalid_files_combined)} files)")
        
        # Save updated checkpoint
        try:
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f"✅ Saved updated checkpoint file")
        except Exception as e:
            print(f"⚠️ Warning: Could not save checkpoint file: {e}")
        
        # Update training cycle info
        scan_progress.update(message="Updating training cycle info")
        
        # Check and update training cycle file
        TRAINING_CYCLE_FILE = Path("./training_cycle_info.json")
        
        # Default training cycle info
        default_cycle_info = {
            "cycle_count": 0,
            "total_valid_data_points": 0,
            "last_training_time": 0,
            "last_valid_count": 0
        }
        
        # Try to read existing training cycle info
        cycle_info = default_cycle_info
        if TRAINING_CYCLE_FILE.exists():
            try:
                with open(TRAINING_CYCLE_FILE, 'r') as f:
                    cycle_info = json.load(f)
                print(f"📊 Loaded existing training cycle info")
                
                # Make sure all fields exist
                for key, value in default_cycle_info.items():
                    if key not in cycle_info:
                        cycle_info[key] = value
            except Exception as e:
                print(f"⚠️ Warning: Could not read training cycle file: {e}")
                cycle_info = default_cycle_info
        
        # Verify last_valid_count matches reality
        if valid_count != cycle_info["last_valid_count"]:
            print(f"⚠️ Warning: Valid file count ({valid_count}) doesn't match training cycle info ({cycle_info['last_valid_count']})")
            # Only update if there's no ongoing training
            TRAINING_LOCK_FILE = Path("./.training_lock")
            if not TRAINING_LOCK_FILE.exists():
                print(f"✅ No training in progress, updating last_valid_count to match reality")
                cycle_info["last_valid_count"] = valid_count
                
                # Save updated training cycle info
                try:
                    with open(TRAINING_CYCLE_FILE, 'w') as f:
                        json.dump(cycle_info, f, indent=2)
                    print(f"✅ Saved updated training cycle info")
                except Exception as e:
                    print(f"⚠️ Warning: Could not save training cycle file: {e}")
            else:
                print(f"⚠️ Training appears to be in progress, not updating last_valid_count")
        
        # Print summary
        print("\n" + "="*50)
        print("📊 DATA FILES SUMMARY")
        print("="*50)
        print(f"Original data files: {original_count}")
        print(f"Valid data files: {valid_count}")
        print(f"Invalid data files: {invalid_count}")
        print(f"Consolidated data files: {consolidated_count}")
        print(f"Training data files: {training_count}")
        print(f"Total original data points (est.): {total_original_points}")
        print(f"Validation rate: {(valid_count/original_count*100) if original_count > 0 else 0:.1f}%")
        print(f"Training cycles completed: {cycle_info['cycle_count']}")
        print("="*50)
        
        scan_progress.complete("Data scan complete")
        return True
        
    except Exception as e:
        scan_progress.complete(f"Error during data scan: {e}")
        print(f"❌ Error scanning data files: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Explicitly import time in the function scope to ensure it's available
    import time

    print("[DEBUG] main() function called")
    # Add debug prints to track script execution
    print("\n" + "="*80)
    print("SCRIPT EXECUTION TRACKING")
    print("="*80)
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"[DEBUG] Python executable: {sys.executable}")
    print(f"[DEBUG] Process PID: {os.getpid()}")

    # Check for any queued signals before doing anything else
    if not signal_queue.empty():
        logger.info("Found queued signal at the start of main - processing it")
        process_signal()
    
    # Clear all pipeline ports to prevent conflicts
    print("\n" + "="*80)
    print("CLEARING PIPELINE PORTS")
    print("="*80)
    ports_cleared_successfully = clear_pipeline_ports()
    if not ports_cleared_successfully:
        print("⚠️ WARNING: Some ports could not be cleared. Pipeline may encounter conflicts.")
    print("="*80 + "\n")
    
    # Check for and clean up stale processes from previous sessions
    check_for_stale_processes()
    
    # Write the current session information to file for recovery if needed
    write_session_info()
    
    # Check and ensure adapter files are available
    print("\n" + "="*80)
    print("CHECKING LORA ADAPTER FILES")
    print("="*80)
    adapter_files_ready = check_and_ensure_adapter_files()
    if adapter_files_ready:
        print("✅ Adapter files are ready for inference")
    else:
        print("⚠️ WARNING: Adapter files may be missing or incomplete")
    print("="*80 + "\n")
    
    try:
        # First check if there's an existing inference server PID file
        if INFERENCE_PROCESS_PID_FILE.exists():
            pid = INFERENCE_PROCESS_PID_FILE.read_text().strip()
            print(f"Found existing PID file: {INFERENCE_PROCESS_PID_FILE}")
            print(f"Contains PID: {pid}")
        else:
            print(f"No existing PID file found at: {INFERENCE_PROCESS_PID_FILE}")
    except Exception as e:
        print(f"Error checking PID file: {e}")
    
    print("="*80)
    
    # Check if project modules are importable
    try:
        root_dir = Path(__file__).resolve().parent
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))
        from src.lora_training_pipeline.utils.helpers import (
            check_dependencies,
            validate_process_log_file,
            ProgressTracker,
            check_pending_errors,
            log_pending_error
        )

        # Check MLflow availability and configuration
        print("\n" + "="*80)
        print("CHECKING MLFLOW AVAILABILITY AND CONFIGURATION")
        print("="*80)

        # Wrap MLflow check in a try-except to ensure it never fails the entire pipeline
        try:
            # Set timeout for MLflow check
            import threading
            import time

            mlflow_check_done = False
            mlflow_available = False
            mlflow_check_result = None

            def run_mlflow_check():
                nonlocal mlflow_check_done, mlflow_available, mlflow_check_result
                try:
                    mlflow_check_result = check_mlflow_availability()
                    mlflow_available = mlflow_check_result
                except Exception as check_err:
                    print(f"❌ Error during MLflow availability check: {str(check_err)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    mlflow_check_result = False
                    mlflow_available = False
                finally:
                    mlflow_check_done = True

            # Start MLflow check in a separate thread
            mlflow_thread = threading.Thread(target=run_mlflow_check)
            mlflow_thread.daemon = True
            mlflow_thread.start()

            # Wait for the check to complete with a timeout
            timeout_seconds = 30
            start_time = time.time()
            while not mlflow_check_done and (time.time() - start_time) < timeout_seconds:
                # CRITICAL: Check for signals before sleeping - CTRL-C responsiveness
                if process_signal():
                    print("\n[DEBUG] Signal processed during MLflow check, exiting")
                    return  # Exit immediately if signal received
                
                # Check shutdown flag for immediate exit
                if shutdown_requested:
                    print("\n[DEBUG] Shutdown requested during MLflow check, exiting immediately")
                    sys.exit(0)
                
                print(f"Waiting for MLflow check to complete... ({int(time.time() - start_time)}s)", end="\r")
                time.sleep(0.5)  # Shorter sleep for better responsiveness

            # Handle timeout
            if not mlflow_check_done:
                print("\n❌ MLflow availability check timed out after 30 seconds")
                print("This usually indicates a connectivity issue or stalled process")
                mlflow_available = False

                # Log the timeout error for tracking
                log_pending_error("MLflow availability check timed out after 30 seconds")

            # Continue with the result
            if mlflow_available:
                print("✅ MLflow is available and correctly configured")
                print("✅ Experiment tracking will be enabled for model training")
            else:
                print("⚠️ MLflow is not properly configured")
                print("ℹ️ Training will still work, but experiment tracking may be limited")

                # Try to fix MLflow automatically
                try:
                    # Look for register_mlflow
                    register_module_name = None
                    for module_name in ["register_mlflow", "register_mlflow_direct", "register_mlflow_compat"]:
                        try:
                            # Try to dynamically import the module
                            register_module = __import__(module_name)
                            register_module_name = module_name
                            print(f"✅ Found MLflow registration module: {module_name}")
                            break
                        except ImportError:
                            print(f"⚠️ Module {module_name} not found")
                            continue

                    # If we found a registration module, use it
                    if register_module_name:
                        if hasattr(register_module, 'main'):
                            print(f"ℹ️ Attempting to automatically register MLflow using {register_module_name}")
                            try:
                                result = register_module.main()
                                if result == 0:
                                    print("✅ Successfully registered MLflow")
                                    mlflow_available = True
                                else:
                                    print(f"⚠️ MLflow registration returned non-zero exit code: {result}")
                            except Exception as reg_err:
                                print(f"❌ Error during MLflow registration: {str(reg_err)}")
                                import traceback
                                print(f"Traceback: {traceback.format_exc()}")
                        else:
                            print(f"⚠️ Module {register_module_name} has no main() function")
                    else:
                        print("⚠️ No MLflow registration module found")
                        print("ℹ️ You can manually register MLflow by running: python register_mlflow.py")

                except Exception as mlflow_err:
                    print(f"❌ Could not auto-register MLflow: {str(mlflow_err)}")
                    print(f"Exception type: {type(mlflow_err).__name__}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    print("ℹ️ You can manually register MLflow by running: python register_mlflow.py")

            # Set the environment variable for child processes
            os.environ["MLFLOW_AVAILABLE"] = "1" if mlflow_available else "0"

        except Exception as mlflow_check_err:
            # Catch all exceptions to ensure the pipeline continues
            print(f"❌ Critical error during MLflow check: {str(mlflow_check_err)}")
            print(f"Type: {type(mlflow_check_err).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            log_pending_error(f"Critical error in MLflow configuration check: {str(mlflow_check_err)}")
            print("ℹ️ Continuing without MLflow integration due to configuration error")
            mlflow_available = False

        # Always set the environment variable (even in case of errors)
        try:
            os.environ["MLFLOW_AVAILABLE"] = "1" if mlflow_available else "0"
            print(f"[DEBUG] Set MLFLOW_AVAILABLE={os.environ['MLFLOW_AVAILABLE']}")
        except Exception as e:
            print(f"[DEBUG] Warning: Failed to set MLFLOW_AVAILABLE environment variable: {e}")
            logger.warning(f"Failed to set MLFLOW_AVAILABLE environment variable: {e}")

        print("="*80 + "\n")

    except ImportError as e:
        print(f"\n{'='*80}\n❌ ERROR: Cannot import project modules: {e}")
        print("This typically happens when:")
        print("1. The project is not installed in development mode")
        print("2. You're not running from the correct directory")
        print("3. Your Python path is not set correctly\n")
        print(f"Current directory: {os.getcwd()}")
        print(f"Python path: {sys.path}\n")
        print("ℹ️ RECOMMENDED ACTION: Run 'uv pip install -e .' from project root")
        print(f"{'='*80}\n")
        
        # Attempt to log this critical error
        try:
            error_path = Path("pending_errors.txt")
            with open(error_path, "a") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CRITICAL: Cannot import project modules: {e}\n")
        except Exception as logging_error:
            # At least print to stdout since we can't write to file
            print(f"CRITICAL: Failed to log error to file: {logging_error}")
            print(f"Original error was: Cannot import project modules: {e}")
        
        # Terminate any processes that might have been started
        if processes:
            print(f"\n{'='*80}\nTerminating processes before exit:")
            for i, process in enumerate(processes):
                if process and process.poll() is None:
                    print(f"Terminating process {i+1}: {process.pid}")
                    try:
                        if os.name == 'nt':  # Windows
                            subprocess.run(['taskkill', '/F', '/PID', str(process.pid)], check=False)
                        else:
                            process.terminate()
                    except Exception as e:
                        print(f"Error terminating process: {e}")
            print(f"{'='*80}\n")
        sys.exit(1)
        
    # Clear the pending_errors.txt file at startup for a fresh run
    print("\n" + "="*80)
    print("INITIALIZING ERROR TRACKING")
    print("="*80)
    
    from src.lora_training_pipeline.utils.helpers import clear_pending_errors, setup_global_exception_handler
    if clear_pending_errors():
        print("✅ Error log initialized: Any runtime errors will be logged to pending_errors.txt")
    else:
        print("⚠️ Warning: Could not initialize error log")
    
    # Set up global exception handler to catch unhandled exceptions
    setup_global_exception_handler()
    
    print("="*80)
    
    # Create a progress tracker for initialization
    init_progress = ProgressTracker("Pipeline Initialization")
    init_progress.start("Checking environment and dependencies")
    
    # Run initial data scan to count files and update counters
    init_progress.update(message="Scanning data files")
    scan_data_files(progress_tracker=init_progress)
    
    # Keep the process log validation - it's still useful
    if os.name == 'nt':
        init_progress.update(message="Validating Windows process log")
        try:
            # The function is already Windows-only, it will return early on other platforms
            validate_process_log_file()
        except Exception as e:
            print(f"⚠️ WARNING: Error validating process log file: {e}")
            print("Continuing with pipeline execution...")
    
    # Check all required modules
    init_progress.update(message="Checking required Python modules")
    required_modules = ['gradio', 'zenml', 'torch', 'transformers', 'peft', 'fastapi', 'mlflow']
    check_dependencies(required_modules, processes)
    
    # Check if MLflow tracker exists
    init_progress.update(message="Checking MLflow tracker")
    try:
        from zenml.client import Client
        client = Client()
        
        # Handle different ZenML versions with different MLflow tracking APIs
        mlflow_tracker_found = False
        tracker_names = []
        
        # Try multiple methods to check for MLflow tracker based on ZenML version
        try:
            if hasattr(client, 'list_experiment_trackers') and callable(client.list_experiment_trackers):
                trackers = client.list_experiment_trackers()
                tracker_names = [t.name for t in trackers]
                mlflow_tracker_found = "mlflow_tracker" in tracker_names
                print(f"✅ Found experiment trackers via list_experiment_trackers: {tracker_names}")
            elif hasattr(client, 'get_stack_component') and callable(client.get_stack_component):
                # Try newer ZenML API
                try:
                    # Try to get MLflow component directly
                    mlflow_component = client.get_stack_component("mlflow_tracker")
                    mlflow_tracker_found = True
                    tracker_names = ["mlflow_tracker"]
                    print(f"✅ Found MLflow tracker via get_stack_component")
                except Exception as stack_comp_err:
                    print(f"[DEBUG] get_stack_component error type: {type(stack_comp_err).__name__}")
                    print(f"[DEBUG] get_stack_component error details: {stack_comp_err}")
                    print(f"[DEBUG] Failed to get MLflow tracker via get_stack_component, trying active_stack")
                    # Try to get active stack components
                    if hasattr(client, 'active_stack') and client.active_stack:
                        components = client.active_stack.components
                        for comp_name, comp in components.items():
                            tracker_names.append(comp_name)
                            if "mlflow" in comp_name.lower():
                                mlflow_tracker_found = True
                        print(f"✅ Found stack components: {tracker_names}")
            elif hasattr(client, 'experiments'):
                # Very old ZenML versions
                print("⚠️ Using legacy ZenML experiments API")
                mlflow_tracker_found = True  # Assume it exists in old versions
        except Exception as method_error:
            print(f"⚠️ Error checking for MLflow tracker method: {method_error}")
        
        # Show appropriate message based on what we found
        if not mlflow_tracker_found:
            print("\n" + "="*80)
            print("⚠️ MLFLOW TRACKER NOT FOUND IN ZENML STACK")
            print("="*80)
            print("This will prevent proper experiment tracking and hyperparameter tuning")
            print("\nTo set up MLflow:")
            print("1. Run the setup script: python register_mlflow.py")
            print("2. Verify with: zenml stack list")
            print("3. If using Docker, ensure MLflow service is running: docker-compose up -d mlflow")
            print("\nHyperparameter tuning requires a working MLflow tracker")
            print("Continuing with limited functionality, but some features may not work correctly")
            print("="*80 + "\n")
        else:
            print(f"✅ MLflow tracker exists in ZenML stack: {tracker_names}")
    except Exception as e:
        print(f"⚠️ WARNING: Could not check MLflow tracker: {e}")
        print("ℹ️ Continuing anyway, but hyperparameter tuning may fail")
            
    # --- Activate Virtual Environment (if using uv)---
    if not os.environ.get("VIRTUAL_ENV"):  # Check if virtual env is already active
        init_progress.update(message="Activating virtual environment")
        if os.name == 'nt':  # Windows
            # Skip trying to activate the virtual environment on Windows
            # as it's already activated in the PowerShell window
            init_progress.complete("Assuming virtual environment is already active")
            print("ℹ️ Continuing with current Python environment")
            # No need to restart the process - continue with current environment
        else:  # Linux/macOS
            activate_script = ".venv/bin/activate"
            # Use shell=True here, as we need to source the activation script.
            init_progress.complete("Virtual environment found, restarting with proper environment")
            start_process(f"source {activate_script} && python run_pipeline.py --active", shell=True)
        
        if os.name != 'nt':  # Only exit on non-Windows platforms
            sys.exit(0) #Exit current process, re run in the activated environment.
        # On Windows, we continue with the current process

    # Complete initialization
    init_progress.complete("Environment check complete")

    # Start components with progress tracking
    components_progress = ProgressTracker("Pipeline Components", total=4)  # Updated for dashboard component
    components_progress.start("Starting all pipeline components")

    print("\n" + "="*80)
    print("INITIALIZING PIPELINE COMPONENTS")
    print("="*80)
    
    # --- Start Dashboard UI FIRST ---
    components_progress.update(message="Starting Dashboard UI")
    dashboard_cmd = [
        "python",
        "-m", "src.lora_training_pipeline.utils.dashboard",
    ]
    
    # Create environment with port configuration
    dashboard_env = {
        "PROCESS_NAME": "PipelineDashboard",
        "DASHBOARD_PORT": str(DASHBOARD_PORT),  # Use the defined constant
        "DASHBOARD_API_URL": DASHBOARD_API_URL  # Set explicit API URL
    }
    
    # Start the process with a descriptive name in logs
    start_process(dashboard_cmd, env=dashboard_env)
    
    print(f"✅ Pipeline Dashboard started (port: {DASHBOARD_PORT})")
    
    # Wait for Dashboard to initialize
    dashboard_startup = ProgressTracker("Dashboard Initialization")
    dashboard_startup.start("Waiting for Dashboard server to start")
    dashboard_startup.start_timer("Starting Dashboard server", interval=1)
    
    # Give Dashboard time to initialize with signal checking
    for i in range(6):  # 6 * 0.5s = 3s total
        # CRITICAL: Check for signals before sleeping - CTRL-C responsiveness
        if process_signal():
            print("\n[DEBUG] Signal processed during Dashboard startup, exiting")
            return None
        
        # Check shutdown flag for immediate exit
        if shutdown_requested:
            print("\n[DEBUG] Shutdown requested during Dashboard startup, exiting immediately")
            sys.exit(0)
        
        time.sleep(0.5)
    
    dashboard_startup.complete("Dashboard server started")
    
    # --- Start Data Collection Gradio App ---
    components_progress.update(message="Starting Data Collection UI")
    
    print("\n" + "="*80)
    print("🔍 DATA COLLECTION UI PROCESS DETECTION")
    print("="*80)
    print(f"📁 PID file path: {DATA_COLLECTION_PID_FILE}")
    
    # Before starting a new UI, check if we already have a running instance
    existing_data_collection_ui = None

    # First check if we have a PID file
    if DATA_COLLECTION_PID_FILE.exists():
        try:
            file_content = DATA_COLLECTION_PID_FILE.read_text().strip()
            print(f"📄 PID file content: '{file_content}'")
            
            # Validate that the content is a number
            if not file_content.isdigit():
                print(f"⚠️ WARNING: PID file contains non-numeric value: '{file_content}'")
                raise ValueError(f"Invalid PID value: {file_content}")
                
            ui_pid = int(file_content)
            print(f"🔍 Found Data Collection UI PID file with PID: {ui_pid}")
            
            # Check if this process is still running
            try:
                if PSUTIL_AVAILABLE:
                    print(f"🔍 Checking if process {ui_pid} is running...")
                    import psutil  # Double-check import
                    process = psutil.Process(ui_pid)
                    
                    if process.is_running():
                        print(f"✅ Process {ui_pid} is running")
                        
                        # Verify it's actually our UI process
                        try:
                            cmdline = ' '.join(process.cmdline()) if process.cmdline() else ''
                            print(f"🔍 Process {ui_pid} command line: {cmdline[:150]}...")
                            
                            if 'gradio_app.py' in cmdline:
                                print(f"✅ Found existing Data Collection UI process: PID={ui_pid}")
                                existing_data_collection_ui = process
                                
                                # Additional process details for debugging
                                try:
                                    create_time = process.create_time()
                                    running_time = time.time() - create_time
                                    print(f"📊 Process details: Created {time.ctime(create_time)} ({running_time:.1f} seconds ago)")
                                    print(f"📊 Process status: {process.status()}")
                                    print(f"📊 Process memory: {process.memory_info().rss / (1024 * 1024):.1f} MB")
                                except Exception as detail_err:
                                    print(f"⚠️ Could not get complete process details: {detail_err}")
                            else:
                                print(f"⚠️ Process {ui_pid} exists but is not the Data Collection UI: {cmdline[:100]}")
                        except Exception as cmd_err:
                            print(f"⚠️ Error getting command line for process {ui_pid}: {cmd_err}")
                    else:
                        print(f"⚠️ Process {ui_pid} from PID file is not running")
                else:
                    print("⚠️ psutil module not available, cannot check if process is running")
            except psutil.NoSuchProcess:
                print(f"⚠️ Process {ui_pid} from PID file does not exist")
            except psutil.AccessDenied:
                print(f"⚠️ Access denied when checking process {ui_pid}")
            except Exception as proc_err:
                print(f"⚠️ Unexpected error checking process {ui_pid}: {type(proc_err).__name__}: {proc_err}")
        except Exception as pid_err:
            print(f"❌ Error reading PID file: {type(pid_err).__name__}: {pid_err}")

    # If no process found from PID file, scan for one
    if not existing_data_collection_ui and PSUTIL_AVAILABLE:
        print("\n" + "-"*60)
        print("🔍 SCANNING FOR EXISTING DATA COLLECTION UI PROCESSES")
        print("-"*60)
        
        found_processes = 0
        scan_start_time = time.time()
        
        try:
            print("👁️ Requesting process list with pid, name, and cmdline info...")
            # Import psutil should be at module level, but double-check it's available
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    pid = proc.info['pid']
                    name = proc.info['name']
                    
                    # Skip processes that definitely aren't Python
                    if name.lower() not in ('python', 'python.exe', 'python3', 'python3.exe'):
                        continue
                        
                    cmdline = proc.info['cmdline']
                    if not cmdline:
                        continue
                        
                    cmdline_str = ' '.join(cmdline)
                    
                    # Only show debug for Python processes to reduce noise
                    if 'python' in name.lower():
                        print(f"🔍 Checking Python process {pid}: {name}")
                        print(f"   Command: {cmdline_str[:100]}...")
                    
                    if 'gradio_app.py' in cmdline_str:
                        found_processes += 1
                        print(f"\n✨ Found potential Data Collection UI: PID={pid}")
                        print(f"   Full command: {cmdline_str[:200]}...")
                        
                        # Check if it's responsive
                        try:
                            print(f"   Testing if process {pid} is responsive...")
                            response_start = time.time()
                            proc.cpu_percent()  # Simple test to see if process responds
                            response_time = time.time() - response_start
                            print(f"✅ Process {pid} is responsive (response time: {response_time:.3f}s)")
                            
                            # Get detailed process info
                            try:
                                create_time = proc.create_time()
                                running_time = time.time() - create_time
                                memory = proc.memory_info().rss / (1024 * 1024)
                                print(f"📊 Process details:")
                                print(f"   - Created: {time.ctime(create_time)} ({running_time:.1f} seconds ago)")
                                print(f"   - Status: {proc.status()}")
                                print(f"   - Memory: {memory:.1f} MB")
                            except Exception as detail_err:
                                print(f"⚠️ Could not get complete process details: {detail_err}")
                                
                            existing_data_collection_ui = proc
                            
                            # Update PID file
                            print(f"📝 Updating PID file: {DATA_COLLECTION_PID_FILE} with PID: {pid}")
                            try:
                                DATA_COLLECTION_PID_FILE.write_text(str(pid))
                                print(f"✅ Successfully updated Data Collection UI PID file with found process: {pid}")
                            except Exception as write_err:
                                print(f"❌ ERROR updating PID file: {type(write_err).__name__}: {write_err}")
                                
                            break
                        except psutil.NoSuchProcess:
                            print(f"⚠️ Process {pid} disappeared during check")
                        except psutil.AccessDenied:
                            print(f"⚠️ Access denied when checking if process {pid} is responsive")
                        except Exception as resp_err:
                            print(f"⚠️ Process {pid} error during responsiveness check: {type(resp_err).__name__}: {resp_err}")
                            print(f"   Process {pid} is not responsive, will not use it")
                except psutil.NoSuchProcess:
                    # Process disappeared during iteration
                    continue
                except psutil.AccessDenied:
                    # Can't access this process
                    continue
                except Exception as proc_err:
                    print(f"⚠️ Error inspecting process: {type(proc_err).__name__}: {proc_err}")
                    continue
                    
            scan_time = time.time() - scan_start_time
            print(f"\n🔍 Process scan complete in {scan_time:.2f} seconds")
            print(f"   Found {found_processes} potential Data Collection UI processes")
            print(f"   Using process: {existing_data_collection_ui.pid if existing_data_collection_ui else 'None'}")
        except Exception as scan_err:
            print(f"❌ ERROR scanning for processes: {type(scan_err).__name__}: {scan_err}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    # Use direct Python command for all platforms - no batch files
    data_collection_cmd = [
        "python",
        "src/lora_training_pipeline/data_collection/gradio_app.py",
    ]
    
    # Create environment with port configuration
    data_collection_env = {
        "PROCESS_NAME": "GradioDataCollection",
        "GRADIO_PORT": str(DATA_COLLECTION_PORT),
        "GRADIO_DATA_COLLECTION_PORT": str(DATA_COLLECTION_PORT),  # Explicit variable name
        "DATA_COLLECTION_API_URL": DATA_COLLECTION_API_URL  # Set explicit API URL
    }
    
    print("\n" + "="*80)
    print("🚀 DATA COLLECTION UI STARTUP")
    print("="*80)
    
    # Start a new process if needed
    if existing_data_collection_ui:
        print(f"✅ Reusing existing Data Collection UI process (PID={existing_data_collection_ui.pid})")
        print(f"📡 Data Collection UI already running on port: {DATA_COLLECTION_PORT}")
        
        # Verify the process is still responsive
        try:
            print(f"🔍 Verifying process {existing_data_collection_ui.pid} is still responsive...")
            response_start = time.time()
            existing_data_collection_ui.cpu_percent()
            response_time = time.time() - response_start
            print(f"✅ Process is still responsive (response time: {response_time:.3f}s)")
            
            # Additional validation that the port is active
            try:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                result = s.connect_ex(('127.0.0.1', DATA_COLLECTION_PORT))
                if result == 0:
                    print(f"✅ Confirmed port {DATA_COLLECTION_PORT} is active")
                else:
                    print(f"⚠️ Port check WARNING: Port {DATA_COLLECTION_PORT} appears to be closed (result={result})")
                    print(f"   Process may be running but not listening yet")
                s.close()
            except Exception as port_err:
                print(f"⚠️ Error checking port: {type(port_err).__name__}: {port_err}")
        except Exception as verify_err:
            print(f"⚠️ WARNING: Process verification failed: {type(verify_err).__name__}: {verify_err}")
            print(f"   Will continue with existing process anyway")
            
        print(f"ℹ️ If Data Collection UI is not accessible at http://localhost:{DATA_COLLECTION_PORT}, restart the pipeline")
    else:
        print("🚀 Starting new Data Collection UI process")
        print(f"📋 Command: {' '.join(data_collection_cmd)}")
        print(f"🔧 Environment variables:")
        for key, value in data_collection_env.items():
            print(f"   - {key}: {value}")
            
        try:
            data_collection_process = start_process(data_collection_cmd, env=data_collection_env)
            
            if data_collection_process:
                print(f"✅ Process started successfully")
                
                # Record the PID for future reference
                if hasattr(data_collection_process, 'pid'):
                    pid = data_collection_process.pid
                    print(f"📝 Writing PID {pid} to file: {DATA_COLLECTION_PID_FILE}")
                    
                    try:
                        DATA_COLLECTION_PID_FILE.write_text(str(pid))
                        print(f"✅ Successfully recorded Data Collection UI PID: {pid}")
                        
                        # Verify file was written correctly
                        try:
                            written_content = DATA_COLLECTION_PID_FILE.read_text().strip()
                            if written_content == str(pid):
                                print(f"✅ PID file verification successful: '{written_content}'")
                            else:
                                print(f"⚠️ WARNING: PID file content mismatch: Expected '{pid}', got '{written_content}'")
                        except Exception as verify_err:
                            print(f"⚠️ WARNING: Could not verify PID file contents: {verify_err}")
                    except Exception as pid_error:
                        print(f"❌ Error recording PID: {type(pid_error).__name__}: {pid_error}")
                        
                    # Basic process validation
                    try:
                        print(f"🔍 Validating new process {pid}...")
                        if psutil.pid_exists(pid):
                            p = psutil.Process(pid)
                            status = p.status()
                            print(f"✅ Process exists with status: {status}")
                        else:
                            print(f"⚠️ WARNING: Process {pid} does not appear in process list!")
                    except Exception as validate_err:
                        print(f"⚠️ WARNING: Process validation error: {type(validate_err).__name__}: {validate_err}")
                else:
                    print("⚠️ WARNING: Could not get PID for Data Collection UI process")
            else:
                print("❌ ERROR: Data Collection UI process failed to start")
        except Exception as start_err:
            print(f"❌ ERROR starting Data Collection UI process: {type(start_err).__name__}: {start_err}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    print(f"✅ Data Collection UI should now be running on port: {DATA_COLLECTION_PORT}")
    print(f"🌐 Access the UI at: http://localhost:{DATA_COLLECTION_PORT}")
    print("="*80)
    
    # Wait for Gradio to initialize with progress indication
    gradio_startup = ProgressTracker("Gradio Initialization")
    gradio_startup.start("Waiting for Gradio server to start")
    gradio_startup.start_timer("Starting Gradio server", interval=1)
    
    # Give Gradio more time to fully initialize with signal checking
    for i in range(10):  # 10 * 0.5s = 5s total
        # CRITICAL: Check for signals before sleeping - CTRL-C responsiveness
        if process_signal():
            print("\n[DEBUG] Signal processed during Gradio startup, exiting")
            return None
        
        # Check shutdown flag for immediate exit
        if shutdown_requested:
            print("\n[DEBUG] Shutdown requested during Gradio startup, exiting immediately")
            sys.exit(0)
        
        time.sleep(0.5)
    
    gradio_startup.complete("Gradio server started")

    # --- Run ZenML Pipeline ---
    components_progress.update(message="Starting ZenML Training Pipeline")
    
    # Use direct Python command for all platforms - no batch files
    # Use simple "python" command - this is more reliable
    zenml_cmd = [
        "python",
        "src/lora_training_pipeline/training/zenml_pipeline.py",
    ]
    
    # Create environment variables to help debug the ZenML pipeline
    zenml_env = {
        "PROCESS_NAME": "ZenMLTrainingPipeline",
        "DEBUG_ZENML": "true",  # Enable additional debugging
        "PYTHONPATH": os.getcwd(),  # Ensure consistent Python path
        "STOP_INFERENCE_ON_EXIT": "false"  # Make sure the ZenML pipeline preserves the inference server
    }
    
    # Start the process with more descriptive name in logs
    start_process(zenml_cmd, env=zenml_env)
    print("✅ ZenML pipeline started")
    print("ℹ️ The pipeline will try multiple execution methods to handle different ZenML versions")
    
    # --- Start FastAPI Inference Server ---
    components_progress.update(message="Starting FastAPI Inference Server")
    
    print("\n" + "="*80)
    print("FASTAPI SERVER PRE-LAUNCH CHECKS")
    print("="*80)
    
    # Check for existing FastAPI server processes using psutil
    existing_fastapi_process = None
    try:
        import psutil
        print("Scanning for existing FastAPI inference server processes...")
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'connections']):
            try:
                cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else ''
                if 'uvicorn' in cmdline and 'fastapi_inference:app' in cmdline:
                    print(f"Found potential FastAPI server process: PID={proc.pid}")
                    print(f"Command: {cmdline[:150]}")
                    
                    # Check if it's listening on our target port
                    try:
                        for conn in proc.connections(kind='inet'):
                            if conn.laddr.port == FASTAPI_INFERENCE_PORT and conn.status == 'LISTEN':
                                print(f"✅ FOUND ACTIVE FASTAPI SERVER: PID={proc.pid} on port {FASTAPI_INFERENCE_PORT}")
                                existing_fastapi_process = proc
                                # Update PID file with the found process
                                try:
                                    INFERENCE_PROCESS_PID_FILE.write_text(str(proc.pid))
                                    print(f"✓ Updated PID file with found server: {proc.pid}")
                                except Exception as pid_write_err:
                                    print(f"⚠️ Warning: Failed to update PID file: {pid_write_err}")
                                break
                    except (psutil.AccessDenied, Exception) as conn_err:
                        print(f"  Error checking connections: {conn_err}")
                        
                    if existing_fastapi_process:
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as proc_err:
                continue
                
        if existing_fastapi_process:
            print(f"✅ Found existing FastAPI server process (PID={existing_fastapi_process.pid})")
            print("Will reuse existing process instead of starting a new one")
        else:
            print("No existing FastAPI server process found")
    except ImportError:
        print("⚠️ psutil not available - can't check for existing FastAPI processes")
    
    # Check PID file only if we didn't find an active process
    if not existing_fastapi_process and INFERENCE_PROCESS_PID_FILE.exists():
        print(f"WARNING: PID file exists at {INFERENCE_PROCESS_PID_FILE}, but no active process was found")
        try:
            # Read the PID and check if process exists but we missed it
            old_pid = int(INFERENCE_PROCESS_PID_FILE.read_text().strip())
            print(f"PID file contains: {old_pid}")
            
            try:
                if PSUTIL_AVAILABLE:  # We have psutil imported
                    # Double-check if process exists
                    try:
                        import psutil  # Double-check import
                        proc = psutil.Process(old_pid)
                        if proc.is_running():
                            cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else ''
                            if 'uvicorn' in cmdline and 'fastapi_inference' in cmdline:
                                print(f"✅ Process {old_pid} is still running and is a FastAPI server")
                                existing_fastapi_process = proc
                            else:
                                print(f"⚠️ Process {old_pid} is running but is not a FastAPI server: {cmdline}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        print(f"Process {old_pid} from PID file doesn't exist or is inaccessible")
            except Exception as proc_check_err:
                print(f"Error checking process from PID file: {proc_check_err}")
            
            # Only remove PID file if we couldn't find a valid process
            if not existing_fastapi_process:
                INFERENCE_PROCESS_PID_FILE.unlink()
                print("✅ Removed old PID file")
        except Exception as e:
            print(f"ERROR: Failed to process PID file: {e}")
            try:
                INFERENCE_PROCESS_PID_FILE.unlink()
                print("✅ Removed invalid PID file")
            except Exception as unlink_err:
                print(f"ERROR: Failed to remove PID file: {unlink_err}")
    elif not existing_fastapi_process:
        print("✅ No existing PID file found, safe to start server")
    
    # Check if port is already in use with enhanced debugging
    try:
        import socket
        
        # Try multiple localhost variants since Windows/WSL can be picky
        hosts_to_try = ["localhost", "127.0.0.1", "::1"]
        port_in_use = False
        successful_connection = None
        
        print("\n" + "="*80)
        print(f"DETAILED PORT {FASTAPI_INFERENCE_PORT} AVAILABILITY CHECK")
        print("="*80)
        
        for host in hosts_to_try:
            try:
                print(f"Testing connectivity to {host}:{FASTAPI_INFERENCE_PORT}...")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)  # Longer timeout for better reliability
                result = s.connect_ex((host, FASTAPI_INFERENCE_PORT))
                s.close()
                
                if result == 0:
                    print(f"⚠️ Port {FASTAPI_INFERENCE_PORT} is in use on {host}")
                    port_in_use = True
                    successful_connection = host
                    
                    # Try to get more info about what's using the port
                    try:
                        import psutil
                        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                            try:
                                for conn in proc.connections(kind='inet'):
                                    if conn.laddr.port == FASTAPI_INFERENCE_PORT:
                                        print(f"  - Process using port: PID={proc.pid}, Name={proc.name()}")
                                        cmdline = " ".join(proc.cmdline()) if proc.cmdline() else "Unknown"
                                        print(f"  - Command: {cmdline[:100]}{'...' if len(cmdline) > 100 else ''}")
                                        print(f"  - Created: {datetime.datetime.fromtimestamp(proc.create_time()).strftime('%H:%M:%S')}")
                            except (psutil.AccessDenied, psutil.NoSuchProcess):
                                pass
                    except Exception as proc_err:
                        print(f"  - Error getting process info: {proc_err}")
                else:
                    print(f"✅ Port {FASTAPI_INFERENCE_PORT} is available on {host} (code: {result})")
            except Exception as host_err:
                print(f"⚠️ Error checking {host}:{FASTAPI_INFERENCE_PORT}: {host_err}")
        
        if port_in_use:
            print(f"\n⚠️ WARNING: Port {FASTAPI_INFERENCE_PORT} is already in use!")
            print(f"  - Successfully connected to: {successful_connection}:{FASTAPI_INFERENCE_PORT}")
            print("Will attempt to start server anyway, but it may fail")
        else:
            print(f"\n✅ Port {FASTAPI_INFERENCE_PORT} is available on all tested hosts")
        
        print("="*80)
    except Exception as e:
        print(f"⚠️ Error checking port availability: {e}")
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")
    
    # Create environment with port configuration
    fastapi_env = {
        "PROCESS_NAME": "FastAPIInferenceServer",
        "FASTAPI_PORT": str(FASTAPI_INFERENCE_PORT),
        "FASTAPI_INFERENCE_PORT": str(FASTAPI_INFERENCE_PORT),  # Explicitly named variable
        "INFERENCE_API_URL": INFERENCE_API_URL,  # Set explicit API URL
        # Set model paths if available
        "LORA_MODEL_PATH": str(Path("./output/best_model").absolute()),
        "MODEL_UPDATE_SIGNAL_FILE": str(MODEL_UPDATE_SIGNAL_FILE.absolute()),
        # Add PYTHONPATH to ensure module imports work
        "PYTHONPATH": os.getcwd()
    }
    
    # Use direct Python with uvicorn module for reliable startup
    # Determine if we're in development mode
    DEV_MODE = os.environ.get("DEV_MODE", "false").lower() == "true"

    # Only use --reload in development mode to prevent duplicate processes
    fastapi_server_cmd = [
        sys.executable,  # Use the current Python interpreter
        "-m", "uvicorn", 
        "src.lora_training_pipeline.inference.fastapi_inference:app", 
        "--host", "0.0.0.0", 
        "--port", str(FASTAPI_INFERENCE_PORT),
        "--workers", "1"  # Ensure only one worker process
    ]

    # Add --reload flag only in development mode
    if DEV_MODE:
        fastapi_server_cmd.append("--reload")
        print("ℹ️ Running FastAPI server in development mode with hot-reload enabled")
    else:
        print("ℹ️ Running FastAPI server in production mode (no hot-reload)")
    
    # Print detailed debug info about the command we're running
    print("\n" + "="*80)
    print("FASTAPI SERVER LAUNCH DETAILS")
    print("="*80)
    print(f"Python interpreter: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Port: {FASTAPI_INFERENCE_PORT}")
    print(f"Module: src.lora_training_pipeline.inference.fastapi_inference:app")
    print(f"Command: {' '.join(fastapi_server_cmd)}")
    print("="*80)
    
    # Only start the server if we didn't find an existing one
    if existing_fastapi_process:
        print(f"Reusing existing FastAPI server process (PID={existing_fastapi_process.pid})")
        fastapi_process = existing_fastapi_process
    else:
        # Check for socket binding capability as final verification
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                # Use 0.0.0.0 to bind to all interfaces like Uvicorn will
                s.bind(("0.0.0.0", FASTAPI_INFERENCE_PORT))
                s.close()
                print(f"✅ VERIFIED: Port {FASTAPI_INFERENCE_PORT} is available for binding")
                port_bindable = True
            except Exception as bind_error:
                print(f"❌ CRITICAL: Port {FASTAPI_INFERENCE_PORT} is NOT available for binding: {bind_error}")
                print("This will cause Uvicorn to fail when starting!")
                port_bindable = False
                
                # Try to find what's using the port
                try:
                    if PSUTIL_AVAILABLE:
                        print("Scanning for process using port...")
                        import psutil  # Double-check import
                        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'connections']):
                            try:
                                for conn in proc.connections(kind='inet'):
                                    if conn.laddr.port == FASTAPI_INFERENCE_PORT:
                                        print(f"⚠️ Port {FASTAPI_INFERENCE_PORT} is being used by:")
                                        print(f"   Process: {proc.name()} (PID={proc.pid})")
                                        cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else 'Unknown'
                                        print(f"   Command: {cmdline[:200]}")
                                        
                                        # If it's our FastAPI server, use it
                                        if 'uvicorn' in cmdline and 'fastapi_inference:app' in cmdline:
                                            print(f"✅ This is our FastAPI server, will use it")
                                            existing_fastapi_process = proc
                                            # Update PID file
                                            INFERENCE_PROCESS_PID_FILE.write_text(str(proc.pid))
                                            break
                            except Exception as e:
                                print(f"[DEBUG] Error updating PID file: {e}")
                                pass
                except Exception as proc_scan_err:
                    print(f"Error scanning for processes using port: {proc_scan_err}")
        except ImportError:
            print("Socket module not available - can't check binding capability")
            port_bindable = True  # Assume it works
    
        # Proceed with starting server if no existing process was found
        if existing_fastapi_process:
            print(f"Found FastAPI server process during port scan (PID={existing_fastapi_process.pid})")
            fastapi_process = existing_fastapi_process
        elif not port_bindable:
            print("⚠️ WARNING: Port is not available, but no FastAPI server process was found.")
            print("Continuing with launch, but it will likely fail...")
            try:
                # Start the inference server process
                fastapi_process = start_process(fastapi_server_cmd, env=fastapi_env)
                print(f"✅ FastAPI Inference Server started (PID: {fastapi_process.pid if hasattr(fastapi_process, 'pid') else 'unknown'})")
                
                # Store the PID in the PID file for tracking
                if fastapi_process and hasattr(fastapi_process, 'pid'):
                    try:
                        INFERENCE_PROCESS_PID_FILE.write_text(str(fastapi_process.pid))
                        print(f"✓ Recorded FastAPI server PID: {fastapi_process.pid} in file: {INFERENCE_PROCESS_PID_FILE}")
                    except Exception as pid_error:
                        print(f"❌ Error recording PID: {pid_error}")
                else:
                    print("⚠️ WARNING: Could not get PID for FastAPI process")
            except Exception as launch_error:
                print(f"❌ ERROR: Failed to start FastAPI server: {launch_error}")
        else:
            print("Starting new FastAPI server process...")
            try:
                # Start the inference server process
                fastapi_process = start_process(fastapi_server_cmd, env=fastapi_env)
                print(f"✅ FastAPI Inference Server started (PID: {fastapi_process.pid if hasattr(fastapi_process, 'pid') else 'unknown'})")
                
                # Store the PID in the PID file for tracking
                if fastapi_process and hasattr(fastapi_process, 'pid'):
                    try:
                        INFERENCE_PROCESS_PID_FILE.write_text(str(fastapi_process.pid))
                        print(f"✓ Recorded FastAPI server PID: {fastapi_process.pid} in file: {INFERENCE_PROCESS_PID_FILE}")
                    except Exception as pid_error:
                        print(f"❌ Error recording PID: {pid_error}")
                else:
                    print("⚠️ WARNING: Could not get PID for FastAPI process")
            except Exception as launch_error:
                print(f"❌ ERROR: Failed to start FastAPI server: {launch_error}")
    
    print("="*80)
    
    # --- Start Inference Gradio App ---
    # Start this *after* the FastAPI server, as the inference UI needs to connect to it
    components_progress.update(message="Waiting for inference server to initialize")
    
    # Track the server initialization waiting period
    server_wait = ProgressTracker("Inference Server Initialization")
    server_wait.start("Waiting for inference server to start")
    
    # Add a verification step to ensure the server is actually running
    server_wait.update(message="Verifying FastAPI server is running")
    
    # Use a more active approach to wait for the server - check if it responds
    import socket
    import time
    
    def check_port_open(host, port, timeout=2):
        """
        Check if a port is open on the given host with detailed error reporting.
        
        Args:
            host: The hostname to connect to
            port: The port number to check
            timeout: Connection timeout in seconds
            
        Returns:
            tuple: (is_open, error_message) where is_open is a boolean and error_message 
                   contains details about any failures (or None if successful)
        """
        try:
            print(f"  • Socket test: Connecting to {host}:{port} (timeout: {timeout}s)")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            # Record start time to measure connection time
            start_time = time.time()
            result = sock.connect_ex((host, port))
            elapsed_time = time.time() - start_time
            
            # Get more details about the socket
            try:
                socket_family = sock.family
                socket_type = sock.type
                socket_details = f"Family: {socket_family}, Type: {socket_type}"
            except Exception as sock_err:
                socket_details = f"Error getting socket details: {sock_err}"
            
            sock.close()
            
            if result == 0:
                print(f"  ✓ Connected successfully to {host}:{port} ({elapsed_time:.2f}s)")
                return True, None
            else:
                error_msg = f"Connection failed with code {result} after {elapsed_time:.2f}s (Socket: {socket_details})"
                print(f"  ✗ {error_msg}")
                return False, error_msg
        except socket.gaierror as e:
            error_msg = f"Address resolution error for {host}: {e}"
            print(f"  ✗ {error_msg}")
            return False, error_msg
        except socket.timeout as e:
            error_msg = f"Connection timed out after {timeout}s: {e}"
            print(f"  ✗ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {type(e).__name__}: {e}"
            print(f"  ✗ {error_msg}")
            return False, error_msg
    
    # Wait actively for server to start, with timeout
    max_wait = 30  # Maximum wait time in seconds
    wait_interval = 1  # Check every second
    server_up = False
    
    server_wait.start_timer("Waiting for inference server to initialize", interval=wait_interval)
    
    start_time = time.time()
    connection_errors = []
    
    # Try multiple hostname variants for localhost since Windows/WSL can be picky
    hosts_to_try = ["localhost", "127.0.0.1", "::1"]
    
    print("\n" + "="*80)
    print(f"FASTAPI SERVER CONNECTIVITY VERIFICATION")
    print("="*80)
    print(f"Starting server connectivity check with {len(hosts_to_try)} hostname variants")
    print(f"Using port: {FASTAPI_INFERENCE_PORT}")
    print(f"Maximum wait time: {max_wait} seconds")
    print(f"Check interval: {wait_interval} seconds")
    
    while time.time() - start_time < max_wait:
        # Try each hostname variant
        for host in hosts_to_try:
            print(f"\nAttempting connection to {host}:{FASTAPI_INFERENCE_PORT}")
            is_open, error = check_port_open(host, FASTAPI_INFERENCE_PORT)
            
            if is_open:
                server_up = True
                print(f"🌐 Successful connection using hostname: {host}")
                # Remember the working hostname for future use
                WORKING_HOSTNAME = host
                break
            else:
                connection_errors.append((host, error))
                
        if server_up:
            break
            
        # Print progress after trying all hosts
        elapsed = time.time() - start_time
        remaining = max_wait - elapsed
        if remaining > 0:
            print(f"⏳ Server not responding yet. Retrying in {wait_interval}s ({remaining:.1f}s remaining)")
            
            # CRITICAL: Check for signals before sleeping - CTRL-C responsiveness
            if process_signal():
                print("\n[DEBUG] Signal processed during FastAPI server wait, exiting")
                return None, None
            
            # Check shutdown flag for immediate exit
            if shutdown_requested:
                print("\n[DEBUG] Shutdown requested during FastAPI server wait, exiting immediately")
                sys.exit(0)
            
            time.sleep(wait_interval)
    
    if server_up:
        server_wait.complete(f"FastAPI server verified running on port {FASTAPI_INFERENCE_PORT}")
        print(f"✅ VERIFICATION: FastAPI server is responding on port {FASTAPI_INFERENCE_PORT}")
        print(f"✅ Working hostname: {WORKING_HOSTNAME}")
        
        # Check if we can also make an HTTP health check
        try:
            import httpx
            http_start = time.time()
            print(f"🔍 Performing HTTP health check at http://{WORKING_HOSTNAME}:{FASTAPI_INFERENCE_PORT}/health")
            response = httpx.get(f"http://{WORKING_HOSTNAME}:{FASTAPI_INFERENCE_PORT}/health", timeout=5)
            http_elapsed = time.time() - http_start
            
            print(f"✅ HTTP health check successful: {response.status_code} ({http_elapsed:.2f}s)")
            print(f"✅ Server response: {response.text[:200]}")
        except Exception as http_err:
            print(f"⚠️ HTTP health check failed: {http_err}")
            print(f"ℹ️ TCP connection works but HTTP request failed - server might still be initializing")
    else:
        server_wait.complete(f"WARNING: Could not verify FastAPI server is running")
        print(f"⚠️ WARNING: Could not verify FastAPI server is running on port {FASTAPI_INFERENCE_PORT}")
        print(f"ℹ️ The Inference UI may not be able to connect to the server")
        
        # Print detailed failure information
        print("\n🔍 CONNECTION FAILURE DETAILS:")
        for i, (host, error) in enumerate(connection_errors[-3:]):  # Show last 3 errors
            print(f"  Attempt {i+1}: {host} - {error}")
            
        print(f"\nℹ️ Will continue anyway, but you may need to start the server manually:")
    
    components_progress.update(message="Starting Inference UI")
    # Use direct Python command for all platforms - no batch files
    inference_cmd = [
        "python",
        "src/lora_training_pipeline/inference/gradio_inference.py",
    ]
    
    # Create environment with port configuration - Be very explicit with URLs
    # Use the working hostname if we found one during server verification
    api_hostname = getattr(globals(), 'WORKING_HOSTNAME', 'localhost')
    
    print(f"\n" + "="*80)
    print(f"SETTING UP INFERENCE UI ENVIRONMENT")
    print("="*80)
    print(f"Using hostname for FastAPI server: {api_hostname}")
    print(f"Using port for FastAPI server: {FASTAPI_INFERENCE_PORT}")
    print(f"Using port for Inference UI: {INFERENCE_UI_PORT}")
    
    # Construct the API URL with the working hostname
    api_url = f"http://{api_hostname}:{FASTAPI_INFERENCE_PORT}"
    print(f"Constructed API URL: {api_url}")
    
    inference_ui_env = {
        "PROCESS_NAME": "GradioInferenceUI",
        "GRADIO_INFERENCE_PORT": str(INFERENCE_UI_PORT),  # The UI's own port
        "GRADIO_PORT": str(INFERENCE_UI_PORT),  # Generic port variable (for compatibility)
        "FASTAPI_INFERENCE_URL": api_url,  # Explicit URL format with working hostname
        "INFERENCE_API_URL": api_url,  # Redundant for safety
        "FASTAPI_INFERENCE_PORT": str(FASTAPI_INFERENCE_PORT),  # Explicit port number
        "FASTAPI_URL": api_url,  # Another common name
        "FASTAPI_HOST": api_hostname,  # Add the hostname explicitly
        "DEBUG_SERVER_CONNECTION": "true",  # Enable extra debug info in the UI
        "PYTHONUNBUFFERED": "1",  # Force unbuffered output for better debugging
        "DEBUG_LEVEL": "DEBUG",  # Set debug level to maximum
        "INFERENCE_DEBUG": "true",  # Custom debug flag
        "GRADIO_UI_DEBUG": "true"  # Custom UI debug flag
    }
    
    # Print detailed debug info about environment variables
    print("\n" + "="*80)
    print("INFERENCE UI ENVIRONMENT VARIABLES")
    print("="*80)
    for key, value in inference_ui_env.items():
        print(f"{key}: {value}")
    print("="*80)
    
    # Before starting the UI, verify that the server's health endpoint is responding
    # This is a deeper verification than just checking if the port is open
    print("\n" + "="*80)
    print("VERIFYING FASTAPI SERVER HEALTH ENDPOINT")
    print("="*80)
    
    try:
        import httpx
        
        health_url = f"{api_url}/health"
        print(f"Testing health endpoint: {health_url}")
        
        try:
            health_start = time.time()
            health_response = httpx.get(health_url, timeout=5)
            health_time = time.time() - health_start
            
            if health_response.status_code == 200:
                print(f"✅ Health check successful: HTTP {health_response.status_code} ({health_time:.2f}s)")
                
                # Try to parse the response as JSON for more info
                try:
                    response_data = health_response.json()
                    print(f"Server details:")
                    print(f"- Server status: {response_data.get('status', 'unknown')}")
                    
                    # Extract server info if available
                    server_info = response_data.get('server_info', {})
                    if server_info:
                        print(f"- Python version: {server_info.get('python_version', 'unknown')}")
                        print(f"- Model loaded: {server_info.get('model_loaded', 'unknown')}")
                        print(f"- Process memory: {server_info.get('process_memory_mb', 'unknown')} MB")
                        print(f"- Server PID: {server_info.get('pid', 'unknown')}")
                except Exception as json_err:
                    print(f"Could not parse JSON response: {json_err}")
                    print(f"Raw response: {health_response.text[:200]}")
            else:
                print(f"⚠️ Health check returned non-200 status: {health_response.status_code}")
                print(f"Response: {health_response.text[:200]}")
        except httpx.TimeoutException:
            print(f"⚠️ Health check timed out after 5 seconds")
            print(f"The server might still be initializing")
        except httpx.RequestError as req_err:
            print(f"⚠️ Health check request failed: {req_err}")
            print(f"The server might be running but not fully initialized")
    except Exception as e:
        print(f"⚠️ Could not perform health check: {e}")
        print(f"Will continue without health verification")
    
    print("="*80)
    
    # Now check the model status endpoint
    print("\n" + "="*80)
    print("CHECKING SERVER MODEL STATUS")
    print("="*80)
    
    try:
        import httpx
        
        model_url = f"{api_url}/model-info"
        print(f"Testing model status endpoint: {model_url}")
        
        try:
            model_start = time.time()
            model_response = httpx.get(model_url, timeout=5)
            model_time = time.time() - model_start
            
            if model_response.status_code == 200:
                print(f"✅ Model info check successful: HTTP {model_response.status_code} ({model_time:.2f}s)")
                
                # Try to parse the response as JSON for more info
                try:
                    model_data = model_response.json()
                    print(f"Model details:")
                    print(f"- Model loaded: {model_data.get('model_loaded', False)}")
                    print(f"- Model path: {model_data.get('model_path', 'unknown')}")
                    print(f"- Model status: {model_data.get('status', 'unknown')}")
                except Exception as json_err:
                    print(f"Could not parse model JSON response: {json_err}")
                    print(f"Raw response: {model_response.text[:200]}")
            else:
                print(f"⚠️ Model check returned non-200 status: {model_response.status_code}")
                print(f"Response: {model_response.text[:200]}")
        except Exception as model_err:
            print(f"⚠️ Model status check failed: {model_err}")
    except Exception as e:
        print(f"⚠️ Could not perform model check: {e}")
    
    print("="*80)
    
    # Before starting a new UI, check if we already have a running instance
    existing_inference_ui = None

    # First check if we have a PID file
    if INFERENCE_UI_PID_FILE.exists():
        try:
            ui_pid = int(INFERENCE_UI_PID_FILE.read_text().strip())
            print(f"Found Inference UI PID file with PID: {ui_pid}")
            
            # Check if this process is still running
            try:
                if PSUTIL_AVAILABLE:
                    import psutil  # Double-check import
                    process = psutil.Process(ui_pid)
                    if process.is_running():
                        # Verify it's actually our UI process
                        cmdline = ' '.join(process.cmdline()) if process.cmdline() else ''
                        if 'gradio_inference.py' in cmdline:
                            print(f"✅ Found existing Inference UI process: PID={ui_pid}")
                            existing_inference_ui = process
                        else:
                            print(f"⚠️ Process {ui_pid} exists but is not the Inference UI: {cmdline[:100]}")
                    else:
                        print(f"⚠️ Process {ui_pid} from PID file is not running")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"Process {ui_pid} from PID file doesn't exist or can't be accessed")
            except Exception as proc_check_err:
                print(f"Error checking process: {proc_check_err}")
                
        except Exception as e:
            print(f"Error reading Inference UI PID file: {e}")

    # If no process found from PID file, scan for one
    if not existing_inference_ui and 'psutil' in sys.modules:
        print("Scanning for existing Inference UI processes...")
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else ''
                    if 'gradio_inference.py' in cmdline:
                        print(f"Found potential Inference UI: PID={proc.pid}")
                        
                        # Check if it's responsive
                        try:
                            proc.cpu_percent()  # Simple test to see if process responds
                            print(f"✅ Process {proc.pid} is responsive")
                            existing_inference_ui = proc
                            
                            # Update PID file
                            INFERENCE_UI_PID_FILE.write_text(str(proc.pid))
                            print(f"✓ Updated Inference UI PID file with found process: {proc.pid}")
                            break
                        except Exception as resp_err:
                            print(f"[DEBUG] Process responsiveness check error type: {type(resp_err).__name__}")
                            print(f"[DEBUG] Process responsiveness check error details: {resp_err}")
                            print(f"Process {proc.pid} is not responsive, will not use it")
                except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                    continue
        except Exception as scan_err:
            print(f"Error scanning for processes: {scan_err}")

    # Start a new process if needed
    if existing_inference_ui:
        print(f"Reusing existing Inference UI process (PID={existing_inference_ui.pid})")
        print(f"✅ Inference UI already running (port: {INFERENCE_UI_PORT})")
        print(f"ℹ️ The Inference UI is connected to FastAPI at: {api_url}")
    else:
        print("Starting new Inference UI process")
        inference_ui_process = start_process(inference_cmd, env=inference_ui_env)
        
        # Record the PID for future reference
        if inference_ui_process and hasattr(inference_ui_process, 'pid'):
            try:
                INFERENCE_UI_PID_FILE.write_text(str(inference_ui_process.pid))
                print(f"✓ Recorded Inference UI PID: {inference_ui_process.pid} in file: {INFERENCE_UI_PID_FILE}")
            except Exception as pid_error:
                print(f"❌ Error recording PID: {pid_error}")
        else:
            print("⚠️ WARNING: Could not get PID for Inference UI process")
            
        print(f"✅ Inference UI started (port: {INFERENCE_UI_PORT})")
        print(f"ℹ️ The Inference UI will connect to FastAPI at: {api_url}")
    
    # Complete component initialization
    components_progress.complete("All components started successfully")
    
    print("\n" + "="*80)
    print("✅ PIPELINE STATUS: All components started successfully")
    print("ℹ️ Data Collection UI:")
    print(f"  - Local:   http://localhost:{DATA_COLLECTION_PORT}")
    print(f"  - Network: http://<your-ip-address>:{DATA_COLLECTION_PORT}")
    print("  - Public:  Check console output for Gradio public URL")
    print("ℹ️ Inference UI:")
    print(f"  - Local:   http://localhost:{INFERENCE_UI_PORT}")
    print(f"  - Network: http://<your-ip-address>:{INFERENCE_UI_PORT}")
    print("  - Public:  Check console output for Gradio public URL")
    print("ℹ️ FastAPI Inference Server:")
    print(f"  - Local:   http://localhost:{FASTAPI_INFERENCE_PORT}")
    print(f"  - Network: http://<your-ip-address>:{FASTAPI_INFERENCE_PORT}")
    print(f"  - Health:  http://localhost:{FASTAPI_INFERENCE_PORT}/health")
    print(f"  - Status:  http://localhost:{FASTAPI_INFERENCE_PORT}/model-info")
    print("ℹ️ Pipeline Dashboard:")
    print(f"  - Local:   http://localhost:{DASHBOARD_PORT}")
    print(f"  - Network: http://<your-ip-address>:{DASHBOARD_PORT}")
    print("ℹ️ NOTE: All services are accessible from the internet")
    print("="*80 + "\n")

    # Define function to check for new data points with progressive thresholds
    def check_data_threshold(base_threshold=10, check_interval=60):
        """
        Checks if the collected data has reached the threshold for training,
        with progressive thresholds based on training cycle count.
        
        Args:
            base_threshold: Base threshold for the first training cycle
            check_interval: Seconds between checks
        """
        from src.lora_training_pipeline.utils.helpers import ProgressTracker, log_pending_error
        import pandas as pd
        import json
        from pathlib import Path
        import subprocess
        import time
        
        # Check for any existing active training processes from previous sessions
        active_training_process = None
        active_training_pid = None
        try:
            import psutil
            
            print("\n" + "="*80)
            print("CHECKING FOR ACTIVE TRAINING PROCESSES")
            print("="*80)
            
            # Load training cycle info to detect active training
            training_info = {}
            try:
                TRAINING_CYCLE_FILE = Path("./training_cycle_info.json")
                if TRAINING_CYCLE_FILE.exists():
                    with open(TRAINING_CYCLE_FILE, 'r') as f:
                        training_info = json.load(f)
                    print(f"Training cycle info loaded:")
                    print(f"- Cycle count: {training_info.get('cycle_count', 0)}")
                    print(f"- Last training time: {training_info.get('last_training_time', 'unknown')}")
                    print(f"- Valid data points: {training_info.get('total_valid_data_points', 0)}")
            except Exception as e:
                print(f"⚠️ Error loading training cycle info: {e}")
                training_info = {}
            
            # Convert last_training_time to epoch time if it's a string
            last_training_time = training_info.get('last_training_time', 0)
            if isinstance(last_training_time, str):
                try:
                    # Convert ISO timestamp to epoch seconds
                    last_training_time = datetime.datetime.fromisoformat(last_training_time).timestamp()
                except Exception as time_conv_err:
                    print(f"[DEBUG] Training time conversion error type: {type(time_conv_err).__name__}")
                    print(f"[DEBUG] Training time conversion error details: {time_conv_err}")
                    print(f"[DEBUG] Invalid training time format: {last_training_time}")
                    last_training_time = 0
                
            # Look for active training processes
            for proc in psutil.process_iter(['pid', 'cmdline', 'name', 'create_time']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    cmdline_str = ' '.join([str(c) for c in cmdline]) if cmdline else ''
                    proc_name = proc.info.get('name', '')
                    
                    # Skip if no command line info
                    if not cmdline_str:
                        continue
                    
                    # Check if this is a training process
                    if 'zenml_pipeline.py' in cmdline_str and 'python' in proc_name.lower():
                        proc_age = datetime.datetime.now().timestamp() - proc.info.get('create_time', 0)
                        
                        # Check if the process was created after the last training time with a 1-minute buffer
                        is_recent = (proc.info.get('create_time', 0) > last_training_time - 60)  
                        
                        # If it's recent, it's likely our active training process
                        if is_recent:
                            active_training_process = proc
                            active_training_pid = proc.pid
                            print(f"✅ Found active training process: PID={proc.pid}, Age={proc_age:.1f} seconds")
                            print(f"Command: {cmdline_str[:100]}")
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Report on active training status
            if active_training_process:
                print(f"🔄 Resuming monitoring of active training process (PID: {active_training_pid})")
                print(f"This training process will be monitored for completion")
                
                # Schedule a monitoring function for this existing process
                def monitor_existing_training_completion():
                    print(f"\n🔄 Monitoring existing training process (PID: {active_training_pid})...")
                    while True:
                        try:
                            # Check for signals first - CRITICAL for CTRL-C responsiveness
                            if process_signal():
                                print("[DEBUG] Signal processed in training monitoring loop, exiting")
                                break
                            
                            # Check if process is still running
                            if not psutil.pid_exists(active_training_pid):
                                print(f"\n🔄 Training process (PID: {active_training_pid}) has completed")
                                # Process has completed, check if it was successful by looking at training cycle info
                                try:
                                    current_info = {}
                                    if TRAINING_CYCLE_FILE.exists():
                                        with open(TRAINING_CYCLE_FILE, 'r') as f:
                                            current_info = json.load(f)
                                    
                                    if current_info.get('cycle_count', 0) > training_info.get('cycle_count', 0):
                                        print("\n" + "="*80)
                                        print("✅ RECOVERED TRAINING CYCLE COMPLETED SUCCESSFULLY")
                                        print(f"Previous cycle count: {training_info.get('cycle_count', 0)}")
                                        print(f"Current cycle count: {current_info.get('cycle_count', 0)}")
                                        print("="*80 + "\n")
                                    else:
                                        print("\n" + "="*80)
                                        print("⚠️ RECOVERED TRAINING PROCESS COMPLETED BUT CYCLE COUNT UNCHANGED")
                                        print("This may indicate the training process failed or was incomplete")
                                        print("="*80 + "\n")
                                except Exception as e:
                                    print(f"⚠️ Error checking training cycle completion: {e}")
                                
                                # Exit the monitoring loop
                                break
                            
                            # Wait before checking again with signal checking
                            for _ in range(10):  # 10 * 0.5s = 5s total
                                # Check for signals before sleeping - CRITICAL for CTRL-C responsiveness
                                if process_signal():
                                    print("[DEBUG] Signal processed in training monitoring sleep, exiting")
                                    return
                                
                                # Check shutdown flag for immediate exit
                                if shutdown_requested:
                                    print("[DEBUG] Shutdown requested in training monitoring sleep, exiting immediately")
                                    return
                                
                                time.sleep(0.5)
                        except Exception as e:
                            print(f"⚠️ Error monitoring training process: {e}")
                            break
                
                # Start monitoring in a background thread
                import threading
                monitoring_thread = threading.Thread(target=monitor_existing_training_completion)
                monitoring_thread.daemon = True
                monitoring_thread.start()
            else:
                print("✅ No active training processes found from previous sessions")
            
            print("="*80)
        except ImportError:
            print("ℹ️ psutil not available - cannot check for active training processes")
        except Exception as e:
            print(f"⚠️ Error checking for active training processes: {e}")
            import traceback
            print(traceback.format_exc())
        
        # Define trigger to clean data automatically when new data points arrive
        def trigger_data_cleaning():
            """Triggers the data cleaning stage when new data points arrive."""
            print("\n" + "="*80)
            print("AUTO-TRIGGERING DATA CLEANING STAGE")
            print("="*80)
            
            try:
                # Import the clean_and_filter_data step
                from src.lora_training_pipeline.data_cleaning.clean_filter import clean_and_filter_data
                
                # Import the time module for throttling
                import time
                
                # Check if we have unprocessed files that need cleaning
                try:
                    import json
                    from pathlib import Path
                    
                    # Count files to determine if cleaning is needed
                    DATA_DIR = Path("./data")
                    
                    # Count original files
                    original_files = list(DATA_DIR.glob(f"{DATASET_NAME}_{VERSION}_original_*.parquet"))
                    original_count = len(original_files)
                    
                    # Count valid and invalid files
                    valid_dir = DATA_DIR / "valid"
                    valid_files = []
                    if valid_dir.exists():
                        valid_files = list(valid_dir.glob(f"{DATASET_NAME}_{VERSION}_valid_*.parquet"))
                    valid_count = len(valid_files)
                    
                    rejected_dir = DATA_DIR / "rejected"
                    invalid_files = []
                    if rejected_dir.exists():
                        invalid_files = list(rejected_dir.glob(f"{DATASET_NAME}_{VERSION}_invalid_*.parquet"))
                    invalid_count = len(invalid_files)
                    
                    # Check if we have unprocessed files
                    processed_count = valid_count + invalid_count
                    has_unprocessed_files = original_count > processed_count
                    
                    if has_unprocessed_files:
                        print(f"✅ Found {original_count - processed_count} unprocessed files that need validation")
                        print(f"✅ Original: {original_count}, Processed: {processed_count} (Valid: {valid_count}, Invalid: {invalid_count})")
                        print(f"✅ Proceeding with cleaning operation regardless of cooldown")
                        # Skip the cooldown check since we have unprocessed files
                    else:
                        # If no unprocessed files, check the cooldown period
                        checkpoint_file = Path("./data/validation_checkpoint.json")
                        cleaning_cooldown = 300  # 5 minutes minimum between cleaning runs
                        
                        if checkpoint_file.exists():
                            with open(checkpoint_file, 'r') as f:
                                checkpoint = json.load(f)
                                last_reprocessed = checkpoint.get("last_reprocessed", 0)
                                current_time = int(time.time())
                                time_since_last_clean = current_time - last_reprocessed
                                
                                if time_since_last_clean < cleaning_cooldown:
                                    print(f"⚠️ No new files to process, and cleaning performed recently ({time_since_last_clean}s ago)")
                                    print(f"⚠️ All files already processed (Original: {original_count}, Processed: {processed_count})")
                                    print(f"⚠️ Skipping redundant cleaning operation to prevent endless revalidation")
                                    print(f"⚠️ Next revalidation will be allowed in {cleaning_cooldown - time_since_last_clean}s")
                                    return False
                except Exception as e:
                    print(f"⚠️ Warning: Could not check unprocessed files: {e}")
                    # Continue anyway since this is just a safeguard
                
                # Check if we should run cleaning at all or just query for current state
                # This will fetch the checkpoint but not modify it
                try:
                    from src.lora_training_pipeline.data_cleaning.clean_filter import load_checkpoint
                    checkpoint = load_checkpoint()
                    
                    # Get current time for comparing cooldown periods
                    current_time = int(time.time())
                    reprocessing_cooldown = 300  # 5 minutes between full reprocessing
                    
                    # Check if all files are already processed and we're in cooldown
                    processed_files_set = set(checkpoint.get("processed_files", []))
                    original_files_on_disk = [f.name for f in Path("./data").glob(f"{DATASET_NAME}_{VERSION}_original_*.parquet")]
                    unprocessed_count = len([f for f in original_files_on_disk if f not in processed_files_set])
                    time_since_last_reprocessed = current_time - checkpoint.get("last_reprocessed", 0)
                    
                    # Skip triggering cleaning if:
                    # 1. No unprocessed files AND 
                    # 2. We're in cooldown period AND
                    # 3. Force reprocess is not set
                    force_reprocess = os.environ.get("FORCE_REPROCESS", "").lower() == "true"
                    
                    if (unprocessed_count == 0 and 
                        time_since_last_reprocessed < reprocessing_cooldown and 
                        not force_reprocess):
                        
                        print("\n" + "="*80)
                        print("⚠️ DATA STATE UNCHANGED: Skipping redundant cleaning operation")
                        print(f"ℹ️ All files already processed ({len(processed_files_set)} of {len(original_files_on_disk)})")
                        print(f"ℹ️ Last cleaning was {time_since_last_reprocessed} seconds ago (cooldown: {reprocessing_cooldown}s)")
                        print(f"ℹ️ To force cleaning set FORCE_REPROCESS=true")
                        
                        # Return current state instead of triggering cleaning
                        valid_dir = Path("./data/valid")
                        valid_files = []
                        if valid_dir.exists():
                            valid_files = list(valid_dir.glob(f"{DATASET_NAME}_{VERSION}_valid_*.parquet"))

                        # Get dataset_ok status from existing files
                        min_required = 10  # Minimum required data points 
                        dataset_ok = len(valid_files) >= min_required
                        
                        print(f"ℹ️ Current status: {len(valid_files)} valid files, threshold is {min_required}")
                        print(f"ℹ️ Dataset sufficient for training: {dataset_ok}")
                        print("="*80 + "\n")
                        
                        # Return current state without triggering any cleaning
                        return dataset_ok
                except Exception as check_error:
                    print(f"⚠️ Error checking data state: {check_error}")
                    print(f"ℹ️ Will proceed with normal cleaning process")
                
                # If we reach here, we should proceed with normal cleaning
                print(f"ℹ️ Proceeding with data cleaning process")
                
                # Set environment variable to ensure inference server is preserved
                os.environ["STOP_INFERENCE_ON_EXIT"] = "false"
                
                # Call the function with disable_cache=True to force reprocessing 
                # but only if enough time has passed since the last run
                result = clean_and_filter_data(disable_cache=True)
                
                # Handle the three return values
                clean_data, dataset_ok, training_file_path = result
                
                print(f"✅ Data cleaning completed")
                print(f"✅ Valid data points: {len(clean_data) if not clean_data.empty else 0}")
                print(f"✅ Dataset sufficient for training: {dataset_ok}")
                print(f"✅ Training file path: {training_file_path if dataset_ok else 'None - insufficient data'}")
                print("="*80 + "\n")
                
                return dataset_ok
            except Exception as e:
                print(f"❌ ERROR: Failed to trigger data cleaning: {e}")
                log_pending_error(f"Auto-trigger data cleaning failed: {e}")
                print("="*80 + "\n")
                return False
        
        DATA_DIR = Path("./data")
        DATASET_NAME = "user_data_ui"
        VERSION = "v1"
        TRAINING_CYCLE_FILE = Path("./training_cycle_info.json")
        
        # Create and load training cycle info
        def load_training_cycle_info():
            if not TRAINING_CYCLE_FILE.exists():
                # Initialize with default values
                cycle_info = {
                    "cycle_count": 0,  # Start at 0, first training will be cycle 1
                    "total_valid_data_points": 0,
                    "last_training_time": 0,
                    "last_valid_count": 0,
                    "total_original_files": 0,
                    "total_valid_files": 0,
                    "total_invalid_files": 0,
                    "last_scan_time": 0
                }
                save_training_cycle_info(cycle_info)
                return cycle_info
            
            try:
                with open(TRAINING_CYCLE_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Could not read training cycle file: {e}")
                log_pending_error(f"Could not read training cycle file: {e}")
                # Return default values as fallback
                return {
                    "cycle_count": 0,
                    "total_valid_data_points": 0,
                    "last_training_time": 0,
                    "last_valid_count": 0,
                    "total_original_files": 0,
                    "total_valid_files": 0,
                    "total_invalid_files": 0,
                    "last_scan_time": 0
                }
        
        def save_training_cycle_info(cycle_info):
            try:
                with open(TRAINING_CYCLE_FILE, "w") as f:
                    json.dump(cycle_info, f, indent=2)
            except Exception as e:
                print(f"⚠️ Warning: Could not save training cycle file: {e}")
                log_pending_error(f"Could not save training cycle file: {e}")
        
        # Get initial training cycle info
        cycle_info = load_training_cycle_info()
        MIN_TRAINING_INTERVAL = 1800  # Minimum 30 minutes between training runs
        
        # Calculate the current threshold based on cycle count
        def calculate_threshold(cycle_count, base):
            # For cycle 0 (first training), use 1x base threshold
            # For cycle 1 (second training), use 2x base threshold, etc.
            multiplier = cycle_count + 1  # +1 because cycle_count starts at 0
            return multiplier * base
        
        monitor_tracker = ProgressTracker("Data Monitor")
        monitor_tracker.start("Starting data monitoring loop")
        
        # Count only valid data points
        def count_valid_data_points():
            # First count all collected data
            all_files = list(DATA_DIR.glob(f"{DATASET_NAME}_{VERSION}_original_*.parquet"))
            if not all_files:
                return 0, 0  # No files found
                
            try:
                # Read and concatenate all files
                df_list = []
                for file in all_files:
                    try:
                        df = pd.read_parquet(file)
                        df_list.append(df)
                    except Exception as e:
                        print(f"⚠️ Warning: Could not read file {file}: {e}")
                        continue
                
                if not df_list:
                    return 0, 0  # No valid files
                    
                # Get total collected data count
                collected_df = pd.concat(df_list)
                total_collected = len(collected_df)
                
                # Only look for validated data in new valid_*.parquet files in the dedicated directory
                valid_files = []
                
                # Look ONLY in the valid directory - this is the source of truth
                valid_dir = DATA_DIR / "valid"
                if valid_dir.exists():
                    new_style_files = list(valid_dir.glob(f"{DATASET_NAME}_{VERSION}_valid_*.parquet"))
                    if new_style_files:
                        valid_files.extend(new_style_files)
                        print(f"Found {len(new_style_files)} valid data files in {valid_dir}")
                
                # Check for consolidated file as a fallback ONLY if no valid files found
                # or if explicitly checking for consolidated results
                if not valid_files:
                    consolidated_files = list(DATA_DIR.glob(f"{DATASET_NAME}_{VERSION}_consolidated_*.parquet"))
                    if consolidated_files:
                        # Sort by timestamp to get the newest one
                        consolidated_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                        valid_files.append(consolidated_files[0])  # Use only the most recent consolidated file
                        print(f"Using consolidated data file: {consolidated_files[0].name}")
                
                if not valid_files:
                    print(f"⚠️ Warning: No valid data files found in any location")
                    return total_collected, 0  # No valid files
                
                # Read valid files
                valid_df_list = []
                for file in valid_files:
                    try:
                        df = pd.read_parquet(file)
                        valid_df_list.append(df)
                        print(f"📊 Reading valid data from: {file.name} ({len(df)} data points)")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not read valid file {file}: {e}")
                        continue
                
                if not valid_df_list:
                    return total_collected, 0
                    
                # Get total valid data count
                valid_df = pd.concat(valid_df_list)
                total_valid = len(valid_df)
                
                return total_collected, total_valid
                
            except Exception as e:
                print(f"⚠️ Warning: Error counting data points: {e}")
                log_pending_error(f"Error counting data points: {e}")
                return 0, 0
        
        # Print threshold info
        current_threshold = calculate_threshold(cycle_info["cycle_count"], base_threshold)
        print("\n" + "="*80)
        print("DATA MONITORING: Starting automatic data threshold monitor with progressive thresholds")
        print(f"ℹ️ Will check for new data every {check_interval} seconds")
        print(f"ℹ️ Current training cycle: {cycle_info['cycle_count']}")
        print(f"ℹ️ Base threshold: {base_threshold} valid data points")
        print(f"ℹ️ Current threshold: {current_threshold} valid data points ({cycle_info['cycle_count'] + 1}x base threshold)")
        print(f"ℹ️ Last recorded valid data points: {cycle_info['total_valid_data_points']}")
        print(f"ℹ️ Training cycle data stored in: {TRAINING_CYCLE_FILE}")
        print("="*80 + "\n")
        
        try:
            while True:
                try:
                    # Check for signals first - CRITICAL for CTRL-C responsiveness
                    if process_signal():
                        print("[DEBUG] Signal processed in main monitoring loop, exiting")
                        break
                    
                    # Also check global signal flag for immediate response
                    if signal_received:
                        print("[DEBUG] Global signal flag detected, processing signal immediately")
                        if process_signal():
                            break
                    
                    # Check shutdown flag for immediate exit
                    if shutdown_requested:
                        print("[DEBUG] Shutdown requested flag detected, exiting immediately")
                        sys.exit(0)
                    
                    # Add periodic status debugging
                    loop_iteration = getattr(main, '_loop_iteration', 0) + 1
                    main._loop_iteration = loop_iteration
                    if loop_iteration % 10 == 0:  # Every 10th iteration
                        print(f"[DEBUG] Main monitoring loop iteration {loop_iteration} - System status OK")
                        print(f"[DEBUG] Processes count: {len(processes)}")
                        print(f"[DEBUG] Signal queue size: {signal_queue.qsize()}")
                        print(f"[DEBUG] Current time: {datetime.datetime.now().isoformat()}")
                    
                    # Count current data points (both collected and valid)
                    total_collected, total_valid = count_valid_data_points()
                    
                    # Get current time and calculate time since last training
                    current_time = time.time()
                    time_since_last_training = current_time - cycle_info["last_training_time"]
                    
                    # Refresh current threshold calculation based on cycle count
                    current_threshold = calculate_threshold(cycle_info["cycle_count"], base_threshold)
                    
                    # Print status update
                    # Get file counts (each file = one data point)
                    original_files = list(DATA_DIR.glob(f"{DATASET_NAME}_{VERSION}_original_*.parquet"))
                    valid_dir = DATA_DIR / "valid"
                    valid_files = []
                    if valid_dir.exists():
                        valid_files = list(valid_dir.glob(f"{DATASET_NAME}_{VERSION}_valid_*.parquet"))
                    rejected_dir = DATA_DIR / "rejected"
                    invalid_files = []
                    if rejected_dir.exists():
                        invalid_files = list(rejected_dir.glob(f"{DATASET_NAME}_{VERSION}_invalid_*.parquet"))
                    
                    print(f"\n{'='*50}")
                    print(f"DATA MONITOR UPDATE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"MONITOR DATA SUMMARY:")
                    print(f"  - Total collected files/data points: {len(original_files)}")
                    print(f"  - Valid files/data points: {len(valid_files)}")
                    print(f"  - Invalid files/data points: {len(invalid_files)}")
                    print(f"  - Note: Each file contains exactly one data point")
                    print(f"TRAINING STATUS:")
                    print(f"  - Valid data points: {total_valid} / {current_threshold} required")
                    print(f"  - Training cycle: {cycle_info['cycle_count']} (multiplier: {cycle_info['cycle_count'] + 1}x)")
                    
                    # Check if we have more collected files than validated files
                    # This is a more reliable way to detect when we need to run cleaning
                    unprocessed_files_exist = len(original_files) > (len(valid_files) + len(invalid_files))
                    new_valid_detected = total_valid > cycle_info["last_valid_count"]
                    
                    if unprocessed_files_exist:
                        print(f"📈 Detected {len(original_files) - (len(valid_files) + len(invalid_files))} unprocessed data files")
                        # Only trigger data cleaning if it's enabled
                        auto_clean = os.environ.get("AUTO_CLEAN_ENABLED", "true").lower() == "true"
                        if auto_clean:
                            print(f"🔄 Triggering data cleaning for unprocessed files...")
                            trigger_data_cleaning()
                        else:
                            print(f"ℹ️ Auto-cleaning disabled. Use run_data_cleaning.py to manually clean data.")
                    elif new_valid_detected:
                        print(f"📈 New valid data detected: +{total_valid - cycle_info['last_valid_count']} data points")
                        print(f"ℹ️ All files appear to be processed already.")
                        
                    # Print cooldown info
                    if time_since_last_training < MIN_TRAINING_INTERVAL:
                        cooldown_remaining = MIN_TRAINING_INTERVAL - time_since_last_training
                        print(f"⏱️ Training cooldown: {cooldown_remaining:.0f} seconds remaining")
                    
                    print(f"{'='*50}\n")
                    
                    # Check for active ZenML pipeline or cleaning process
                    active_pipeline = False
                    try:
                        # Method 1: Check for training lock file
                        TRAINING_LOCK_FILE = Path("./.training_lock") 
                        if TRAINING_LOCK_FILE.exists():
                            active_pipeline = True
                            print(f"ℹ️ ZenML pipeline appears to be active (lock file exists)")
                            
                        # Method 2: Check for ZenML processes
                        try:
                            import psutil
                            # Track different types of processes
                            zenml_processes = []
                            gradio_processes = []
                            fastapi_processes = []
                            
                            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                                try:
                                    cmdline = proc.info.get('cmdline', [])
                                    # Handle case where cmdline might be None
                                    if cmdline is None:
                                        cmdline = []
                                    
                                    # Safely convert cmdline to string, handling None values
                                    try:
                                        cmd_str = ' '.join([str(c) for c in cmdline if c is not None])
                                    except (TypeError, AttributeError):
                                        # If cmdline is not iterable or has other issues, skip this process
                                        continue
                                    
                                    # Check for different types of processes
                                    if cmdline and 'zenml' in cmd_str:
                                        zenml_processes.append(proc)
                                    elif cmdline and 'gradio' in cmd_str:
                                        gradio_processes.append(proc)
                                    elif cmdline and ('uvicorn' in cmd_str and 'fastapi_inference' in cmd_str):
                                        fastapi_processes.append(proc)
                                except Exception as e:
                                    print(f"[DEBUG] Error processing process info: {e}")
                                    print(f"[DEBUG] Process info: {getattr(proc, 'info', 'No info available')}")
                                    continue
                                    
                            # Print status of all components
                            if zenml_processes:
                                active_pipeline = True
                                print(f"ℹ️ ZenML processes detected ({len(zenml_processes)} running)")
                                
                            # ENHANCED DEBUGGING: Analyze all processes with detailed info
                            print("\n" + "="*80)
                            print("DETAILED PROCESS ANALYSIS")
                            print("="*80)
                            
                            # First analyze Gradio processes to understand the UI backends
                            print("\n📊 GRADIO PROCESS ANALYSIS:")
                            
                            # Get current running processes by type using psutil
                            data_collection_processes = []
                            inference_ui_processes = []
                            unknown_processes = []
                            
                            # For backward compatibility
                            gradio_data_collection_count = 0
                            gradio_inference_count = 0
                            gradio_unknown_count = 0
                            
                            # More robust process identification
                            for proc in psutil.process_iter(['pid', 'cmdline', 'name', 'create_time']):
                                try:
                                    # Skip non-Python processes to avoid unnecessary checks
                                    if not 'python' in proc.name().lower():
                                        continue
                                    
                                    cmd = ' '.join([str(c) for c in proc.cmdline()]) if proc.cmdline() else ''
                                    if not cmd:
                                        continue
                                        
                                    # Get process environment if available
                                    try:
                                        proc_env = proc.environ()
                                    except (psutil.AccessDenied, Exception):
                                        proc_env = {}
                                    
                                    # First check PROCESS_NAME environment variable (most reliable)
                                    process_type = None
                                    if 'PROCESS_NAME' in proc_env:
                                        process_type = proc_env['PROCESS_NAME']
                                    
                                    # Then analyze command line
                                    if not process_type:
                                        if 'gradio_app.py' in cmd:
                                            process_type = 'GradioDataCollection'
                                        elif 'gradio_inference.py' in cmd:
                                            process_type = 'GradioInferenceUI'
                                    
                                    # Track processes by type
                                    if process_type == 'GradioDataCollection':
                                        data_collection_processes.append(proc)
                                        gradio_data_collection_count += 1
                                    elif process_type == 'GradioInferenceUI':
                                        inference_ui_processes.append(proc)
                                        gradio_inference_count += 1
                                    elif 'gradio' in cmd.lower():
                                        # Unknown Gradio process
                                        unknown_processes.append(proc)
                                        gradio_unknown_count += 1
                                except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                                    continue
                           
                            # Print findings about Gradio processes
                            print(f"ℹ️ Found {len(data_collection_processes)} Data Collection UI processes")
                            print(f"ℹ️ Found {len(inference_ui_processes)} Inference UI processes")
                            print(f"ℹ️ Found {len(unknown_processes)} unknown Gradio-related processes")
                            
                            # Print details for each process type
                            if data_collection_processes:
                                print("\nData Collection UI Processes:")
                                for i, proc in enumerate(data_collection_processes):
                                    try:
                                        # Get basic info
                                        p_pid = proc.pid if hasattr(proc, 'pid') else "unknown"
                                        
                                        # Get creation time
                                        try:
                                            create_time = proc.create_time()
                                            formatted_time = datetime.datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S')
                                        except Exception as time_err:
                                            print(f"[DEBUG] Data collection process create_time error type: {type(time_err).__name__}")
                                            print(f"[DEBUG] Data collection process create_time error details: {time_err}")
                                            print(f"[DEBUG] Failed to get create_time for process {p_pid}")
                                            formatted_time = "unknown"
                                        
                                        # Get command line
                                        try:
                                            cmd = ' '.join([str(c) for c in proc.cmdline()]) if proc.cmdline() else 'empty'
                                            cmd_preview = cmd[:100] + "..." if len(cmd) > 100 else cmd
                                        except Exception as cmd_err:
                                            print(f"[DEBUG] Process cmdline error type: {type(cmd_err).__name__}")
                                            print(f"[DEBUG] Process cmdline error details: {cmd_err}")
                                            cmd_preview = "unknown"
                                            
                                        # Try to get environment variables
                                        env_vars = []
                                        try:
                                            with proc.oneshot():
                                                # Get only relevant environment variables
                                                env_dict = proc.environ()
                                                for key in ['FASTAPI_INFERENCE_URL', 'INFERENCE_API_URL', 'GRADIO_PORT', 'PYTHONPATH']:
                                                    if key in env_dict:
                                                        env_vars.append(f"{key}={env_dict[key]}")
                                        except Exception as e:
                                            print(f"[DEBUG] Error accessing process environment: {e}")
                                            env_vars = ["Cannot access environment"]
                                            
                                        print(f"  Process {i+1}: PID={p_pid}")
                                        print(f"    Created: {formatted_time}")
                                        print(f"    Command: {cmd_preview}")
                                        print(f"    Environment: {', '.join(env_vars)}")
                                    except Exception as proc_err:
                                        print(f"  Process {i+1}: [Error: {proc_err}]")
                            
                            if inference_ui_processes:
                                print("\nInference UI Processes:")
                                for i, proc in enumerate(inference_ui_processes):
                                    try:
                                        # Get basic info
                                        p_pid = proc.pid if hasattr(proc, 'pid') else "unknown"
                                        
                                        # Get creation time
                                        try:
                                            create_time = proc.create_time()
                                            formatted_time = datetime.datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S')
                                        except Exception:
                                            formatted_time = "unknown"
                                        
                                        # Get command line
                                        try:
                                            cmd = ' '.join([str(c) for c in proc.cmdline()]) if proc.cmdline() else 'empty'
                                            cmd_preview = cmd[:100] + "..." if len(cmd) > 100 else cmd
                                        except Exception as cmd_err:
                                            print(f"[DEBUG] Process cmdline error type: {type(cmd_err).__name__}")
                                            print(f"[DEBUG] Process cmdline error details: {cmd_err}")
                                            cmd_preview = "unknown"
                                            
                                        # Try to get environment variables
                                        env_vars = []
                                        try:
                                            with proc.oneshot():
                                                # Get only relevant environment variables
                                                env_dict = proc.environ()
                                                for key in ['FASTAPI_INFERENCE_URL', 'INFERENCE_API_URL', 'GRADIO_PORT', 'PYTHONPATH']:
                                                    if key in env_dict:
                                                        env_vars.append(f"{key}={env_dict[key]}")
                                        except Exception as e:
                                            print(f"[DEBUG] Error accessing process environment: {e}")
                                            env_vars = ["Cannot access environment"]
                                            
                                        print(f"  Process {i+1}: PID={p_pid}")
                                        print(f"    Created: {formatted_time}")
                                        print(f"    Command: {cmd_preview}")
                                        print(f"    Environment: {', '.join(env_vars)}")
                                    except Exception as proc_err:
                                        print(f"  Process {i+1}: [Error: {proc_err}]")
                            
                            # Summary of Gradio processes
                            print(f"\n  Summary of Gradio processes:")
                            print(f"  - Data Collection UIs: {gradio_data_collection_count}")
                            print(f"  - Inference UIs: {gradio_inference_count}")
                            print(f"  - Unknown/Other: {gradio_unknown_count}")
                            # Check for excessive Gradio processes
                            if len(gradio_processes) > 2:
                                print(f"⚠️ Warning: More than 2 Gradio processes detected - this may indicate stale processes")
                                print(f"⚠️ Automatically cleaning up excess Gradio processes...")
                                # Automatically clean up without asking for user input
                                print(f"Starting automatic cleanup...")
                                # Group processes by type
                                data_collection_procs = []
                                inference_procs = []
                                unknown_procs = []
                                
                                # Add detailed debugging info for process inspection
                                print("\n🔍 DETAILED PROCESS INSPECTION:")
                                print(f"Total Gradio processes found: {len(gradio_processes)}")
                                
                                for i, p in enumerate(gradio_processes):
                                    try:
                                        # Get basic process info with error handling
                                        p_pid = p.pid if hasattr(p, 'pid') else "unknown"
                                        
                                        p_name = "unknown"
                                        try:
                                            p_name = p.name()
                                        except Exception as name_err:
                                            p_name = f"error:{name_err}"
                                            
                                        p_create_time = "unknown"
                                        try:
                                            p_create_time = datetime.datetime.fromtimestamp(p.create_time()).strftime('%H:%M:%S')
                                        except Exception as time_err:
                                            p_create_time = f"error:{time_err}"
                                        
                                        print(f"  Process {i+1}: PID={p_pid}, Name={p_name}, Created={p_create_time}")
                                        
                                        # Try to get command line
                                        cmd = "empty"
                                        try:
                                            cmd = ' '.join([str(c) for c in p.cmdline()]) if p.cmdline() else 'empty'
                                            cmd_preview = cmd[:100] + "..." if len(cmd) > 100 else cmd
                                            print(f"    Command: {cmd_preview}")
                                        except Exception as cmd_err:
                                            print(f"    ⚠️ Error accessing command: {cmd_err}")
                                            
                                        # Try to get process status
                                        try:
                                            p_status = p.status()
                                            print(f"    Status: {p_status}")
                                        except Exception as status_err:
                                            print(f"    Status: unknown ({status_err})")
                                        
                                        # Try to get memory info    
                                        try:
                                            p_memory = p.memory_info().rss / 1024 / 1024  # MB
                                            print(f"    Memory: {p_memory:.1f} MB")
                                        except Exception:
                                            print(f"    Memory: unknown")
                                            
                                        # Categorize the process
                                        if 'gradio_app.py' in cmd:
                                            data_collection_procs.append(p)
                                            print(f"    ✓ Categorized as: Data Collection UI")
                                        elif 'gradio_inference.py' in cmd:
                                            inference_procs.append(p)
                                            print(f"    ✓ Categorized as: Inference UI")
                                        else:
                                            unknown_procs.append(p)
                                            print(f"    ⚠️ Categorized as: Unknown Gradio-related process")
                                            
                                    except Exception as proc_err:
                                        print(f"  Process {i+1}: ❌ Critical inspection error: {proc_err}")
                                        try:
                                            unknown_procs.append(p)
                                            print(f"    Added to unknown processes list")
                                        except Exception as add_err:
                                            print(f"    ❌ Fatal error: Could not add to unknown list: {add_err}")
                                
                                print(f"Found: {len(data_collection_procs)} data collection, {len(inference_procs)} inference, {len(unknown_procs)} unknown")
                                
                                # Process cleanup - separated from the "if" conditions to ensure it always runs
                                # Clean up Data Collection processes if needed
                                if len(data_collection_procs) > 1:
                                    print("\n🔧 CLEANING UP DATA COLLECTION PROCESSES")
                                    # Sort by creation time (newest first)
                                    data_collection_procs.sort(key=lambda p: p.create_time() if hasattr(p, 'create_time') and callable(p.create_time) else 0, reverse=True)
                                    
                                    # Keep the newest one
                                    keep_proc = data_collection_procs[0]
                                    
                                    # Verify that the process is responsive
                                    proc_responsive = True
                                    try:
                                        # Simple check if process responds
                                        keep_proc.cpu_percent()
                                    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                                        proc_responsive = False
                                        
                                    if proc_responsive:
                                        print(f"✅ Keeping newest data collection process (PID={keep_proc.pid}, created: {time.ctime(keep_proc.create_time())})")
                                        
                                        # Update the PID file to reflect the process we're keeping
                                        try:
                                            DATA_COLLECTION_PID_FILE.write_text(str(keep_proc.pid))
                                            print(f"✓ Updated Data Collection UI PID file with kept process: {keep_proc.pid}")
                                        except Exception as pid_err:
                                            print(f"⚠️ Error updating PID file: {pid_err}")
                                        
                                        # Terminate all others
                                        for p in data_collection_procs[1:]:
                                            print(f"🔄 Terminating older data collection process {p.pid}...")
                                            try:
                                                p.terminate()
                                                gone, alive = psutil.wait_procs([p], timeout=5)
                                                if p in alive:
                                                    print(f"⚠️ Process {p.pid} did not terminate, using kill()")
                                                    p.kill()
                                            except Exception as term_err:
                                                print(f"⚠️ Error terminating process: {term_err}")
                                    else:
                                        print(f"⚠️ Newest process (PID={keep_proc.pid}) is not responsive!")
                                        # Try the next one
                                        if len(data_collection_procs) > 1:
                                            keep_proc = data_collection_procs[1]
                                            print(f"Trying next newest process (PID={keep_proc.pid})")
                                            try:
                                                # Check if it's responsive
                                                keep_proc.cpu_percent()
                                                print(f"✅ This process is responsive, keeping it")
                                                
                                                # Update the PID file to reflect the process we're keeping
                                                try:
                                                    DATA_COLLECTION_PID_FILE.write_text(str(keep_proc.pid))
                                                    print(f"✓ Updated Data Collection UI PID file with kept process: {keep_proc.pid}")
                                                except Exception as pid_err:
                                                    print(f"⚠️ Error updating PID file: {pid_err}")
                                                
                                                # Terminate all others
                                                for p in data_collection_procs[:1] + data_collection_procs[2:]:
                                                    print(f"🔄 Terminating data collection process {p.pid}...")
                                                    try:
                                                        p.terminate()
                                                        gone, alive = psutil.wait_procs([p], timeout=5)
                                                        if p in alive:
                                                            print(f"⚠️ Process {p.pid} did not terminate, using kill()")
                                                            p.kill()
                                                    except Exception as term_err:
                                                        print(f"⚠️ Error terminating process: {term_err}")
                                            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                                                print(f"⚠️ No responsive data collection processes found!")
                                                
                                # Clean up Inference UI processes - ALWAYS do this regardless of data collection process status
                                if len(inference_procs) > 1:
                                    print("\n🔧 CLEANING UP INFERENCE UI PROCESSES")
                                    # Sort by creation time (oldest first)
                                    inference_procs.sort(key=lambda p: p.create_time() if hasattr(p, 'create_time') and callable(p.create_time) else 0)
                                    
                                    # Find the first responsive process
                                    keep_proc = None
                                    for proc in inference_procs:
                                        try:
                                            # Check if process responds
                                            proc.cpu_percent()
                                            # If we reach here, process is responsive
                                            keep_proc = proc
                                            break
                                        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                                            continue
                                    
                                    # If no responsive process found, try to keep the oldest one anyway
                                    if keep_proc is None and inference_procs:
                                        keep_proc = inference_procs[0]
                                        print(f"⚠️ No responsive inference UI processes found, attempting to keep oldest (PID={keep_proc.pid})")
                                    
                                    # Verify that the kept process is responsive
                                    proc_responsive = False
                                    if keep_proc:
                                        try:
                                            # Double-check process responsiveness
                                            keep_proc.cpu_percent()
                                            proc_responsive = True
                                        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                                            proc_responsive = False
                                        
                                    if proc_responsive:
                                        print(f"✅ Keeping oldest responsive inference UI process (PID={keep_proc.pid}, created: {time.ctime(keep_proc.create_time())})")
                                        
                                        # Terminate all others except the one we're keeping
                                        for p in inference_procs:
                                            if p.pid != keep_proc.pid:
                                                print(f"🔄 Terminating other inference UI process {p.pid}...")
                                                try:
                                                    p.terminate()
                                                    gone, alive = psutil.wait_procs([p], timeout=5)
                                                    if p in alive:
                                                        print(f"⚠️ Process {p.pid} did not terminate, using kill()")
                                                        p.kill()
                                                except Exception as term_err:
                                                    print(f"⚠️ Error terminating process: {term_err}")
                                    else:
                                        print(f"⚠️ No responsive inference UI processes found!")
                                        # Try the next one
                                        if len(inference_procs) > 1:
                                            keep_proc = inference_procs[1]
                                            print(f"Trying next newest process (PID={keep_proc.pid})")
                                            try:
                                                # Check if it's responsive
                                                keep_proc.cpu_percent()
                                                print(f"✅ This process is responsive, keeping it")
                                                # Terminate all others
                                                for p in inference_procs[:1] + inference_procs[2:]:
                                                    print(f"🔄 Terminating inference UI process {p.pid}...")
                                                    try:
                                                        p.terminate()
                                                        gone, alive = psutil.wait_procs([p], timeout=5)
                                                        if p in alive:
                                                            print(f"⚠️ Process {p.pid} did not terminate, using kill()")
                                                            p.kill()
                                                    except Exception as term_err:
                                                        print(f"⚠️ Error terminating process: {term_err}")
                                            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                                                print(f"⚠️ No responsive inference UI processes found!")
                                                
                                # Terminate unknown Gradio processes
                                if unknown_procs:
                                    print("\n🔧 CLEANING UP UNKNOWN GRADIO PROCESSES")
                                    for p in unknown_procs:
                                        print(f"🔄 Terminating unknown Gradio process {p.pid}...")
                                        try:
                                            p.terminate()
                                            gone, alive = psutil.wait_procs([p], timeout=5)
                                            if p in alive:
                                                print(f"⚠️ Process {p.pid} did not terminate, using kill()")
                                                p.kill()
                                        except Exception as term_err:
                                            print(f"⚠️ Error terminating process: {term_err}")
                                            
                                # Create/update PID files for tracking UI processes
                                try:
                                    # Save the PID of the Data Collection UI for future reference
                                    if data_collection_procs:
                                        # Get the process we kept
                                        kept_dc_proc = data_collection_procs[0]
                                        # Use the global DATA_COLLECTION_PID_FILE constant
                                        DATA_COLLECTION_PID_FILE.write_text(str(kept_dc_proc.pid))
                                        print(f"✅ Saved Data Collection PID: {kept_dc_proc.pid}")
                                        
                                    # Save the PID of the Inference UI for future reference
                                    if inference_procs:
                                        # Get the process we kept
                                        kept_inf_proc = inference_procs[0]
                                        # Use the global INFERENCE_UI_PID_FILE constant
                                        INFERENCE_UI_PID_FILE.write_text(str(kept_inf_proc.pid))
                                        print(f"✅ Saved Inference UI PID: {kept_inf_proc.pid}")
                                except Exception as pid_write_err:
                                    print(f"⚠️ Error writing PID files: {pid_write_err}")
                                    
                                print(f"✅ Automatic cleanup complete")
                            
                            # Monitor FastAPI inference server
                            if fastapi_processes:
                                # Check for signals before doing potentially long analysis
                                if process_signal():
                                    print("[DEBUG] Signal processed before FastAPI analysis, skipping")
                                    break
                                    
                                print(f"\n📊 FASTAPI PROCESS ANALYSIS:")
                                print(f"ℹ️ FastAPI inference server detected ({len(fastapi_processes)} running)")
                                
                                # Enhanced FastAPI process analysis
                                for i, proc in enumerate(fastapi_processes):
                                    try:
                                        cmdline_info = proc.info.get('cmdline', [])
                                        if cmdline_info is None:
                                            cmdline_info = []
                                        cmd = ' '.join([str(c) for c in cmdline_info if c is not None])
                                        cmd_preview = cmd[:100] + "..." if len(cmd) > 100 else cmd
                                        
                                        # Get creation time and environment
                                        try:
                                            create_time = proc.create_time()
                                            formatted_time = datetime.datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S')
                                            
                                            # Try to get environment variables
                                            env_vars = []
                                            try:
                                                with proc.oneshot():
                                                    # Get only relevant environment variables
                                                    env_dict = proc.environ()
                                                    for key in ['FASTAPI_INFERENCE_PORT', 'LORA_MODEL_PATH', 'MODEL_UPDATE_SIGNAL_FILE']:
                                                        if key in env_dict:
                                                            env_vars.append(f"{key}={env_dict[key]}")
                                            except Exception as e:
                                                # Check if this is a process termination error
                                                if "no longer exists" in str(e) or "NoSuchProcess" in str(e):
                                                    print(f"[DEBUG] FastAPI process {proc.pid} terminated during analysis")
                                                    env_vars = ["Process terminated"]
                                                else:
                                                    print(f"[DEBUG] Error accessing FastAPI process environment: {e}")
                                                    env_vars = ["Cannot access environment"]
                                                
                                            # Try to get listening port
                                            port_info = "Unknown"
                                            try:
                                                with proc.oneshot():
                                                    connections = proc.connections('inet')
                                                    listening_ports = [conn.laddr.port for conn in connections if conn.status == 'LISTEN']
                                                    if listening_ports:
                                                        port_info = f"Listening on port(s): {', '.join(map(str, listening_ports))}"
                                            except Exception as e:
                                                # Check if this is a process termination error
                                                if "no longer exists" in str(e) or "NoSuchProcess" in str(e):
                                                    print(f"[DEBUG] FastAPI process {proc.pid} terminated during port analysis")
                                                    port_info = "Process terminated"
                                                else:
                                                    print(f"[DEBUG] Error determining FastAPI port info: {e}")
                                                    port_info = "Cannot determine port"
                                                
                                            print(f"  Process {i+1}: PID={proc.pid}")
                                            print(f"    Created: {formatted_time}")
                                            print(f"    Command: {cmd_preview}")
                                            print(f"    Port: {port_info}")
                                            print(f"    Environment: {', '.join(env_vars)}")
                                        except Exception as detail_err:
                                            print(f"  Process {i+1}: PID={proc.pid}")
                                            print(f"    Command: {cmd_preview}")
                                            print(f"    [Error accessing detailed process info: {detail_err}]")
                                    except Exception as proc_err:
                                        print(f"  Process {i+1}: [Error: {proc_err}]")
                                
                                # Check for multiple FastAPI processes
                                if len(fastapi_processes) > 1:
                                    print(f"\n⚠️ Warning: Multiple FastAPI processes detected - this may cause conflicts")
                                    print(f"⚠️ Only one FastAPI server should be running on port {FASTAPI_INFERENCE_PORT}")
                                
                                # Test connectivity to each FastAPI server to verify which one is working
                                print(f"\n📡 Testing FastAPI server connectivity:")
                                for i, proc in enumerate(fastapi_processes):
                                    try:
                                        pid = proc.pid
                                        # Try to get the port
                                        port = FASTAPI_INFERENCE_PORT  # Default
                                        try:
                                            with proc.oneshot():
                                                connections = proc.connections('inet')
                                                listening_ports = [conn.laddr.port for conn in connections if conn.status == 'LISTEN']
                                                if listening_ports:
                                                    port = listening_ports[0]
                                        except Exception as e:
                                            # Check if this is a process termination error
                                            if "no longer exists" in str(e) or "NoSuchProcess" in str(e):
                                                print(f"[DEBUG] FastAPI process {proc.pid} terminated during connectivity test")
                                            else:
                                                print(f"[DEBUG] Error getting port info for FastAPI process: {e}")
                                            pass
                                            
                                        # Test connection to the port
                                        import socket
                                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                        s.settimeout(1)
                                        result = s.connect_ex(('localhost', port))
                                        s.close()
                                        
                                        if result == 0:
                                            print(f"  ✅ Server {i+1} (PID {pid}) is accepting connections on port {port}")
                                            
                                            # Try to check health endpoint 
                                            try:
                                                import requests
                                                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                                                if response.status_code == 200:
                                                    print(f"  ✅ Health endpoint check succeeded: {response.status_code}")
                                                    if "model_loaded" in response.text:
                                                        if "true" in response.text.lower():
                                                            print(f"  ✅ Server has a model loaded")
                                                        else:
                                                            print(f"  ⚠️ Server is running but no model is loaded")
                                                else:
                                                    print(f"  ⚠️ Health endpoint returned non-200 status: {response.status_code}")
                                            except Exception as req_err:
                                                print(f"  ⚠️ Health endpoint check failed: {req_err}")
                                        else:
                                            print(f"  ❌ Server {i+1} (PID {pid}) is NOT accepting connections on port {port}")
                                    except Exception as conn_err:
                                        print(f"  ❌ Error testing server {i+1}: {conn_err}")
                            else:
                                print(f"\n❌ NO FASTAPI PROCESSES DETECTED - Inference server is not running")
                        except ImportError:
                            pass  # psutil not available
                        
                        # Method 3: Check for cleaning in progress
                        CLEANING_IN_PROGRESS_FILE = Path("./.cleaning_in_progress")
                        if CLEANING_IN_PROGRESS_FILE.exists():
                            active_pipeline = True
                            print(f"ℹ️ Data cleaning appears to be in progress (marker file exists)")
                            
                        if active_pipeline:
                            print(f"⚠️ Will not trigger new pipeline run while another is active")
                            # Skip to next monitoring iteration with signal checking
                            for _ in range(20):  # 20 * 0.5s = 10s total
                                # Check for signals before sleeping - CRITICAL for CTRL-C responsiveness
                                if process_signal():
                                    print("[DEBUG] Signal processed during monitoring retry delay, exiting")
                                    return
                                
                                # Check shutdown flag for immediate exit
                                if shutdown_requested:
                                    print("[DEBUG] Shutdown requested during monitoring retry delay, exiting immediately")
                                    sys.exit(0)
                                
                                time.sleep(0.5)
                            continue
                            
                    except Exception as e:
                        print(f"ℹ️ Error checking for active pipelines: {e}")
                    
                    # Check if we should trigger training:
                    # 1. We have enough valid data (current threshold)
                    # 2. We have new valid data since last training
                    # 3. Enough time has passed since last training
                    # 4. No active pipeline is running
                    if (total_valid >= current_threshold and 
                        total_valid > cycle_info["total_valid_data_points"] and 
                        time_since_last_training >= MIN_TRAINING_INTERVAL):
                        
                        print("\n" + "="*80)
                        print(f"🎯 PROGRESSIVE THRESHOLD REACHED: {total_valid} valid data points")
                        print(f"ℹ️ Current threshold: {current_threshold} (Cycle {cycle_info['cycle_count']}, {cycle_info['cycle_count'] + 1}x base threshold)")
                        
                        # Generate and print detailed training history report
                        print("\n" + "="*80)
                        print("📊 TRAINING HISTORY REPORT")
                        print("="*80)
                        print(f"Training cycles completed: {cycle_info['cycle_count']}")
                        
                        # Format the last training time nicely
                        if cycle_info['last_training_time'] > 0:
                            last_training_formatted = datetime.datetime.fromtimestamp(cycle_info['last_training_time']).strftime("%Y-%m-%d %H:%M:%S")
                            # Calculate time since last training
                            time_since_last = current_time - cycle_info['last_training_time']
                            if time_since_last < 60:
                                time_since_formatted = f"{time_since_last:.1f} seconds"
                            elif time_since_last < 3600:
                                time_since_formatted = f"{time_since_last / 60:.1f} minutes"
                            elif time_since_last < 86400:
                                time_since_formatted = f"{time_since_last / 3600:.1f} hours"
                            else:
                                time_since_formatted = f"{time_since_last / 86400:.1f} days"
                                
                            print(f"Last training time: {last_training_formatted} ({time_since_formatted} ago)")
                        else:
                            print(f"Last training time: Never")
                            
                        print(f"Last training data points: {cycle_info.get('total_valid_data_points', 0)}")
                        print(f"Current valid data points: {total_valid}")
                        print(f"New data since last training: {total_valid - cycle_info.get('total_valid_data_points', 0)}")
                        
                        # Check for any running training processes
                        training_in_progress = False
                        
                        # First check if we detected an active training process during recovery
                        # (this is a variable that was set at the beginning of check_data_threshold)
                        if 'active_training_process' in locals() and active_training_process is not None and active_training_pid is not None:
                            try:
                                if psutil.pid_exists(active_training_pid):
                                    training_in_progress = True
                                    print(f"\n⚠️ TRAINING ALREADY IN PROGRESS: Using recovered process (PID: {active_training_pid})")
                                    print(f"This process was detected during session recovery and is being monitored.")
                            except Exception as e:
                                print(f"⚠️ Error checking recovered training process: {e}")
                        
                        # Method 1: Check for training lock file
                        TRAINING_LOCK_FILE = Path("./.training_lock")
                        if TRAINING_LOCK_FILE.exists():
                            training_in_progress = True
                            print(f"ℹ️ Training lock file exists at {TRAINING_LOCK_FILE}")
                            print(f"\n⚠️ WARNING: Training lock file exists!")
                            print(f"⚠️ Created: {datetime.datetime.fromtimestamp(TRAINING_LOCK_FILE.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Method 2: Check for ZenML processes
                        try:
                            import psutil
                            zenml_processes = []
                            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                                try:
                                    cmdline = proc.info.get('cmdline', [])
                                    if cmdline is None:
                                        cmdline = []
                                    if cmdline and 'zenml' in ' '.join([str(c) for c in cmdline if c is not None]):
                                        zenml_processes.append(proc)
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    continue
                                    
                            if zenml_processes:
                                training_in_progress = True
                                print(f"\n⚠️ WARNING: {len(zenml_processes)} ZenML processes found running!")
                                for i, proc in enumerate(zenml_processes[:3]):  # Show first 3 processes
                                    try:
                                        cmd = ' '.join([str(c) for c in proc.info.get('cmdline', [])])
                                        print(f"⚠️ Process {i+1}: PID {proc.info.get('pid')} - {cmd[:80]}...")
                                    except Exception as e:
                                        print(f"⚠️ Process {i+1}: PID {proc.info.get('pid')} - (command details unavailable: {e})")
                                
                                if len(zenml_processes) > 3:
                                    print(f"⚠️ ... and {len(zenml_processes) - 3} more ZenML processes")
                        except ImportError:
                            print(f"\nℹ️ Note: Install psutil for better process detection: uv pip install psutil")
                            # Check if we can find ZenML processes another way
                            try:
                                # Try using subprocess to run a basic check
                                import subprocess
                                result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
                                if "zenml" in result.stdout:
                                    training_in_progress = True
                                    print(f"\n⚠️ WARNING: ZenML processes found running in 'ps aux' output!")
                            except Exception as e:
                                print(f"ℹ️ Could not check for ZenML processes without psutil: {e}")
                        except Exception as e:
                            print(f"ℹ️ Error checking for ZenML processes: {e}")
                            
                        # Only proceed if no training is in progress
                        if training_in_progress:
                            print("\n" + "="*80)
                            print("❌ TRAINING TRIGGER ABORTED: Another training run is already in progress")
                            print("ℹ️ Wait for the current training to complete before starting a new one")
                            print("ℹ️ If no training is actually running, the lock file may be stale.")
                            print("ℹ️ To fix this, delete the .training_lock file and try again.")
                            print("="*80 + "\n")
                            continue  # Skip to next iteration of the monitoring loop
                            
                        print("\n🚀 AUTOMATICALLY TRIGGERING TRAINING PIPELINE")
                        print("="*80 + "\n")
                        
                        # Launch the ZenML pipeline
                        pipeline_tracker = ProgressTracker("Training Trigger")
                        pipeline_tracker.start("Launching ZenML training pipeline")
                        
                        # Update the cycle info before training
                        cycle_info["last_training_time"] = current_time
                        cycle_info["last_valid_count"] = total_valid
                        save_training_cycle_info(cycle_info)
                        
                        # Use direct Python command to run ZenML pipeline
                        # Use simple "python" command - this is more reliable
                        # Set up environment for ZenML pipeline with MLflow integration
                        zenml_env = os.environ.copy()

                        # Set MLflow tracking URI if available - with enhanced debugging
                        print("\n[ZENML-SUBPROCESS-DEBUG] Setting up environment for ZenML pipeline subprocess")
                        try:
                            # Try to import ZenML and get the tracking URI
                            try:
                                print("[ZENML-SUBPROCESS-DEBUG] Attempting to import ZenML MLflow integration...")
                                from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
                                tracking_uri = get_tracking_uri()
                                if tracking_uri:
                                    zenml_env["MLFLOW_TRACKING_URI"] = tracking_uri
                                    print(f"✅ Setting MLFLOW_TRACKING_URI={tracking_uri} for subprocess")
                                    print(f"[ZENML-SUBPROCESS-DEBUG] URI type: {type(tracking_uri)}, length: {len(str(tracking_uri))}")
                                else:
                                    print("❌ Failed to get tracking URI from ZenML (returned None/empty)")
                                    print("[ZENML-SUBPROCESS-DEBUG] get_tracking_uri returned empty value")
                                    # Log error to pending errors
                                    try:
                                        log_pending_error("ZenML get_tracking_uri returned empty tracking URI")
                                    except Exception as log_err:
                                        print(f"[ZENML-SUBPROCESS-DEBUG] Failed to log error: {str(log_err)}")
                            except ImportError as import_err:
                                # Check if MLflow tracking URI is set in environment
                                print(f"[ZENML-SUBPROCESS-DEBUG] Import error: {str(import_err)}")
                                print(f"[ZENML-SUBPROCESS-DEBUG] Checking for environment variable fallback")
                                if "MLFLOW_TRACKING_URI" in os.environ:
                                    print(f"✅ Using existing MLFLOW_TRACKING_URI: {os.environ['MLFLOW_TRACKING_URI']}")
                                    # Ensure it's copied to the subprocess environment
                                    zenml_env["MLFLOW_TRACKING_URI"] = os.environ["MLFLOW_TRACKING_URI"]
                                    print(f"[ZENML-SUBPROCESS-DEBUG] Copied from parent environment to subprocess")
                                else:
                                    # Set default tracking URI as fallback
                                    print("[ZENML-SUBPROCESS-DEBUG] No environment variable, creating default URI")
                                    try:
                                        # Create mlruns directory if it doesn't exist
                                        mlruns_dir = os.path.join(os.getcwd(), '.zen', '.mlruns')
                                        os.makedirs(mlruns_dir, exist_ok=True)
                                        default_uri = f"file://{mlruns_dir}"
                                        zenml_env["MLFLOW_TRACKING_URI"] = default_uri
                                        print(f"⚠️ Setting default MLFLOW_TRACKING_URI={default_uri}")
                                        print(f"[ZENML-SUBPROCESS-DEBUG] Created directory: {mlruns_dir}")
                                        # Set in parent environment too
                                        os.environ["MLFLOW_TRACKING_URI"] = default_uri
                                        print(f"[ZENML-SUBPROCESS-DEBUG] Also set in parent environment")
                                    except Exception as dir_err:
                                        print(f"❌ Error creating default mlruns directory: {str(dir_err)}")
                                        print(f"[ZENML-SUBPROCESS-DEBUG] Exception type: {type(dir_err).__name__}")
                                        try:
                                            log_pending_error(f"Failed to create mlruns directory: {str(dir_err)}")
                                        except Exception as log_err:
                                            print(f"[ZENML-SUBPROCESS-DEBUG] Failed to log error: {str(log_err)}")
                        except Exception as e:
                            print(f"❌ Error setting MLflow environment: {str(e)}")
                            print(f"[ZENML-SUBPROCESS-DEBUG] Exception type: {type(e).__name__}")
                            print(f"[ZENML-SUBPROCESS-DEBUG] Exception details: {str(e)}")
                            # Log error
                            try:
                                import traceback
                                error_details = traceback.format_exc()
                                log_pending_error(f"Error setting up MLflow for ZenML subprocess: {str(e)}\n{error_details}")
                            except Exception as log_err:
                                print(f"[ZENML-SUBPROCESS-DEBUG] Failed to log error: {str(log_err)}")
                            # Continue anyway - ZenML should use its own config
                            print("[ZENML-SUBPROCESS-DEBUG] Will continue without setting MLFLOW_TRACKING_URI")

                        # Print final environment for debugging
                        print("\n[ZENML-SUBPROCESS-DEBUG] === Environment Summary ===")
                        for key in ["MLFLOW_TRACKING_URI", "PYTHONPATH", "ZENML_CONFIG_PATH"]:
                            if key in zenml_env:
                                print(f"[ZENML-SUBPROCESS-DEBUG] {key} = {zenml_env[key]}")
                            else:
                                print(f"[ZENML-SUBPROCESS-DEBUG] {key} not set in environment")

                        # Set up command to run ZenML pipeline
                        zenml_cmd = [
                            "python",
                            "src/lora_training_pipeline/training/zenml_pipeline.py",
                        ]
                        
                        try:
                            # Check if we're using a recovered process or starting a new one
                            if 'active_training_process' in locals() and active_training_process is not None and active_training_pid is not None:
                                # Use the recovered process
                                proc = active_training_process
                                print(f"ℹ️ Using recovered training process (PID: {active_training_pid})")
                                print(f"ℹ️ This process was detected during session recovery")
                                pipeline_tracker.update(message=f"Monitoring recovered training pipeline with PID {active_training_pid}")
                                print("ℹ️ Training is already running in the background")
                            else:
                                # Start a new training process with detailed debugging
                                # Add PROCESS_NAME to the environment we created earlier
                                zenml_env["PROCESS_NAME"] = "ZenMLTrainingPipeline"

                                # Add enhanced debugging environment variables
                                zenml_env["PYTHONUNBUFFERED"] = "1"  # Ensure unbuffered output for better debugging
                                zenml_env["MLFLOW_DEBUG"] = "1"      # Enable MLflow debug logging if supported
                                zenml_env["ZENML_LOG_LEVEL"] = "DEBUG"  # Set ZenML logging to DEBUG level

                                # Print the command for debugging
                                cmd_str = " ".join(zenml_cmd)
                                print(f"\n[SUBPROCESS-DEBUG] Executing command: {cmd_str}")

                                # Print important environment variables for debugging
                                print(f"[SUBPROCESS-DEBUG] Environment variables for subprocess:")
                                for key in ["MLFLOW_TRACKING_URI", "PYTHONPATH", "ZENML_CONFIG_PATH", "PROCESS_NAME"]:
                                    if key in zenml_env:
                                        print(f"[SUBPROCESS-DEBUG]   {key} = {zenml_env[key]}")
                                    else:
                                        print(f"[SUBPROCESS-DEBUG]   {key} not set in environment")

                                # Create a log file to capture subprocess output
                                log_dir = Path("./subprocess_logs")
                                log_dir.mkdir(exist_ok=True)
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                log_path = log_dir / f"zenml_{timestamp}.log"

                                try:
                                    # Start the process with our enhanced environment and log file
                                    print(f"[SUBPROCESS-DEBUG] Redirecting output to {log_path}")
                                    with open(log_path, "w") as log_file:
                                        # Write header to log file
                                        log_file.write(f"=== ZenML Pipeline Subprocess Log ===\n")
                                        log_file.write(f"Command: {cmd_str}\n")
                                        log_file.write(f"Started: {timestamp}\n")
                                        log_file.write(f"Working Directory: {os.getcwd()}\n")
                                        log_file.write(f"Environment:\n")
                                        for key in ["MLFLOW_TRACKING_URI", "PYTHONPATH", "ZENML_CONFIG_PATH"]:
                                            if key in zenml_env:
                                                log_file.write(f"  {key} = {zenml_env[key]}\n")
                                        log_file.write("="*50 + "\n\n")
                                        log_file.flush()

                                        # Start the process, redirecting output to log file
                                        proc = subprocess.Popen(
                                            zenml_cmd,
                                            env=zenml_env,
                                            stdout=log_file,
                                            stderr=log_file,
                                            text=True
                                        )
                                except Exception as proc_err:
                                    # Handle process start errors
                                    error_msg = f"Failed to start ZenML subprocess: {str(proc_err)}"
                                    print(f"❌ {error_msg}")
                                    print(f"[SUBPROCESS-DEBUG] Exception type: {type(proc_err).__name__}")
                                    print(f"[SUBPROCESS-DEBUG] Exception details: {str(proc_err)}")
                                    # Log error to pending_errors
                                    try:
                                        log_pending_error(error_msg)
                                    except Exception as log_err:
                                        print(f"[SUBPROCESS-DEBUG] Failed to log error: {str(log_err)}")
                                    # Raise the error to ensure the caller knows something went wrong
                                    raise RuntimeError(error_msg) from proc_err

                                print(f"✅ ZenML Pipeline started with PID {proc.pid} (logs: {log_path})")
                                pipeline_tracker.update(message=f"Training pipeline launched with PID {proc.pid}")
                                print(f"ℹ️ New ZenML Pipeline started with PID {proc.pid}")
                                print("ℹ️ Training is running in the background")
                            
                            # Monitor the process without blocking
                            print("ℹ️ Monitor will continue checking for new data")
                            
                            # Schedule function to check when training completes
                            def check_training_completion():
                                nonlocal cycle_info
                                proc_pid = proc.pid if hasattr(proc, 'pid') else active_training_pid
                                
                                # Check if process is still running
                                is_running = False
                                try:
                                    # First try proc.poll() for new processes
                                    if hasattr(proc, 'poll'):
                                        is_running = proc.poll() is None
                                    else:
                                        # For recovered processes, use psutil
                                        try:
                                            import psutil
                                            is_running = psutil.pid_exists(proc_pid)
                                        except (ImportError, Exception) as e:
                                            print(f"⚠️ Error checking process status: {e}")
                                            # Assume it's still running if we can't check
                                            is_running = True
                                except Exception as e:
                                    print(f"⚠️ Error in completion check: {e}")
                                    is_running = False
                                
                                if is_running:
                                    # Process still running, check again in 30 seconds (chunked for responsiveness)
                                    print("[DEBUG] Training process still running, waiting 30 seconds (chunked)")
                                    for i in range(30):
                                        if process_signal():
                                            print("[DEBUG] Signal processed during training wait, exiting")
                                            return
                                        time.sleep(1)
                                    check_training_completion()
                                else:
                                    # Process completed
                                    success = False
                                    
                                    # Check if the process completed successfully
                                    try:
                                        # For new processes, we can check the return code
                                        if hasattr(proc, 'returncode'):
                                            success = proc.returncode == 0
                                        else:
                                            # For recovered processes, we need to infer from training cycle info
                                            current_info = {}
                                            try:
                                                if TRAINING_CYCLE_FILE.exists():
                                                    with open(TRAINING_CYCLE_FILE, 'r') as f:
                                                        current_info = json.load(f)
                                                
                                                if current_info.get('cycle_count', 0) > cycle_info.get('cycle_count', 0):
                                                    success = True
                                            except Exception as e:
                                                print(f"⚠️ Error checking training cycle completion: {e}")
                                    except Exception as e:
                                        print(f"⚠️ Error determining process success: {e}")
                                    
                                    if success:
                                        print("\n" + "="*80)
                                        print("✅ TRAINING CYCLE COMPLETED SUCCESSFULLY")
                                        
                                        # Update cycle count and total data points
                                        new_cycle_info = load_training_cycle_info()  # Reload in case file was updated
                                        new_cycle_info["cycle_count"] += 1
                                        new_cycle_info["total_valid_data_points"] = total_valid
                                        save_training_cycle_info(new_cycle_info)
                                        
                                        # Print updated cycle info
                                        print(f"ℹ️ Training cycle incremented to: {new_cycle_info['cycle_count']}")
                                        print(f"ℹ️ Next threshold will be: {calculate_threshold(new_cycle_info['cycle_count'], base_threshold)}")
                                        print("="*80 + "\n")
                                        
                                        # Update global cycle_info to match
                                        cycle_info = new_cycle_info
                                    else:
                                        print("\n" + "="*80)
                                        print(f"❌ TRAINING CYCLE FAILED: Return code {proc.returncode}")
                                        print("ℹ️ Training cycle count not incremented")
                                        print("ℹ️ Check logs for errors")
                                        print("="*80 + "\n")
                            
                            # Start checking for completion in a separate thread
                            import threading
                            completion_thread = threading.Thread(target=check_training_completion)
                            completion_thread.daemon = True
                            completion_thread.start()
                            
                        except Exception as e:
                            pipeline_tracker.update(message=f"Error launching pipeline: {e}")
                            print(f"❌ ERROR: Failed to start ZenML pipeline: {e}")
                            log_pending_error(f"Failed to start ZenML pipeline: {e}")
                        
                        pipeline_tracker.complete("Automatic pipeline trigger complete")
                        
                    elif total_valid >= current_threshold and time_since_last_training < MIN_TRAINING_INTERVAL:
                        print(f"ℹ️ Threshold met but training cooldown active ({time_since_last_training:.0f}/{MIN_TRAINING_INTERVAL}s)")
                    elif total_valid >= current_threshold and total_valid <= cycle_info["total_valid_data_points"]:
                        print(f"ℹ️ Threshold met but no new valid data since last training cycle")
                    else:
                        remaining = current_threshold - total_valid
                        print(f"ℹ️ Waiting for more data: need {remaining} more valid data points to reach threshold")
                    
                    # Update the last valid count and total files count in the cycle info
                    print(f"[DEBUG] Updating cycle info with current data counts")
                    cycle_info["last_valid_count"] = total_valid
                    cycle_info["total_original_files"] = len(original_files)
                    cycle_info["total_valid_files"] = len(valid_files)
                    cycle_info["total_invalid_files"] = len(invalid_files)
                    cycle_info["last_scan_time"] = int(time.time())
                    
                    print(f"[DEBUG] Saving training cycle info: {cycle_info}")
                    save_training_cycle_info(cycle_info)
                    
                except Exception as e:
                    print(f"⚠️ Error in data monitoring loop: {e}")
                    print(f"[DEBUG] Exception type: {type(e).__name__}")
                    import traceback
                    print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                    log_pending_error(f"Error in data monitoring loop: {e}")
                    logger.error(f"Data monitoring loop error: {e}", exc_info=True)
                    # Continue monitoring despite errors
                
                # Wait before checking again, but in smaller chunks to be responsive to interrupts
                print(f"[DEBUG] Waiting {check_interval} seconds before next data check (chunked for responsiveness)")
                remaining_time = check_interval
                chunks_slept = 0
                while remaining_time > 0:
                    # Check for signals before sleeping - CRITICAL for CTRL-C responsiveness
                    if process_signal():
                        print("[DEBUG] Signal processed during sleep, breaking immediately")
                        break
                    
                    # Also check global signal flag for immediate response
                    if signal_received:
                        print("[DEBUG] Global signal flag detected during sleep, processing immediately")
                        if process_signal():
                            break
                    
                    # Check shutdown flag for immediate exit
                    if shutdown_requested:
                        print("[DEBUG] Shutdown requested during sleep, exiting immediately")
                        sys.exit(0)
                    
                    sleep_time = min(0.5, remaining_time)  # Sleep max 0.5 second at a time for better responsiveness
                    time.sleep(sleep_time)
                    remaining_time -= sleep_time
                    chunks_slept += 1
                    if chunks_slept % 20 == 0:  # Log every 10 seconds (20 * 0.5s)
                        print(f"[DEBUG] Sleep progress: {chunks_slept * 0.5:.1f} seconds elapsed, {remaining_time:.1f} seconds remaining")
        
        except KeyboardInterrupt:
            monitor_tracker.complete("Data monitoring stopped by user")
            raise  # Re-raise to be caught by the outer try/except
    
    # Start the monitoring in the main loop
    try:
        print("Pipeline running. Press Ctrl+C to exit...")
        print("\n" + "="*80)
        print("IMPORTANT: The inference server will continue running in the background")
        print("If you need to stop it manually, use:")
        print("  - Press Ctrl+C to stop all components")
        print("  - Or manually find and terminate the FastAPI process (PID in inference_process.pid)")
        print("="*80 + "\n")
        
        # Start data monitoring with base threshold of 10 and check every 60 seconds
        check_data_threshold(base_threshold=10, check_interval=60)
        
        # If we got here, it means the data monitoring loop ended normally
        # (which is unlikely since it's designed to run until interrupted)
        print("\n" + "="*80)
        print("PIPELINE COMPLETED NORMALLY")
        print("="*80 + "\n")
        complete_session()
        
    except KeyboardInterrupt:
        print("Exiting...")
        shutdown_progress = ProgressTracker("Pipeline Shutdown")
        shutdown_progress.start("Stopping all pipeline processes")
        stop_processes()
        
        # Update session info for keyboard interrupt
        try:
            if ACTIVE_SESSION_FILE.exists():
                session_data = read_session_info(ACTIVE_SESSION_FILE)
                if session_data:
                    session_data["end_time"] = datetime.datetime.now().isoformat()
                    session_data["status"] = "user_interrupted"
                    
                    # Write updated data to the session file
                    with open(ACTIVE_SESSION_FILE, 'w') as f:
                        json.dump(session_data, f, indent=2)
                    
                    # Copy to last session file as well
                    with open(LAST_SESSION_FILE, 'w') as f:
                        json.dump(session_data, f, indent=2)
        except Exception:
            pass
        
        # Notify user about the FastAPI server status
        fastapi_pids = get_fastapi_pids()
        if fastapi_pids:
            pid_str = ", ".join(str(pid) for pid in fastapi_pids)
            shutdown_progress.update(message="Checking inference server status")
            print("\n" + "="*80)
            print(f"⚠️ FastAPI inference server (PIDs: {pid_str}) is still running")
            print(f"ℹ️ This allows model inference to continue working even after the pipeline script ends")
            print(f"ℹ️ To manually terminate the server, use your system's process manager")
            print(f"ℹ️ On Windows: taskkill /F /PID {pid_str}")
            print(f"ℹ️ On Linux/Mac: kill {pid_str}")
            print("="*80 + "\n")
        else:
            print("No FastAPI inference server detected")
            
        shutdown_progress.complete("Pipeline shutdown complete")
        
        # Mark the session as completed successfully
        complete_session()
        
    except Exception as e:
        # Handle unexpected errors
        print(f"\n{'='*80}")
        print(f"UNHANDLED ERROR IN PIPELINE: {e}")
        print(f"{'='*80}\n")
        
        # Try to record the error in the session info
        try:
            if ACTIVE_SESSION_FILE.exists():
                session_data = read_session_info(ACTIVE_SESSION_FILE)
                if session_data:
                    session_data["end_time"] = datetime.datetime.now().isoformat()
                    session_data["status"] = "crashed"
                    session_data["error"] = str(e)
                    
                    # Write updated data to the session file
                    with open(ACTIVE_SESSION_FILE, 'w') as f:
                        json.dump(session_data, f, indent=2)
                    
                    # Copy to last session file as well
                    with open(LAST_SESSION_FILE, 'w') as f:
                        json.dump(session_data, f, indent=2)
        except Exception:
            pass
            
        # Create a progress tracker for the error shutdown
        shutdown_progress = ProgressTracker("Pipeline Shutdown")
        shutdown_progress.start("Stopping all pipeline processes after error")
        
        # Stop processes even on unexpected errors
        stop_processes()
        
        # Notify user about the FastAPI server status
        fastapi_pids = get_fastapi_pids()
        if fastapi_pids:
            pid_str = ", ".join(str(pid) for pid in fastapi_pids)
            shutdown_progress.update(message="Checking inference server status")
            print("\n" + "="*80)
            print(f"⚠️ FastAPI inference server (PIDs: {pid_str}) is still running")
            print(f"ℹ️ This allows model inference to continue working even after the pipeline script ends")
            print(f"ℹ️ To manually terminate the server, use your system's process manager")
            print(f"ℹ️ On Windows: taskkill /F /PID {pid_str}")
            print(f"ℹ️ On Linux/Mac: kill {pid_str}")
            print("="*80 + "\n")
        else:
            print("No FastAPI inference server detected")
            
        # Complete the progress tracker
        shutdown_progress.complete("Pipeline shutdown after error")

def delete_all_datasets():
    """
    Deletes all dataset files and resets all counters, but doesn't stop any running training.
    Will not run if cleaning is in progress to avoid interrupting the cleaning stage.
    Training files in the training directory are preserved for active training.
    
    Returns:
        dict: A status report with counts of deleted files or error message
    """
    from pathlib import Path
    import shutil
    import json
    import os
    from src.lora_training_pipeline.config import DATASET_NAME, DATA_VERSION
    
    DATA_DIR = Path("./data")
    VALID_DIR = DATA_DIR / "valid"
    REJECTED_DIR = DATA_DIR / "rejected"
    TRAINING_DIR = DATA_DIR / "training"
    CHECKPOINT_FILE = DATA_DIR / "validation_checkpoint.json"
    TRAINING_CYCLE_FILE = Path("./training_cycle_info.json")
    
    # Create training directory if it doesn't exist
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize counts for reporting
    delete_counts = {
        "original_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "cleaned_files": 0,
        "consolidated_files": 0,
        "other_files": 0,
        "copied_to_training": 0
    }
    
    print("\n" + "="*80)
    print("STARTING DATASET CLEANUP")
    print("="*80)
    
    # Check if cleaning is in progress with enhanced detection of stale markers
    CLEANING_IN_PROGRESS_FILE = Path("./.cleaning_in_progress")
    if CLEANING_IN_PROGRESS_FILE.exists():
        try:
            import time
            import json
            import os
            
            # Check if the marker is stale (older than 1 hour)
            marker_creation_time = CLEANING_IN_PROGRESS_FILE.stat().st_mtime
            if time.time() - marker_creation_time > 3600:  # 1 hour timeout
                print(f"⚠️ Found stale cleaning marker (older than 1 hour), removing it")
                CLEANING_IN_PROGRESS_FILE.unlink()
                # Continue with deletion since marker was stale
            else:
                # Try to read the marker file to check if the process is still running
                try:
                    with open(CLEANING_IN_PROGRESS_FILE, 'r') as f:
                        marker_data = json.load(f)
                        pid = marker_data.get("pid")
                        
                        if pid:
                            # Check if the process is still running
                            # First try using psutil if available
                            try:
                                # Note: you need to install psutil with: uv pip install psutil
                                # for enhanced process monitoring
                                import psutil
                                if psutil.pid_exists(pid):
                                    # Process is still running, can't delete
                                    error_message = f"Cleaning is currently in progress (PID: {pid}). Cannot delete datasets during cleaning stage."
                                    print(f"\n❌ ERROR: {error_message}")
                                    return {"error": error_message, "status": "cleaning_in_progress"}
                                else:
                                    # Process is no longer running, marker is orphaned
                                    print(f"⚠️ Found orphaned cleaning marker (PID {pid} not running), removing it")
                                    CLEANING_IN_PROGRESS_FILE.unlink()
                                    # Continue with deletion since process is not running
                            except ImportError:
                                # No psutil available, use a basic check with os (less reliable)
                                print(f"ℹ️ Note: For enhanced orphaned process detection, install psutil: uv pip install psutil")
                                
                                # Check if the marker was created recently (last 10 minutes)
                                if time.time() - marker_creation_time < 600:  # 10 minutes
                                    # Recent marker, assume process is still running
                                    error_message = f"Cleaning may be in progress (PID: {pid}, marker created {int(time.time() - marker_creation_time)}s ago). Cannot delete datasets during cleaning stage."
                                    print(f"\n❌ ERROR: {error_message}")
                                    print(f"ℹ️ Note: Install psutil for better orphaned process detection: uv pip install psutil")
                                    return {"error": error_message, "status": "cleaning_in_progress"}
                                else:
                                    # Older marker, probably orphaned
                                    print(f"⚠️ Found likely orphaned cleaning marker (PID {pid}, marker {int(time.time() - marker_creation_time)}s old), removing it")
                                    CLEANING_IN_PROGRESS_FILE.unlink()
                                    # Continue with deletion
                        else:
                            # No PID in marker file, assume cleaning is in progress
                            error_message = "Cleaning appears to be in progress (PID unknown). Cannot delete datasets during cleaning stage."
                            print(f"\n❌ ERROR: {error_message}")
                            return {"error": error_message, "status": "cleaning_in_progress"}
                except Exception as e:
                    # Couldn't read the marker file, check if it's recent
                    if time.time() - marker_creation_time < 600:  # 10 minutes
                        # Recent marker, assume cleaning is in progress
                        error_message = f"Cleaning appears to be in progress (marker created {int(time.time() - marker_creation_time)}s ago). Cannot delete datasets during cleaning stage."
                        print(f"\n❌ ERROR: {error_message}")
                        return {"error": error_message, "status": "cleaning_in_progress"}
                    else:
                        # Old marker but not over 1 hour, might be stale
                        print(f"⚠️ Found potentially stale cleaning marker ({int(time.time() - marker_creation_time)}s old), removing it")
                        CLEANING_IN_PROGRESS_FILE.unlink()
                        # Continue with deletion since marker might be stale
        except ImportError as e:
            # If psutil is not available, use a simpler check
            print(f"⚠️ Warning: Enhanced process checking unavailable ({e}), using simple marker check")
            # Fall back to simple timeout-based check
            if time.time() - CLEANING_IN_PROGRESS_FILE.stat().st_mtime > 1800:  # 30 minutes
                print(f"⚠️ Found stale cleaning marker (older than 30 minutes), removing it")
                CLEANING_IN_PROGRESS_FILE.unlink()
                # Continue with deletion since marker was stale
            else:
                # Recent marker, assume cleaning is in progress
                error_message = "Cleaning appears to be in progress. Cannot delete datasets during cleaning stage."
                print(f"\n❌ ERROR: {error_message}")
                return {"error": error_message, "status": "cleaning_in_progress"}
        except Exception as e:
            # If any other error occurs, be conservative
            error_message = f"Cleaning marker exists and couldn't verify status: {e}. Cannot delete datasets during potential cleaning."
            print(f"\n❌ ERROR: {error_message}")
            return {"error": error_message, "status": "unknown_cleaning_status"}
    
    # Check if training is running
    TRAINING_LOCK_FILE = Path("./.training_lock")
    training_running = TRAINING_LOCK_FILE.exists()
    
    if training_running:
        print("⚠️ Training is currently running - will preserve datasets for active training")
        # We no longer need to copy files since training now uses files from the training directory
        print("ℹ️ Training files already exist in training directory and will be preserved")
    
    # Delete files in main data directory
    if DATA_DIR.exists():
        # Delete original files
        original_files = list(DATA_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_original_*.parquet"))
        for file in original_files:
            try:
                file.unlink()
                delete_counts["original_files"] += 1
            except Exception as e:
                print(f"⚠️ Error deleting {file.name}: {e}")
        
        # Delete valid files in main directory (if any)
        valid_files_main = list(DATA_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_valid_*.parquet"))
        for file in valid_files_main:
            try:
                file.unlink()
                delete_counts["valid_files"] += 1
            except Exception as e:
                print(f"⚠️ Error deleting {file.name}: {e}")
        
        # Delete legacy cleaned files
        cleaned_files = list(DATA_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_cleaned_*.parquet"))
        for file in cleaned_files:
            try:
                file.unlink()
                delete_counts["cleaned_files"] += 1
            except Exception as e:
                print(f"⚠️ Error deleting {file.name}: {e}")
        
        # Delete consolidated files
        consolidated_files = list(DATA_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_consolidated_*.parquet"))
        for file in consolidated_files:
            try:
                file.unlink()
                delete_counts["consolidated_files"] += 1
            except Exception as e:
                print(f"⚠️ Error deleting {file.name}: {e}")
    
    # Delete files in valid directory
    if VALID_DIR.exists():
        valid_files = list(VALID_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_valid_*.parquet"))
        for file in valid_files:
            try:
                file.unlink()
                delete_counts["valid_files"] += 1
            except Exception as e:
                print(f"⚠️ Error deleting {file.name}: {e}")
    
    # Delete files in rejected directory
    if REJECTED_DIR.exists():
        invalid_files = list(REJECTED_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_invalid_*.parquet"))
        for file in invalid_files:
            try:
                file.unlink()
                delete_counts["invalid_files"] += 1
            except Exception as e:
                print(f"⚠️ Error deleting {file.name}: {e}")
    
    # Reset checkpoint file
    if CHECKPOINT_FILE.exists():
        try:
            # Initialize with empty lists
            checkpoint_data = {
                "processed_files": [],
                "last_processed_timestamp": "",
                "valid_files": [],
                "invalid_files": []
            }
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print("✅ Reset data cleaning checkpoint file")
        except Exception as e:
            print(f"⚠️ Error resetting checkpoint file: {e}")
    
    # Note: We DO NOT reset the training checkpoint
    # The training process should continue to manage its own files
    
    # Reset training cycle info
    if TRAINING_CYCLE_FILE.exists():
        try:
            # Don't reset cycle count - only reset data point counts
            with open(TRAINING_CYCLE_FILE, 'r') as f:
                cycle_info = json.load(f)
            
            # Keep cycle count but reset data points
            cycle_info["total_valid_data_points"] = 0
            cycle_info["last_valid_count"] = 0
            
            with open(TRAINING_CYCLE_FILE, 'w') as f:
                json.dump(cycle_info, f, indent=2)
            print("✅ Reset training cycle data counts")
        except Exception as e:
            print(f"⚠️ Error updating training cycle file: {e}")
    
    # Count preserved training files
    training_files_count = 0
    if TRAINING_DIR.exists():
        training_files = list(TRAINING_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_training_*.parquet"))
        training_files_count = len(training_files)
    
    # Calculate total files deleted
    total_deleted = sum(delete_counts.values())
    
    print("\n" + "="*80)
    print("DATASET CLEANUP SUMMARY")
    print("="*80)
    print(f"✅ Total files deleted: {total_deleted}")
    print(f"  - Original files: {delete_counts['original_files']}")
    print(f"  - Valid files: {delete_counts['valid_files']}")
    print(f"  - Invalid files: {delete_counts['invalid_files']}")
    print(f"  - Cleaned files (legacy): {delete_counts['cleaned_files']}")
    print(f"  - Consolidated files: {delete_counts['consolidated_files']}")
    if training_files_count > 0:
        print(f"✅ Training files preserved: {training_files_count}")
    print("✅ Reset all data counters")
    print("="*80)
    
    # Add training files count to the result
    delete_counts["training_files_preserved"] = training_files_count
    
    return delete_counts

if __name__ == "__main__":
    # Install emergency keyboard interrupt handler IMMEDIATELY
    def emergency_exit_handler(sig, frame):
        """Emergency exit handler that responds immediately to CTRL-C"""
        print("\n🚨 EMERGENCY EXIT: Keyboard interrupt detected!")
        print("Forcing immediate shutdown...")
        import os
        os._exit(1)
    
    # Install the emergency handler before any other setup
    import signal
    signal.signal(signal.SIGINT, emergency_exit_handler)
    print("[DEBUG] Emergency keyboard interrupt handler installed")
    
    try:
        print("\n" + "="*80)
        print("PIPELINE STARTUP DEBUG INFO")
        print("="*80)
        print(f"[DEBUG] Script execution started at: {datetime.datetime.now().isoformat()}")
        print(f"[DEBUG] Python version: {sys.version}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] Script path: {__file__}")
        print(f"[DEBUG] Command line arguments: {sys.argv}")
        print(f"[DEBUG] Process PID: {os.getpid()}")
        print(f"[DEBUG] User: {os.environ.get('USER', os.environ.get('USERNAME', 'unknown'))}")
        print(f"[DEBUG] PSUTIL_AVAILABLE: {PSUTIL_AVAILABLE}")
        print(f"[DEBUG] ENHANCED_PROCESS_MANAGEMENT: {ENHANCED_PROCESS_MANAGEMENT}")
        
        # Check environment
        important_env_vars = ['PYTHONPATH', 'PATH', 'VIRTUAL_ENV', 'CONDA_DEFAULT_ENV']
        for var in important_env_vars:
            value = os.environ.get(var, 'NOT_SET')
            if len(value) > 100:
                value = value[:100] + "..."
            print(f"[DEBUG] {var}: {value}")
        
        print("="*80 + "\n")
        
        # Process any queued signals before starting
        if not signal_queue.empty():
            print("[DEBUG] Processing queued signals before startup")
            process_signal()

        # Replace emergency handler with sophisticated signal handling system
        print("[DEBUG] Installing sophisticated signal handling system...")
        signal.signal(signal.SIGINT, non_blocking_signal_handler)  # Handle Ctrl+C
        signal.signal(signal.SIGTERM, non_blocking_signal_handler)  # Handle termination signals
        print("[DEBUG] Sophisticated signal handlers installed")
        
        if "--delete-datasets" in sys.argv: # Special flag for dataset deletion
            print("[DEBUG] Dataset deletion mode requested")
            delete_all_datasets()
        else:
            print("[DEBUG] Normal pipeline execution mode")
            # Normal execution - use enhanced process management if available
            if ENHANCED_PROCESS_MANAGEMENT:
                print("\n" + "="*80)
                print("STARTING PIPELINE WITH ENHANCED PROCESS MANAGEMENT")
                print("="*80)
                print("[DEBUG] Calling run_pipeline_with_enhanced_process_management()...")
                try:
                    # Run the enhanced version that prevents duplicate processes
                    data_collection_process, inference_ui_process, fastapi_process = run_pipeline_with_enhanced_process_management()
                    print(f"[DEBUG] Enhanced process management returned:")
                    print(f"  - data_collection_process: {data_collection_process}")
                    print(f"  - inference_ui_process: {inference_ui_process}")
                    print(f"  - fastapi_process: {fastapi_process}")
                    
                    # Add the processes to the global processes list for cleanup on exit
                    initial_process_count = len(processes)
                    if data_collection_process:
                        processes.append(data_collection_process)
                        print(f"[DEBUG] Added data collection process (PID: {data_collection_process.pid})")
                    if inference_ui_process:
                        processes.append(inference_ui_process)
                        print(f"[DEBUG] Added inference UI process (PID: {inference_ui_process.pid})")
                    if fastapi_process:
                        processes.append(fastapi_process)
                        print(f"[DEBUG] Added FastAPI process (PID: {fastapi_process.pid})")
                    
                    final_process_count = len(processes)
                    print(f"[DEBUG] Process list grew from {initial_process_count} to {final_process_count} processes")
                    
                except Exception as enhanced_error:
                    print(f"[DEBUG] ERROR in enhanced process management: {enhanced_error}")
                    print(f"[DEBUG] Exception type: {type(enhanced_error).__name__}")
                    import traceback
                    print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                    raise

                # Continue with the rest of the script, which includes monitoring and other processes
                print("[DEBUG] Calling main() function...")
                try:
                    main()
                    print("[DEBUG] main() function completed, entering signal processing loop")
                except KeyboardInterrupt:
                    print("\n[DEBUG] KeyboardInterrupt caught during main() execution")
                    print("Keyboard interrupt received during startup. Shutting down...")
                    stop_processes()
                    sys.exit(0)
                except Exception as main_error:
                    print(f"[DEBUG] ERROR in main() function: {main_error}")
                    print(f"[DEBUG] Exception type: {type(main_error).__name__}")
                    import traceback
                    print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                    raise

                # Main loop to process signals and keep the program running
                loop_iteration = 0
                while True:
                    try:
                        loop_iteration += 1
                        if loop_iteration % 600 == 0:  # Log every 60 seconds (600 * 0.1s)
                            print(f"[DEBUG] Signal processing loop iteration {loop_iteration}, processes: {len(processes)}")
                        
                        # Check shutdown flag for immediate exit
                        if shutdown_requested:
                            print("[DEBUG] Shutdown requested in main signal loop, exiting immediately")
                            sys.exit(0)
                        
                        # Process any queued signals
                        if process_signal():
                            print("[DEBUG] Signal processed, breaking from main loop")
                            break  # Exit if a signal was processed (it will call exit(0))
                        time.sleep(0.1)  # Shorter sleep for more responsive interrupts
                    except KeyboardInterrupt:
                        print("\n[DEBUG] KeyboardInterrupt caught in main loop")
                        print("Keyboard interrupt received. Shutting down...")
                        stop_processes()
                        sys.exit(0)
            else:
                # Run the standard version if enhanced process management is not available
                print("\n" + "="*80)
                print("STARTING PIPELINE WITH STANDARD PROCESS MANAGEMENT")
                print("="*80)
                try:
                    main()
                except KeyboardInterrupt:
                    print("\n[DEBUG] KeyboardInterrupt caught during main() execution (standard mode)")
                    print("Keyboard interrupt received during startup. Shutting down...")
                    stop_processes()
                    sys.exit(0)
                except Exception as main_error:
                    print(f"[DEBUG] ERROR in main() function (standard mode): {main_error}")
                    print(f"[DEBUG] Exception type: {type(main_error).__name__}")
                    import traceback
                    print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                    raise

                # Main loop to process signals and keep the program running
                while True:
                    try:
                        # Check shutdown flag for immediate exit
                        if shutdown_requested:
                            print("[DEBUG] Shutdown requested in standard signal loop, exiting immediately")
                            sys.exit(0)
                        
                        # Process any queued signals
                        if process_signal():
                            break  # Exit if a signal was processed (it will call exit(0))
                        time.sleep(0.1)  # Shorter sleep for more responsive interrupts
                    except KeyboardInterrupt:
                        print("\nKeyboard interrupt received. Shutting down...")
                        stop_processes()
                        sys.exit(0)
    except Exception as e:
        # Use thread-safe error reporting
        print(f"[DEBUG] Critical error caught at main level: {type(e).__name__}: {e}")
        logger.error(f"Critical error in main program: {e}")
        sys.stderr.write(f"\n[CRITICAL ERROR] {type(e).__name__}: {e}\n")
        
        import traceback
        tb = traceback.format_exc()
        print(f"[DEBUG] Full traceback:\n{tb}")
        logger.error(f"Traceback: {tb}")
        sys.stderr.write(f"\n{tb}\n")
        sys.stderr.flush()

        # Try to process any pending signals before exiting
        try:
            print("[DEBUG] Attempting to process pending signals before exit")
            process_signal()
        except Exception as signal_e:
            print(f"[DEBUG] Error processing signals during shutdown: {signal_e}")

        # Try to clean up processes
        try:
            print("[DEBUG] Attempting to clean up processes before exit")
            stop_processes()
        except Exception as cleanup_e:
            print(f"[DEBUG] Error during cleanup: {cleanup_e}")

        print("[DEBUG] Exiting with error code 1")
        # Exit with error code
        sys.exit(1)