# LoRA_Training_Pipeline/src/inference/fastapi_inference.py
# ----------------------------------------
# FastAPI application for model inference.
# ----------------------------------------

# Check required dependencies first
try:
    import sys
    from pathlib import Path
    import os
    import time
    import traceback
    import datetime
    
    # Install global exception handler for FastAPI inference
    def fastapi_exception_handler(exc_type, exc_value, exc_traceback):
        """Handle any unhandled exceptions in FastAPI inference with comprehensive logging."""
        error_msg = f"""
CRITICAL UNHANDLED EXCEPTION IN FASTAPI INFERENCE
===============================================
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
            error_file = Path("fastapi_critical_errors.log")
            with open(error_file, "a", encoding="utf-8") as f:
                f.write(error_msg + "\n" + "="*80 + "\n")
            print(f"[CRITICAL] FastAPI error logged to {error_file}", file=sys.stderr)
        except Exception as log_err:
            print(f"[CRITICAL] Failed to write FastAPI error log: {log_err}", file=sys.stderr)
        
        # Try to clean up singleton lock before exiting
        try:
            release_singleton_lock()
            print("[CLEANUP] Singleton lock released due to critical error", file=sys.stderr)
        except Exception as cleanup_err:
            print(f"[CLEANUP] Failed to release singleton lock: {cleanup_err}", file=sys.stderr)
        
        # Call the original exception handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    # Install the exception handler
    sys.excepthook = fastapi_exception_handler
    print("[DEBUG] FastAPI global exception handler installed")
    
    # Add the project root to PATH so we can import our module
    root_dir = Path(__file__).resolve().parents[3]  # Go up 3 levels from current file
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.lora_training_pipeline.utils.helpers import check_dependencies
except ImportError as e:
    print(f"[DEBUG] Dependency import error type: {type(e).__name__}")
    print(f"[DEBUG] Dependency import error details: {e}")
    print(f"ERROR: Missing critical dependency: {e}")
    print("Please make sure all dependencies are installed with: uv pip install -e .")
    sys.exit(1)
except Exception as critical_err:
    print(f"[DEBUG] Critical initialization error type: {type(critical_err).__name__}")
    print(f"[DEBUG] Critical initialization error details: {critical_err}")
    print(f"CRITICAL ERROR during FastAPI initialization: {critical_err}")
    sys.exit(1)

# Check specific dependencies for this script
check_dependencies(['fastapi', 'transformers', 'peft', 'torch'])

# Also try to import ollama for fallback model
print("[DEBUG] Attempting to import Ollama library for fallback model...")
try:
    import ollama
    OLLAMA_AVAILABLE = True
    print("[DEBUG] ✅ Ollama library imported successfully for fallback model")
    print(f"[DEBUG] Ollama module path: {getattr(ollama, '__file__', 'Unknown')}")
    print(f"[DEBUG] Ollama version: {getattr(ollama, '__version__', 'Unknown')}")
    
    # Test basic Ollama functions availability
    available_functions = []
    for func_name in ['list', 'generate', 'pull', 'show']:
        if hasattr(ollama, func_name):
            available_functions.append(func_name)
    print(f"[DEBUG] Available Ollama functions: {available_functions}")
    
except ImportError as import_err:
    OLLAMA_AVAILABLE = False
    print(f"[DEBUG] ❌ Ollama library import failed: {type(import_err).__name__}: {import_err}")
    print("[DEBUG] Ollama fallback will not be available")
    print("[DEBUG] To install Ollama: pip install ollama")
except Exception as unexpected_err:
    OLLAMA_AVAILABLE = False
    print(f"[DEBUG] ❌ Unexpected error during Ollama import: {type(unexpected_err).__name__}: {unexpected_err}")
    print("[DEBUG] Ollama fallback will not be available")
    import traceback
    print(f"[DEBUG] Ollama import traceback: {traceback.format_exc()}")

from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from typing import Dict
from src.lora_training_pipeline.config import BASE_MODEL_NAME

# --- Configuration ---
# Get paths from env vars with fallbacks
LORA_MODEL_PATH = os.getenv("LORA_MODEL_PATH", "./output/best_model")
MODEL_UPDATE_SIGNAL_FILE = Path(os.getenv("MODEL_UPDATE_SIGNAL_FILE", "./.model_update"))

# Ollama fallback configuration
DEFAULT_OLLAMA_MODEL = "gemma3:1b"  # Default Ollama model to use as fallback
OLLAMA_SERVER_URL = "http://localhost:11434"  # Default Ollama server URL

# Print configuration info at startup
print("\n" + "="*80)
print("FASTAPI INFERENCE SERVER CONFIGURATION")
print("="*80)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print(f"LORA_MODEL_PATH: {LORA_MODEL_PATH}")
print(f"MODEL_UPDATE_SIGNAL_FILE: {MODEL_UPDATE_SIGNAL_FILE}")
print(f"DEFAULT_OLLAMA_MODEL: {DEFAULT_OLLAMA_MODEL}")
print(f"OLLAMA_SERVER_URL: {OLLAMA_SERVER_URL}")
print(f"OLLAMA_AVAILABLE: {OLLAMA_AVAILABLE}")

# Print ALL environment variables for debugging
print("\nALL ENVIRONMENT VARIABLES:")
for key, value in sorted(os.environ.items()):
    print(f"- {key}: {value}")

# --- SINGLETON PROTECTION ---
# Implement strict singleton pattern to prevent multiple FastAPI servers on same port
import atexit
import signal
import json
import datetime

# Singleton lock file path
FASTAPI_SINGLETON_LOCK = Path("./fastapi_inference_singleton.lock")
FASTAPI_PID_FILE = Path("./inference_process.pid")

def create_singleton_lock():
    """Create exclusive lock to ensure only one FastAPI server runs on port 8001"""
    print("\n" + "="*80)
    print("FASTAPI SINGLETON PROTECTION - ACQUIRING EXCLUSIVE LOCK")
    print("="*80)
    
    current_pid = os.getpid()
    current_time = datetime.datetime.now().isoformat()
    
    # Check if lock file exists
    if FASTAPI_SINGLETON_LOCK.exists():
        print(f"Found existing lock file: {FASTAPI_SINGLETON_LOCK}")
        try:
            with open(FASTAPI_SINGLETON_LOCK, 'r') as f:
                lock_data = json.load(f)
            
            existing_pid = lock_data.get('pid')
            lock_time = lock_data.get('timestamp', 'unknown')
            
            print(f"Existing lock PID: {existing_pid}, created at: {lock_time}")
            
            # Check if the process is still running
            if existing_pid:
                try:
                    import psutil
                    if psutil.pid_exists(existing_pid):
                        # Process exists, check if it's actually our FastAPI server
                        try:
                            proc = psutil.Process(existing_pid)
                            cmdline = ' '.join(proc.cmdline())
                            if 'fastapi_inference' in cmdline or 'uvicorn' in cmdline:
                                print(f"❌ CRITICAL: Another FastAPI inference server is already running!")
                                print(f"❌ PID: {existing_pid}")
                                print(f"❌ Command: {cmdline}")
                                print(f"❌ Cannot start duplicate server on port 8001")
                                print("To force start (NOT RECOMMENDED), set FASTAPI_FORCE_START=true")
                                
                                if os.environ.get("FASTAPI_FORCE_START", "").lower() != "true":
                                    sys.exit(1)
                                else:
                                    print("⚠️ FORCE START enabled - killing existing server")
                                    proc.terminate()
                                    time.sleep(2)
                                    if proc.is_running():
                                        proc.kill()
                            else:
                                print(f"Existing PID {existing_pid} is not a FastAPI server, removing stale lock")
                                FASTAPI_SINGLETON_LOCK.unlink()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            print(f"PID {existing_pid} not accessible, removing stale lock")
                            FASTAPI_SINGLETON_LOCK.unlink()
                    else:
                        print(f"PID {existing_pid} no longer exists, removing stale lock")
                        FASTAPI_SINGLETON_LOCK.unlink()
                except ImportError:
                    print("⚠️ psutil not available, cannot verify existing process")
                    # Check lock age as fallback
                    lock_age = time.time() - FASTAPI_SINGLETON_LOCK.stat().st_mtime
                    if lock_age > 3600:  # 1 hour
                        print(f"Lock file is old ({lock_age:.1f}s), removing stale lock")
                        FASTAPI_SINGLETON_LOCK.unlink()
                    else:
                        print("❌ Cannot verify if server is running, exiting for safety")
                        sys.exit(1)
        except Exception as e:
            print(f"[DEBUG] Error reading lock file: {type(e).__name__}: {e}")
            print("Removing corrupted lock file")
            FASTAPI_SINGLETON_LOCK.unlink()
    
    # Create new lock
    lock_data = {
        'pid': current_pid,
        'timestamp': current_time,
        'port': 8001,
        'process_type': 'fastapi_inference',
        'command': ' '.join(sys.argv),
        'python_executable': sys.executable
    }
    
    try:
        with open(FASTAPI_SINGLETON_LOCK, 'w') as f:
            json.dump(lock_data, f, indent=2)
        
        print(f"✅ Singleton lock acquired successfully")
        print(f"✅ Lock file: {FASTAPI_SINGLETON_LOCK}")
        print(f"✅ Server PID: {current_pid}")
        
        # Also create/update the PID file for compatibility
        with open(FASTAPI_PID_FILE, 'w') as f:
            json.dump(lock_data, f, indent=2)
        
        print(f"✅ PID file updated: {FASTAPI_PID_FILE}")
        
    except Exception as e:
        print(f"❌ CRITICAL: Failed to create singleton lock: {e}")
        sys.exit(1)

def release_singleton_lock():
    """Release the singleton lock when server shuts down"""
    print("\n" + "="*80)
    print("FASTAPI SINGLETON PROTECTION - RELEASING LOCK")
    print("="*80)
    
    try:
        if FASTAPI_SINGLETON_LOCK.exists():
            FASTAPI_SINGLETON_LOCK.unlink()
            print(f"✅ Released singleton lock: {FASTAPI_SINGLETON_LOCK}")
        
        if FASTAPI_PID_FILE.exists():
            FASTAPI_PID_FILE.unlink()
            print(f"✅ Removed PID file: {FASTAPI_PID_FILE}")
            
    except Exception as e:
        print(f"[DEBUG] Error releasing singleton lock: {type(e).__name__}: {e}")

# Register cleanup handlers
atexit.register(release_singleton_lock)
signal.signal(signal.SIGTERM, lambda s, f: (release_singleton_lock(), sys.exit(0)))
signal.signal(signal.SIGINT, lambda s, f: (release_singleton_lock(), sys.exit(0)))

# --- PORT CONFLICT DETECTION ---
def is_port_in_use(port: int) -> bool:
    """
    Check if a port is in use by attempting to bind to it.
    
    Args:
        port: Port number to check
        
    Returns:
        bool: True if port is in use, False if available
    """
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            result = sock.connect_ex(('localhost', port))
            is_in_use = result == 0
            
            if is_in_use:
                print(f"[DEBUG] Port {port} is IN USE (connect result: {result})")
                
                # Try to identify what's using the port
                try:
                    import psutil
                    for proc in psutil.process_iter(['pid', 'name', 'connections']):
                        try:
                            connections = proc.info.get('connections', [])
                            if connections:
                                for conn in connections:
                                    if hasattr(conn, 'laddr') and conn.laddr.port == port:
                                        print(f"[DEBUG] Port {port} is used by PID {proc.info['pid']} ({proc.info['name']})")
                                        break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                except ImportError:
                    print(f"[DEBUG] psutil not available for process identification")
            else:
                print(f"[DEBUG] Port {port} is AVAILABLE (connect result: {result})")
                
            return is_in_use
            
    except Exception as e:
        print(f"[DEBUG] Error checking port {port}: {type(e).__name__}: {e}")
        # If we can't check reliably, assume it might be in use for safety
        return True

# Verify port 8001 is available before starting
print("\n" + "="*80)
print("PORT AVAILABILITY CHECK")
print("="*80)

if is_port_in_use(8001):
    print("❌ CRITICAL: Port 8001 is already in use!")
    print("❌ Cannot start FastAPI server on occupied port")
    print("❌ Please check for existing inference servers")
    if os.environ.get("FASTAPI_FORCE_START", "").lower() != "true":
        sys.exit(1)
    else:
        print("⚠️ FORCE START enabled - will attempt to start anyway")
else:
    print("✅ Port 8001 is available for FastAPI server")

# Acquire the singleton lock before doing anything else
create_singleton_lock()

print("="*80 + "\n")

# Print specific environment variables
print("\nIMPORTANT CONFIGURATION:")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"FASTAPI_PORT: {os.environ.get('FASTAPI_PORT', 'Not set')}")
print(f"FASTAPI_INFERENCE_PORT: {os.environ.get('FASTAPI_INFERENCE_PORT', 'Not set')}")
print(f"INFERENCE_API_URL: {os.environ.get('INFERENCE_API_URL', 'Not set')}")
print(f"PROCESS_NAME: {os.environ.get('PROCESS_NAME', 'Not set')}")
print(f"Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Print network info
try:
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\nNETWORK INFO:")
    print(f"Hostname: {hostname}")
    print(f"Local IP: {local_ip}")
    
    # Check if we can bind to the port
    port = int(os.environ.get('FASTAPI_INFERENCE_PORT', '8001'))
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.bind(('0.0.0.0', port))
        print(f"Port {port} is available for binding")
        test_socket.close()
    except Exception as bind_err:
        print(f"[DEBUG] Port binding error type: {type(bind_err).__name__}")
        print(f"[DEBUG] Port binding error details: {bind_err}")
        print(f"Port {port} is NOT available for binding: {bind_err}")
    
    # Try to connect to ourselves
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(1)
        test_result = test_socket.connect_ex(('127.0.0.1', port))
        print(f"Self-connection to 127.0.0.1:{port} result: {test_result} (0=success)")
        test_socket.close()
    except Exception as conn_err:
        print(f"[DEBUG] Self-connection error type: {type(conn_err).__name__}")
        print(f"[DEBUG] Self-connection error details: {conn_err}")
        print(f"Self-connection error: {conn_err}")
    
except Exception as net_err:
    print(f"[DEBUG] Network info error type: {type(net_err).__name__}")
    print(f"[DEBUG] Network info error details: {net_err}")
    print(f"Error getting network info: {net_err}")
    import traceback
    print(f"[DEBUG] Network info error traceback: {traceback.format_exc()}")
    
print("="*80)

# --- FastAPI App ---
# Try to use ProcessWatchdog to ensure only one instance runs and properly handle resources
try:
    from src.lora_training_pipeline.utils.process_watchdog import ProcessWatchdog
    port = int(os.environ.get('FASTAPI_INFERENCE_PORT', '8001'))

    # Create process watchdog for FastAPI inference server
    watchdog = ProcessWatchdog("fastapi_inference", port)
    
    # Check if we can start the server
    if not watchdog.can_start():
        print(f"❌ ERROR: {watchdog.error_message}")
        print("The server cannot start because of resource conflicts.")
        print("To check running services and clean up stale processes:")
        print("python -m src.lora_training_pipeline.utils.process_watchdog --status")
        print("python -m src.lora_training_pipeline.utils.process_watchdog --cleanup")
        sys.exit(1)

    # We can start the server, continue with FastAPI initialization
    app = FastAPI()
    print(f"✅ ProcessWatchdog: Verified no other FastAPI server is running on port {port}")

    # Register the current process PID
    pid = os.getpid()
    if not watchdog.register_pid(pid, {
        "app_type": "fastapi",
        "app_name": "inference_server",
        "start_time": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "model_path": LORA_MODEL_PATH
    }):
        print(f"⚠️ Warning: Failed to register PID: {watchdog.error_message}")
        print("The application will continue, but process management may be affected.")
    
    # The watchdog automatically registers cleanup handlers for signals and atexit

except ImportError:
    # Fall back to standard FastAPI with basic port checking
    from fastapi import FastAPI
    app = FastAPI()
    print("⚠️ ProcessWatchdog not available - using basic port conflict detection")

    # Basic port check
    try:
        port = int(os.environ.get('FASTAPI_INFERENCE_PORT', '8001'))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            result = s.connect_ex(('127.0.0.1', port))
            if result == 0:
                print(f"❌ ERROR: Port {port} is already in use by another process")
                print("The server cannot start because the port is already in use.")
                sys.exit(1)
            else:
                print(f"✅ Port {port} is available for binding")
    except Exception as e:
        print(f"⚠️ Error checking port: {e}")

    # Write standard JSON PID file manually
    try:
        import json
        pid_file = "./inference_process.pid"
        pid = os.getpid()

        with open(pid_file, 'w') as f:
            json.dump({
                "pid": pid,
                "port": int(os.environ.get('FASTAPI_INFERENCE_PORT', '8001')),
                "process_type": "fastapi_inference",
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')
            }, f, indent=2)

        print(f"✅ Wrote PID file {pid_file} with PID {pid}")

        # Register cleanup handler
        import atexit
        def cleanup_pid_file():
            try:
                if os.path.exists(pid_file):
                    os.unlink(pid_file)
                    print(f"Removed PID file {pid_file}")
            except Exception as e:
                print(f"Error removing PID file: {e}")
        atexit.register(cleanup_pid_file)

    except Exception as e:
        print(f"⚠️ Error writing PID file: {e}")

# --- Model and Tokenizer (Global Variables) ---
model = None  # Global variable to store the model
tokenizer = None  # Global variable to store the tokenizer
last_update_check = 0  # Track last time we checked for updates
using_ollama_fallback = False  # Track if we're using Ollama fallback
ollama_model_verified = False  # Track if Ollama model has been verified


def verify_ollama_model(model_name: str = DEFAULT_OLLAMA_MODEL) -> bool:
    """
    Verify that the specified Ollama model is available and working.
    
    Args:
        model_name: Name of the Ollama model to verify
        
    Returns:
        bool: True if model is available and working, False otherwise
    """
    print(f"[DEBUG] === verify_ollama_model() ENTRY ===")
    print(f"[DEBUG] Function called with model_name: {model_name}")
    print(f"[DEBUG] OLLAMA_AVAILABLE global flag: {OLLAMA_AVAILABLE}")
    
    if not OLLAMA_AVAILABLE:
        print(f"[DEBUG] ❌ Ollama library not available, cannot verify model {model_name}")
        print(f"[DEBUG] Reason: Ollama import failed during startup")
        print(f"[DEBUG] === verify_ollama_model() EXIT (early - no library) ===")
        return False
    
    try:
        print(f"[DEBUG] 🔍 Starting verification for Ollama model: {model_name}")
        print(f"[DEBUG] Expected Ollama server URL: {OLLAMA_SERVER_URL}")
        
        # First check if Ollama server is running
        print(f"[DEBUG] Step 1: Testing Ollama server connectivity on port 11434...")
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            print(f"[DEBUG] Created socket with 2-second timeout")
            
            result = sock.connect_ex(('localhost', 11434))
            print(f"[DEBUG] Socket connect_ex result: {result} (0=success, non-zero=failure)")
            
            sock.close()
            print(f"[DEBUG] Socket closed")
            
            if result != 0:
                print(f"[DEBUG] ❌ Ollama server not accessible on port 11434 (connect result: {result})")
                print(f"[DEBUG] This usually means:")
                print(f"[DEBUG] 1. Ollama server is not running")
                print(f"[DEBUG] 2. Ollama is running on a different port")
                print(f"[DEBUG] 3. Firewall is blocking the connection")
                print(f"[DEBUG] === verify_ollama_model() EXIT (server unreachable) ===")
                return False
            
            print(f"[DEBUG] ✅ Ollama server is accessible on port 11434")
        except Exception as socket_err:
            print(f"[DEBUG] ❌ Socket error during server connectivity test: {type(socket_err).__name__}: {socket_err}")
            print(f"[DEBUG] === verify_ollama_model() EXIT (socket error) ===")
            return False
        
        # Try to list models to see if our target model is available
        print(f"[DEBUG] Step 2: Listing available models from Ollama server...")
        try:
            print(f"[DEBUG] Calling ollama.list()...")
            models_response = ollama.list()
            print(f"[DEBUG] ollama.list() returned type: {type(models_response)}")
            print(f"[DEBUG] ollama.list() raw response: {models_response}")
            
            available_models = []
            print(f"[DEBUG] Parsing models from response...")
            
            if hasattr(models_response, 'models'):
                print(f"[DEBUG] Response has 'models' attribute")
                models = models_response.models
                print(f"[DEBUG] models attribute type: {type(models)}")
                print(f"[DEBUG] models attribute value: {models}")
                
                if isinstance(models, list):
                    print(f"[DEBUG] Processing {len(models)} models from list...")
                    for i, model in enumerate(models):
                        print(f"[DEBUG] Model {i}: type={type(model)}, value={model}")
                        if hasattr(model, 'get'):
                            model_name_extracted = model.get('name', model)
                            print(f"[DEBUG] Extracted name via get(): {model_name_extracted}")
                        else:
                            model_name_extracted = str(model)
                            print(f"[DEBUG] Extracted name via str(): {model_name_extracted}")
                        available_models.append(model_name_extracted)
                else:
                    print(f"[DEBUG] ⚠️ models attribute is not a list: {type(models)}")
            elif isinstance(models_response, dict) and 'models' in models_response:
                print(f"[DEBUG] Response is dict with 'models' key")
                models = models_response['models']
                print(f"[DEBUG] models dict value type: {type(models)}")
                print(f"[DEBUG] models dict value: {models}")
                
                if isinstance(models, list):
                    print(f"[DEBUG] Processing {len(models)} models from dict list...")
                    for i, model in enumerate(models):
                        print(f"[DEBUG] Model {i}: type={type(model)}, value={model}")
                        if hasattr(model, 'get'):
                            model_name_extracted = model.get('name', model)
                            print(f"[DEBUG] Extracted name via get(): {model_name_extracted}")
                        else:
                            model_name_extracted = str(model)
                            print(f"[DEBUG] Extracted name via str(): {model_name_extracted}")
                        available_models.append(model_name_extracted)
                else:
                    print(f"[DEBUG] ⚠️ models dict value is not a list: {type(models)}")
            else:
                print(f"[DEBUG] ⚠️ Unexpected response format")
                print(f"[DEBUG] Response type: {type(models_response)}")
                print(f"[DEBUG] Response has 'models' attr: {hasattr(models_response, 'models')}")
                print(f"[DEBUG] Response is dict: {isinstance(models_response, dict)}")
                if isinstance(models_response, dict):
                    print(f"[DEBUG] Dict keys: {list(models_response.keys())}")
            
            print(f"[DEBUG] ✅ Successfully parsed {len(available_models)} available Ollama models")
            print(f"[DEBUG] Available Ollama models: {available_models}")
            
            # Check if our target model is in the list
            print(f"[DEBUG] Step 3: Checking if target model '{model_name}' is available...")
            model_matches = []
            for i, available_model in enumerate(available_models):
                if model_name in available_model:
                    print(f"[DEBUG] Model match found: '{available_model}' contains '{model_name}'")
                    model_matches.append(available_model)
                else:
                    print(f"[DEBUG] Model {i}: '{available_model}' does not contain '{model_name}'")
            
            model_available = len(model_matches) > 0
            print(f"[DEBUG] Target model available: {model_available}")
            if model_available:
                print(f"[DEBUG] Matching models: {model_matches}")
            
            if not model_available:
                print(f"[DEBUG] ❌ Model {model_name} not found in available models")
                print(f"[DEBUG] Step 4: Attempting to pull model {model_name}...")
                try:
                    print(f"[DEBUG] Calling ollama.pull('{model_name}')...")
                    pull_result = ollama.pull(model_name)
                    print(f"[DEBUG] ollama.pull() returned: {pull_result}")
                    print(f"[DEBUG] ✅ Successfully pulled model {model_name}")
                    model_available = True
                    print(f"[DEBUG] Updated model_available to: {model_available}")
                except Exception as pull_err:
                    print(f"[DEBUG] ❌ Failed to pull model {model_name}")
                    print(f"[DEBUG] Pull error type: {type(pull_err).__name__}")
                    print(f"[DEBUG] Pull error details: {pull_err}")
                    import traceback
                    print(f"[DEBUG] Pull error traceback: {traceback.format_exc()}")
                    print(f"[DEBUG] Possible reasons for pull failure:")
                    print(f"[DEBUG] 1. Network connectivity issues")
                    print(f"[DEBUG] 2. Model name is incorrect")
                    print(f"[DEBUG] 3. Ollama server is busy or unresponsive")
                    print(f"[DEBUG] 4. Insufficient disk space")
                    print(f"[DEBUG] === verify_ollama_model() EXIT (pull failed) ===")
                    return False
            else:
                print(f"[DEBUG] ✅ Model {model_name} is already available")
            
            if model_available:
                # Test the model with a simple prompt
                print(f"[DEBUG] Step 5: Testing model {model_name} with simple prompt...")
                test_prompt = "Hello"
                print(f"[DEBUG] Test prompt: '{test_prompt}'")
                
                try:
                    print(f"[DEBUG] Calling ollama.generate(model='{model_name}', prompt='{test_prompt}')...")
                    test_response = ollama.generate(model=model_name, prompt=test_prompt)
                    print(f"[DEBUG] ollama.generate() returned type: {type(test_response)}")
                    print(f"[DEBUG] ollama.generate() raw response: {test_response}")
                    
                    # Parse the response to verify it's valid
                    response_text = None
                    if isinstance(test_response, str):
                        response_text = test_response.strip()
                        print(f"[DEBUG] Parsed response as string: '{response_text}'")
                    elif hasattr(test_response, 'response'):
                        response_text = test_response.response
                        print(f"[DEBUG] Parsed response via .response attribute: '{response_text}'")
                    elif hasattr(test_response, 'get'):
                        response_text = test_response.get('response', str(test_response))
                        print(f"[DEBUG] Parsed response via .get() method: '{response_text}'")
                    else:
                        response_text = str(test_response)
                        print(f"[DEBUG] Parsed response via str(): '{response_text}'")
                    
                    if response_text and len(response_text.strip()) > 0:
                        print(f"[DEBUG] ✅ Model {model_name} verified and working")
                        print(f"[DEBUG] Test response length: {len(response_text)} characters")
                        print(f"[DEBUG] Test response preview: {response_text[:100]}..." if len(response_text) > 100 else f"[DEBUG] Test response: {response_text}")
                        print(f"[DEBUG] === verify_ollama_model() EXIT (success) ===")
                        return True
                    else:
                        print(f"[DEBUG] ❌ Model {model_name} returned empty or invalid response")
                        print(f"[DEBUG] Response text: '{response_text}'")
                        print(f"[DEBUG] === verify_ollama_model() EXIT (empty response) ===")
                        return False
                        
                except Exception as generate_err:
                    print(f"[DEBUG] ❌ Model generation test failed")
                    print(f"[DEBUG] Generation error type: {type(generate_err).__name__}")
                    print(f"[DEBUG] Generation error details: {generate_err}")
                    import traceback
                    print(f"[DEBUG] Generation error traceback: {traceback.format_exc()}")
                    print(f"[DEBUG] === verify_ollama_model() EXIT (generation failed) ===")
                    return False
            else:
                print(f"[DEBUG] ❌ Model {model_name} not available after pull attempt")
                print(f"[DEBUG] === verify_ollama_model() EXIT (model unavailable) ===")
                return False
                
        except Exception as list_err:
            print(f"[DEBUG] ❌ Error during model listing or testing phase")
            print(f"[DEBUG] List/test error type: {type(list_err).__name__}")
            print(f"[DEBUG] List/test error details: {list_err}")
            import traceback
            print(f"[DEBUG] List/test error traceback: {traceback.format_exc()}")
            print(f"[DEBUG] This could indicate:")
            print(f"[DEBUG] 1. Ollama server API incompatibility")
            print(f"[DEBUG] 2. Ollama server internal error")
            print(f"[DEBUG] 3. Network timeout or connectivity issue")
            print(f"[DEBUG] === verify_ollama_model() EXIT (list/test error) ===")
            return False
            
    except Exception as e:
        print(f"[DEBUG] ❌ Unexpected error during Ollama model verification")
        print(f"[DEBUG] Verification error type: {type(e).__name__}")
        print(f"[DEBUG] Verification error details: {e}")
        print(f"[DEBUG] Model being verified: {model_name}")
        import traceback
        print(f"[DEBUG] Verification error traceback: {traceback.format_exc()}")
        print(f"[DEBUG] === verify_ollama_model() EXIT (unexpected error) ===")
        return False

def load_model_and_tokenizer(model_path: str = LORA_MODEL_PATH):
    """
    Loads the base model and LoRA adapter, or falls back to Ollama model.
    
    Args:
        model_path: Path to the LoRA adapter model
        
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global model, tokenizer, using_ollama_fallback, ollama_model_verified  # Declare as global to modify the global variables
    
    print(f"[DEBUG] === load_model_and_tokenizer() ENTRY ===")
    print(f"[DEBUG] Function called with model_path: {model_path}")
    print(f"[DEBUG] Current global state:")
    print(f"[DEBUG]   model: {model}")
    print(f"[DEBUG]   tokenizer: {tokenizer}")
    print(f"[DEBUG]   using_ollama_fallback: {using_ollama_fallback}")
    print(f"[DEBUG]   ollama_model_verified: {ollama_model_verified}")
    print(f"[DEBUG]   OLLAMA_AVAILABLE: {OLLAMA_AVAILABLE}")
    print(f"[DEBUG]   DEFAULT_OLLAMA_MODEL: {DEFAULT_OLLAMA_MODEL}")
    
    # First check if the model path exists and contains files
    if not os.path.exists(model_path):
        print(f"⚠️ Model path does not exist: {model_path}")
        print("ℹ️ Attempting to use Ollama fallback model instead...")
        
        # Try to use Ollama fallback
        if OLLAMA_AVAILABLE and verify_ollama_model(DEFAULT_OLLAMA_MODEL):
            print(f"✅ Using Ollama fallback model: {DEFAULT_OLLAMA_MODEL}")
            model = None  # We don't load the model in memory for Ollama
            tokenizer = None
            using_ollama_fallback = True
            ollama_model_verified = True
            return True
        else:
            print("❌ Ollama fallback not available")
            print("ℹ️ The inference server will start without a model")
            print("ℹ️ Please train a model first or ensure Ollama is running with gemma3:1b")
        
        # Create a directory notice file to help users understand the issue
        try:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            notice_file = Path(model_path).parent / "MODEL_DIRECTORY_MISSING.txt"
            with open(notice_file, 'w') as f:
                f.write(f"""
MODEL DIRECTORY MISSING

The model directory {model_path} does not exist.
This usually happens when:
1. You haven't trained a model yet
2. The training process failed or was interrupted
3. The model directory was accidentally deleted or moved

To fix this issue:
1. Run the training pipeline: python run_pipeline.py
2. Wait for the training to complete successfully
3. The model will be automatically loaded by the inference server

Alternatively, the server can use Ollama fallback model:
1. Install and start Ollama server
2. Pull the gemma3:1b model: ollama pull gemma3:1b
3. Restart this inference server

This notice was created on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
            print(f"ℹ️ Created notice file at {notice_file}")
        except Exception as e:
            print(f"ℹ️ Could not create notice file: {e}")
            
        model = None
        tokenizer = None
        using_ollama_fallback = False
        ollama_model_verified = False
        return False
    elif not any(os.listdir(model_path) if os.path.isdir(model_path) else []):
        print(f"⚠️ Model directory exists but is empty: {model_path}")
        print("ℹ️ Attempting to use Ollama fallback model instead...")
        
        # Try to use Ollama fallback
        if OLLAMA_AVAILABLE and verify_ollama_model(DEFAULT_OLLAMA_MODEL):
            print(f"✅ Using Ollama fallback model: {DEFAULT_OLLAMA_MODEL}")
            model = None  # We don't load the model in memory for Ollama
            tokenizer = None
            using_ollama_fallback = True
            ollama_model_verified = True
            return True
        else:
            print("❌ Ollama fallback not available")
            print("ℹ️ The inference server will start without a model")
            print("ℹ️ Please train a model first or ensure Ollama is running with gemma3:1b")
        
        # Create a directory notice file to help users understand the issue
        try:
            notice_file = Path(model_path) / "EMPTY_MODEL_DIRECTORY.txt"
            with open(notice_file, 'w') as f:
                f.write(f"""
EMPTY MODEL DIRECTORY

The model directory {model_path} exists but contains no model files.
This usually happens when:
1. The training process failed or was interrupted
2. The model files were accidentally deleted
3. The wrong directory was specified

To fix this issue:
1. Run the training pipeline: python run_pipeline.py
2. Wait for the training to complete successfully
3. The model will be automatically loaded by the inference server

Alternatively, the server can use Ollama fallback model:
1. Install and start Ollama server
2. Pull the gemma3:1b model: ollama pull gemma3:1b
3. Restart this inference server

This notice was created on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
            print(f"ℹ️ Created notice file at {notice_file}")
        except Exception as e:
            print(f"ℹ️ Could not create notice file: {e}")
            
        model = None
        tokenizer = None
        using_ollama_fallback = False
        ollama_model_verified = False
        return False
    
    # If we reach here, the model path exists and has files
    # Reset Ollama fallback flags since we're loading a real LoRA model
    using_ollama_fallback = False
    ollama_model_verified = False
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading base model from {BASE_MODEL_NAME}...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device
        )
        
        print(f"Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        
        print(f"Loading LoRA adapter from {model_path}...")
        model = PeftModel.from_pretrained(model, model_path, adapter_name="chat_adapter") # Load the LoRA adapter
        model.eval()  # Set the model to evaluation mode
        print(f"✅ Model and tokenizer loaded successfully from {model_path} on device: {device}")
        using_ollama_fallback = False  # We successfully loaded a LoRA model
        return True

    except Exception as e:
        print(f"⚠️ Error loading model or tokenizer: {e}")
        print("ℹ️ Attempting to use Ollama fallback model instead...")
        
        # Try to use Ollama fallback if LoRA loading fails
        if OLLAMA_AVAILABLE and verify_ollama_model(DEFAULT_OLLAMA_MODEL):
            print(f"✅ Using Ollama fallback model: {DEFAULT_OLLAMA_MODEL}")
            model = None  # We don't load the model in memory for Ollama
            tokenizer = None
            using_ollama_fallback = True
            ollama_model_verified = True
            return True
        else:
            print("❌ Ollama fallback not available")
            print("ℹ️ The inference server will continue running without a model")
            print("ℹ️ You can train a model or fix the issue and restart the server")
            model = None
            tokenizer = None
            using_ollama_fallback = False
            ollama_model_verified = False
            return False

def check_for_model_update():
    """Checks for a model update signal and reloads if needed."""
    global model, tokenizer, last_update_check, LORA_MODEL_PATH

    if time.time() - last_update_check < 5:  # Check at most every 5 seconds
        return

    last_update_check = time.time()

    if MODEL_UPDATE_SIGNAL_FILE.exists():
        try:
            new_model_path = MODEL_UPDATE_SIGNAL_FILE.read_text().strip()
            print(f"Detected model update signal for path: {new_model_path}")
            
            # Check if the model path is different or if we currently have no model loaded
            if new_model_path != LORA_MODEL_PATH or model is None:
                print(f"Model update required. Current: {LORA_MODEL_PATH}, New: {new_model_path}")
                
                # Unload the previous model if it exists
                if model is not None:
                    print("Unloading previous model...")
                    try:
                        del model
                        del tokenizer
                        torch.cuda.empty_cache()  # Clear GPU memory
                    except Exception as e:
                        print(f"Warning during model unloading: {e}")
                
                # Load the new model
                print(f"Loading new model from: {new_model_path}")
                success = load_model_and_tokenizer(new_model_path)
                
                if success:
                    # Update the global path only if loading was successful
                    old_path = LORA_MODEL_PATH
                    LORA_MODEL_PATH = new_model_path
                    print(f"✅ Model path updated: {old_path} → {LORA_MODEL_PATH}")
                else:
                    print(f"⚠️ Failed to load new model. Keeping current path: {LORA_MODEL_PATH}")
            else:
                print(f"Model path unchanged, no reload needed: {LORA_MODEL_PATH}")
            
            # Remove the signal file regardless of outcome
            MODEL_UPDATE_SIGNAL_FILE.unlink()
            
        except Exception as e:
            print(f"⚠️ Error during model update check: {e}")
            try:
                # Always try to remove the signal file to prevent repeated errors
                if MODEL_UPDATE_SIGNAL_FILE.exists():
                    MODEL_UPDATE_SIGNAL_FILE.unlink()
            except Exception as signal_cleanup_err:
                print(f"[DEBUG] Error cleaning signal file: {type(signal_cleanup_err).__name__}: {signal_cleanup_err}")
                pass

@app.on_event("startup")
async def startup_event():
    """Loads the model when the application starts."""
    # If we get here, we've already passed the port conflict check
    # because ServiceManager would have exited the app if conflicts existed

    # Load the model
    try:
        success = load_model_and_tokenizer()
        if success:
            if using_ollama_fallback:
                print(f"✅ Ollama fallback model ({DEFAULT_OLLAMA_MODEL}) loaded successfully at startup")
            else:
                print("✅ LoRA model loaded successfully at startup")
        else:
            print("⚠️ No model available at startup")
            print("ℹ️ The server will continue running with limited functionality")
    except Exception as e:
        print(f"⚠️ Failed to load model at startup: {e}")
        print("ℹ️ The server will continue running, and you can train a model later")
        # Don't re-raise - allow server to start even without a model

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the server is running."""
    # Print detailed debug info when health endpoint is hit
    print("\n" + "="*80)
    print("HEALTH CHECK ENDPOINT HIT")
    print("="*80)
    print(f"Request received at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Get info about the server
        import platform
        import psutil
        
        try:
            # Process information with error handling
            current_process = psutil.Process()
            process_memory = round(current_process.memory_info().rss / (1024 * 1024), 2)
            process_create_time = current_process.create_time()
            process_uptime = time.time() - process_create_time
            process_create_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(process_create_time))
            
            # Get connection information
            connections = []
            try:
                for conn in current_process.connections():
                    connections.append({
                        "fd": conn.fd if hasattr(conn, 'fd') else None,
                        "type": conn.type,
                        "local_addr": f"{conn.laddr.ip}:{conn.laddr.port}" if hasattr(conn, 'laddr') else None,
                        "remote_addr": f"{conn.raddr.ip}:{conn.raddr.port}" if hasattr(conn, 'raddr') and conn.raddr else None,
                        "status": conn.status
                    })
            except Exception as conn_err:
                connections = [{"error": f"Could not get connections: {conn_err}"}]
                
            # Build server info dictionary
            server_info = {
                "python_version": platform.python_version(),
                "python_executable": sys.executable,
                "system": platform.system(),
                "cpu_count": psutil.cpu_count(),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024 * 1024 * 1024), 2),
                "process_memory_mb": process_memory,
                "model_loaded": model is not None,
                "model_type": "LoRA fine-tuned" if model is not None else "No model loaded",
                "fastapi_version": "0.100+",  # Fixed reference to undefined variable
                "pid": os.getpid(),
                "process_create_time": process_create_time_str,
                "process_uptime_seconds": round(process_uptime, 2),
                "process_uptime_human": f"{int(process_uptime // 3600)}h {int((process_uptime % 3600) // 60)}m {int(process_uptime % 60)}s",
                "connections": connections,
                "current_time": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as proc_err:
            # Fallback with minimal server info
            server_info = {
                "python_version": platform.python_version(),
                "system": platform.system(),
                "model_loaded": model is not None,
                "pid": os.getpid(),
                "error": f"Error getting detailed process info: {proc_err}"
            }
        
        print("Server information:")
        for key, value in server_info.items():
            if key != "connections":  # Skip printing the full connections info to keep log cleaner
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {len(value)} connections")
                
        print("="*80)
        
        # Add the server info to the response
        return {
            "status": "ok", 
            "server_running": True,
            "time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "server_info": server_info
        }
    except Exception as e:
        # Error handling for the health check
        import traceback
        stack_trace = traceback.format_exc()
        
        print(f"ERROR in health check: {e}")
        print(f"Stack trace: {stack_trace}")
        
        error_info = {
            "error": str(e),
            "error_type": type(e).__name__,
            "pid": os.getpid(),
            "time": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add stack trace if debug mode
        if os.environ.get("DEBUG_LEVEL", "").upper() == "DEBUG":
            error_info["stack_trace"] = stack_trace
            
        return {
            "status": "error",
            "server_running": True,  # Server is running but health check had an error
            "time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "error_info": error_info
        }

@app.get("/model-info")
async def model_info():
    """Returns information about the loaded model."""
    global model, tokenizer, using_ollama_fallback, ollama_model_verified
    
    # Determine if we have any model available
    if using_ollama_fallback and ollama_model_verified:
        # Using Ollama fallback
        return {
            "model_loaded": True,
            "model_path": None,
            "model_type": f"Ollama fallback model ({DEFAULT_OLLAMA_MODEL})",
            "model_name": DEFAULT_OLLAMA_MODEL,
            "server_url": OLLAMA_SERVER_URL,
            "status": "ready_ollama_fallback"
        }
    elif model is not None and tokenizer is not None:
        # Using LoRA model
        return {
            "model_loaded": True,
            "model_path": LORA_MODEL_PATH,
            "model_type": "LoRA fine-tuned model",
            "model_name": BASE_MODEL_NAME,
            "status": "ready_lora"
        }
    else:
        # No model available
        return {
            "model_loaded": False,
            "model_path": None,
            "model_type": None,
            "model_name": None,
            "status": "no_model"
        }

def generate_with_ollama(prompt: str) -> str:
    """Generate text using Ollama fallback model."""
    print(f"[DEBUG] === generate_with_ollama() ENTRY ===")
    print(f"[DEBUG] Function called with prompt length: {len(prompt)} characters")
    print(f"[DEBUG] Prompt preview: {prompt[:100]}..." if len(prompt) > 100 else f"[DEBUG] Full prompt: {prompt}")
    print(f"[DEBUG] Target Ollama model: {DEFAULT_OLLAMA_MODEL}")
    print(f"[DEBUG] Ollama server URL: {OLLAMA_SERVER_URL}")
    
    try:
        print(f"[DEBUG] 🚀 Starting Ollama generation...")
        print(f"[DEBUG] Calling ollama.generate(model='{DEFAULT_OLLAMA_MODEL}', prompt=...) ...")
        
        # Check if Ollama is still available before generation
        if not OLLAMA_AVAILABLE:
            print(f"[DEBUG] ❌ OLLAMA_AVAILABLE is False, cannot generate")
            raise HTTPException(status_code=503, detail="Ollama library not available")
        
        generation_start_time = time.time()
        response = ollama.generate(model=DEFAULT_OLLAMA_MODEL, prompt=prompt)
        generation_end_time = time.time()
        generation_duration = generation_end_time - generation_start_time
        
        print(f"[DEBUG] ✅ ollama.generate() completed in {generation_duration:.2f} seconds")
        print(f"[DEBUG] Response type: {type(response)}")
        print(f"[DEBUG] Raw response: {response}")
        
        # Parse the response based on its type with detailed logging
        generated_text = None
        parse_method = None
        
        if isinstance(response, str):
            generated_text = response.strip()
            parse_method = "direct string"
            print(f"[DEBUG] Parsed as direct string")
        elif hasattr(response, 'response'):
            generated_text = response.response
            parse_method = ".response attribute"
            print(f"[DEBUG] Parsed via .response attribute")
            print(f"[DEBUG] response.response value: {generated_text}")
        elif hasattr(response, 'get'):
            generated_text = response.get('response', str(response))
            parse_method = ".get() method"
            print(f"[DEBUG] Parsed via .get() method")
            print(f"[DEBUG] .get('response') returned: {generated_text}")
        else:
            generated_text = str(response)
            parse_method = "str() conversion"
            print(f"[DEBUG] Parsed via str() conversion")
        
        print(f"[DEBUG] Parse method used: {parse_method}")
        print(f"[DEBUG] Generated text length: {len(generated_text)} characters")
        print(f"[DEBUG] Generated text preview: {generated_text[:200]}..." if len(generated_text) > 200 else f"[DEBUG] Full generated text: {generated_text}")
        
        if not generated_text or len(generated_text.strip()) == 0:
            print(f"[DEBUG] ❌ Generated text is empty or whitespace-only")
            print(f"[DEBUG] Raw generated_text: '{generated_text}'")
            raise HTTPException(status_code=500, detail="Ollama returned empty response")
        
        print(f"[DEBUG] ✅ Ollama generation successful")
        print(f"[DEBUG] === generate_with_ollama() EXIT (success) ===")
        return generated_text
        
    except HTTPException:
        print(f"[DEBUG] 🔄 Re-raising HTTPException as-is")
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        print(f"[DEBUG] ❌ Ollama generation error occurred")
        print(f"[DEBUG] Error type: {type(e).__name__}")
        print(f"[DEBUG] Error details: {e}")
        print(f"[DEBUG] Prompt that caused error: {prompt}")
        print(f"[DEBUG] Model attempted: {DEFAULT_OLLAMA_MODEL}")
        import traceback
        print(f"[DEBUG] Error traceback: {traceback.format_exc()}")
        print(f"[DEBUG] === generate_with_ollama() EXIT (error) ===")
        
        error_detail = f"Ollama generation error ({type(e).__name__}): {e}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/generate/")
async def generate_text(request_data: Dict):
    """Generates text based on the input prompt."""
    global model, tokenizer, using_ollama_fallback, ollama_model_verified
    
    print(f"[DEBUG] === generate_text() ENTRY ===")
    print(f"[DEBUG] Request received at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DEBUG] Request data type: {type(request_data)}")
    print(f"[DEBUG] Request data: {request_data}")
    
    print(f"[DEBUG] Current model state:")
    print(f"[DEBUG]   model: {model}")
    print(f"[DEBUG]   tokenizer: {tokenizer}")
    print(f"[DEBUG]   using_ollama_fallback: {using_ollama_fallback}")
    print(f"[DEBUG]   ollama_model_verified: {ollama_model_verified}")
    print(f"[DEBUG]   OLLAMA_AVAILABLE: {OLLAMA_AVAILABLE}")
    
    # Validate prompt
    prompt = request_data.get("prompt")
    print(f"[DEBUG] Extracted prompt from request: {prompt}")
    print(f"[DEBUG] Prompt type: {type(prompt)}")
    
    if not prompt:
        print(f"[DEBUG] ❌ Prompt is missing or empty")
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    if not isinstance(prompt, str):
        print(f"[DEBUG] ❌ Prompt is not a string: {type(prompt)}")
        raise HTTPException(status_code=400, detail="Prompt must be a string")
    
    print(f"[DEBUG] ✅ Prompt validation passed")
    print(f"[DEBUG] Prompt length: {len(prompt)} characters")
    print(f"[DEBUG] Prompt preview: {prompt[:100]}..." if len(prompt) > 100 else f"[DEBUG] Full prompt: {prompt}")

    # Check if we should use Ollama fallback
    print(f"[DEBUG] Determining which model to use...")
    print(f"[DEBUG] using_ollama_fallback: {using_ollama_fallback}")
    print(f"[DEBUG] ollama_model_verified: {ollama_model_verified}")
    
    if using_ollama_fallback and ollama_model_verified:
        print(f"[DEBUG] 🎯 Decision: Using Ollama fallback model for generation")
        print(f"[DEBUG] Ollama model: {DEFAULT_OLLAMA_MODEL}")
        
        try:
            print(f"[DEBUG] Calling generate_with_ollama()...")
            generated_text = generate_with_ollama(prompt)
            print(f"[DEBUG] ✅ Ollama generation successful")
            print(f"[DEBUG] Generated text length: {len(generated_text)} characters")
            
            response_data = {
                "generated_text": generated_text,
                "model_used": DEFAULT_OLLAMA_MODEL,
                "model_type": "ollama_fallback",
                "generation_time": time.strftime('%Y-%m-%d %H:%M:%S'),
                "prompt_length": len(prompt),
                "response_length": len(generated_text)
            }
            print(f"[DEBUG] Response data prepared: {response_data}")
            print(f"[DEBUG] === generate_text() EXIT (Ollama success) ===")
            return response_data
            
        except HTTPException as http_err:
            print(f"[DEBUG] 🔄 HTTPException from Ollama generation")
            print(f"[DEBUG] HTTPException status: {http_err.status_code}")
            print(f"[DEBUG] HTTPException detail: {http_err.detail}")
            print(f"[DEBUG] === generate_text() EXIT (Ollama HTTP error) ===")
            raise  # Re-raise HTTP exceptions as-is
        except Exception as ollama_err:
            print(f"[DEBUG] ❌ Unexpected error during Ollama generation")
            print(f"[DEBUG] Ollama error type: {type(ollama_err).__name__}")
            print(f"[DEBUG] Ollama error details: {ollama_err}")
            import traceback
            print(f"[DEBUG] Ollama error traceback: {traceback.format_exc()}")
            print(f"[DEBUG] === generate_text() EXIT (Ollama unexpected error) ===")
            
            error_detail = f"Ollama generation error ({type(ollama_err).__name__}): {ollama_err}"
            raise HTTPException(status_code=500, detail=error_detail)
    
    # Use LoRA model if available
    print(f"[DEBUG] 🎯 Decision: Attempting to use LoRA model for generation")
    print(f"[DEBUG] Checking current LoRA model state...")
    
    # Keep a reference to the current model and tokenizer to prevent them from being
    # deleted during generation if an update happens
    current_model = model
    current_tokenizer = tokenizer
    print(f"[DEBUG] Current model snapshot: {current_model}")
    print(f"[DEBUG] Current tokenizer snapshot: {current_tokenizer}")
    
    # Only check for updates if we're not already in a valid generation context
    if current_model is None or current_tokenizer is None:
        print(f"[DEBUG] LoRA model or tokenizer is None, checking for updates...")
        print(f"[DEBUG] Calling check_for_model_update()...")
        
        try:
            check_for_model_update()  # Check for updates only if we don't have a model yet
            print(f"[DEBUG] ✅ check_for_model_update() completed")
        except Exception as update_err:
            print(f"[DEBUG] ❌ Error during model update check")
            print(f"[DEBUG] Update error type: {type(update_err).__name__}")
            print(f"[DEBUG] Update error details: {update_err}")
            import traceback
            print(f"[DEBUG] Update error traceback: {traceback.format_exc()}")
        
        current_model = model  # Update references
        current_tokenizer = tokenizer
        print(f"[DEBUG] Updated model snapshot: {current_model}")
        print(f"[DEBUG] Updated tokenizer snapshot: {current_tokenizer}")
    else:
        print(f"[DEBUG] LoRA model and tokenizer are available, skipping update check")
    
    # If still no LoRA model and Ollama fallback is not working, return error
    if current_model is None or current_tokenizer is None:
        print(f"[DEBUG] ❌ No LoRA model available after update check")
        print(f"[DEBUG] current_model: {current_model}")
        print(f"[DEBUG] current_tokenizer: {current_tokenizer}")
        print(f"[DEBUG] Attempting emergency Ollama fallback...")
        
        if OLLAMA_AVAILABLE:
            print(f"[DEBUG] Ollama library is available, verifying model...")
            
            try:
                ollama_verification = verify_ollama_model(DEFAULT_OLLAMA_MODEL)
                print(f"[DEBUG] Emergency Ollama verification result: {ollama_verification}")
                
                if ollama_verification:
                    # Try to use Ollama as last resort
                    print(f"[DEBUG] 🚨 Using emergency Ollama fallback for this request")
                    print(f"[DEBUG] Emergency model: {DEFAULT_OLLAMA_MODEL}")
                    
                    try:
                        print(f"[DEBUG] Calling emergency generate_with_ollama()...")
                        generated_text = generate_with_ollama(prompt)
                        print(f"[DEBUG] ✅ Emergency Ollama generation successful")
                        
                        emergency_response = {
                            "generated_text": generated_text,
                            "model_used": DEFAULT_OLLAMA_MODEL,
                            "model_type": "ollama_emergency_fallback",
                            "generation_time": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "prompt_length": len(prompt),
                            "response_length": len(generated_text),
                            "fallback_reason": "LoRA model unavailable"
                        }
                        print(f"[DEBUG] Emergency response prepared: {emergency_response}")
                        print(f"[DEBUG] === generate_text() EXIT (emergency Ollama success) ===")
                        return emergency_response
                        
                    except Exception as emergency_ollama_err:
                        print(f"[DEBUG] ❌ Emergency Ollama generation failed")
                        print(f"[DEBUG] Emergency error type: {type(emergency_ollama_err).__name__}")
                        print(f"[DEBUG] Emergency error details: {emergency_ollama_err}")
                        import traceback
                        print(f"[DEBUG] Emergency error traceback: {traceback.format_exc()}")
                        print(f"[DEBUG] ⚠️ Emergency Ollama fallback failed: {emergency_ollama_err}")
                else:
                    print(f"[DEBUG] ❌ Emergency Ollama verification failed")
            except Exception as verify_err:
                print(f"[DEBUG] ❌ Error during emergency Ollama verification")
                print(f"[DEBUG] Verification error type: {type(verify_err).__name__}")
                print(f"[DEBUG] Verification error details: {verify_err}")
                import traceback
                print(f"[DEBUG] Verification error traceback: {traceback.format_exc()}")
        else:
            print(f"[DEBUG] ❌ Ollama library not available for emergency fallback")
        
        print(f"[DEBUG] 🚫 All model options exhausted")
        print(f"[DEBUG] === generate_text() EXIT (no model available) ===")
        raise HTTPException(
            status_code=503, 
            detail="No model available. Please train a LoRA model or ensure Ollama is running with gemma3:1b."
        )

    try:
        print(f"[DEBUG] 🚀 Starting LoRA model generation")
        print(f"[DEBUG] Using LoRA model for generation")
        print(f"[DEBUG] Model device: {current_model.device}")
        print(f"[DEBUG] Model type: {type(current_model)}")
        
        # Tokenize the input
        print(f"[DEBUG] Tokenizing input prompt...")
        try:
            tokenization_start = time.time()
            input_ids = current_tokenizer(prompt, return_tensors="pt").input_ids
            tokenization_end = time.time()
            print(f"[DEBUG] ✅ Tokenization completed in {tokenization_end - tokenization_start:.3f} seconds")
            print(f"[DEBUG] Input IDs shape: {input_ids.shape}")
            print(f"[DEBUG] Input IDs length: {input_ids.size(1)} tokens")
            
            # Move to device
            print(f"[DEBUG] Moving input to device: {current_model.device}")
            input_ids = input_ids.to(current_model.device)
            print(f"[DEBUG] ✅ Input moved to device successfully")
            
        except Exception as tokenize_err:
            print(f"[DEBUG] ❌ Tokenization error")
            print(f"[DEBUG] Tokenization error type: {type(tokenize_err).__name__}")
            print(f"[DEBUG] Tokenization error details: {tokenize_err}")
            raise tokenize_err
        
        # Generate with the model
        print(f"[DEBUG] Starting model generation...")
        try:
            generation_start = time.time()
            with torch.no_grad():  # Ensure no gradients are calculated
                print(f"[DEBUG] Calling current_model.generate()...")
                output_ids = current_model.generate(
                    input_ids, 
                    max_length=256, 
                    pad_token_id=current_tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.7
                )
            generation_end = time.time()
            print(f"[DEBUG] ✅ Model generation completed in {generation_end - generation_start:.3f} seconds")
            print(f"[DEBUG] Output IDs shape: {output_ids.shape}")
            print(f"[DEBUG] Generated tokens: {output_ids.size(1) - input_ids.size(1)}")
            
        except Exception as generation_err:
            print(f"[DEBUG] ❌ Model generation error")
            print(f"[DEBUG] Generation error type: {type(generation_err).__name__}")
            print(f"[DEBUG] Generation error details: {generation_err}")
            raise generation_err
        
        # Decode the output
        print(f"[DEBUG] Decoding generated tokens...")
        try:
            decode_start = time.time()
            generated_text = current_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            decode_end = time.time()
            print(f"[DEBUG] ✅ Decoding completed in {decode_end - decode_start:.3f} seconds")
            print(f"[DEBUG] Generated text length: {len(generated_text)} characters")
            print(f"[DEBUG] Generated text preview: {generated_text[:200]}..." if len(generated_text) > 200 else f"[DEBUG] Full generated text: {generated_text}")
            
        except Exception as decode_err:
            print(f"[DEBUG] ❌ Decoding error")
            print(f"[DEBUG] Decoding error type: {type(decode_err).__name__}")
            print(f"[DEBUG] Decoding error details: {decode_err}")
            raise decode_err
        
        # Check for model updates after generation
        print(f"[DEBUG] Checking for model updates after generation...")
        try:
            check_for_model_update()
            print(f"[DEBUG] ✅ Post-generation model update check completed")
        except Exception as update_check_err:
            print(f"[DEBUG] ⚠️ Error during post-generation model update check")
            print(f"[DEBUG] Update check error type: {type(update_check_err).__name__}")
            print(f"[DEBUG] Update check error details: {update_check_err}")
            # Don't fail the request due to update check errors
        
        lora_response = {
            "generated_text": generated_text,
            "model_used": BASE_MODEL_NAME,
            "model_type": "lora_finetuned",
            "generation_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "prompt_length": len(prompt),
            "response_length": len(generated_text),
            "input_tokens": input_ids.size(1),
            "generated_tokens": output_ids.size(1) - input_ids.size(1)
        }
        print(f"[DEBUG] LoRA response prepared: {lora_response}")
        print(f"[DEBUG] === generate_text() EXIT (LoRA success) ===")
        return lora_response

    except Exception as lora_err:
        print(f"[DEBUG] ❌ LoRA generation error occurred")
        print(f"[DEBUG] LoRA error type: {type(lora_err).__name__}")
        print(f"[DEBUG] LoRA error details: {lora_err}")
        import traceback
        print(f"[DEBUG] LoRA error traceback: {traceback.format_exc()}")
        
        error_detail = str(lora_err)
        print(f"[DEBUG] ⚠️ LoRA generation error: {error_detail}")
        
        # Try Ollama as emergency fallback if LoRA fails
        print(f"[DEBUG] Attempting emergency Ollama fallback due to LoRA failure...")
        print(f"[DEBUG] OLLAMA_AVAILABLE: {OLLAMA_AVAILABLE}")
        
        if OLLAMA_AVAILABLE:
            print(f"[DEBUG] Ollama library available, verifying model for emergency fallback...")
            
            try:
                emergency_verification = verify_ollama_model(DEFAULT_OLLAMA_MODEL)
                print(f"[DEBUG] Emergency verification result: {emergency_verification}")
                
                if emergency_verification:
                    print(f"[DEBUG] 🚨 LoRA failed, trying Ollama emergency fallback")
                    print(f"[DEBUG] Emergency fallback model: {DEFAULT_OLLAMA_MODEL}")
                    
                    try:
                        print(f"[DEBUG] Calling emergency generate_with_ollama() due to LoRA failure...")
                        generated_text = generate_with_ollama(prompt)
                        print(f"[DEBUG] ✅ Emergency Ollama fallback successful")
                        
                        emergency_fallback_response = {
                            "generated_text": generated_text,
                            "model_used": DEFAULT_OLLAMA_MODEL,
                            "model_type": "ollama_emergency_fallback",
                            "generation_time": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "prompt_length": len(prompt),
                            "response_length": len(generated_text),
                            "lora_error": error_detail,
                            "fallback_reason": "LoRA generation failed"
                        }
                        print(f"[DEBUG] Emergency fallback response prepared: {emergency_fallback_response}")
                        print(f"[DEBUG] === generate_text() EXIT (emergency fallback success) ===")
                        return emergency_fallback_response
                        
                    except Exception as emergency_ollama_err:
                        print(f"[DEBUG] ❌ Emergency Ollama fallback also failed")
                        print(f"[DEBUG] Emergency Ollama error type: {type(emergency_ollama_err).__name__}")
                        print(f"[DEBUG] Emergency Ollama error details: {emergency_ollama_err}")
                        import traceback
                        print(f"[DEBUG] Emergency Ollama error traceback: {traceback.format_exc()}")
                        print(f"[DEBUG] ⚠️ Emergency Ollama fallback failed: {emergency_ollama_err}")
                else:
                    print(f"[DEBUG] ❌ Emergency Ollama verification failed")
            except Exception as emergency_verify_err:
                print(f"[DEBUG] ❌ Error during emergency Ollama verification")
                print(f"[DEBUG] Emergency verify error type: {type(emergency_verify_err).__name__}")
                print(f"[DEBUG] Emergency verify error details: {emergency_verify_err}")
                import traceback
                print(f"[DEBUG] Emergency verify error traceback: {traceback.format_exc()}")
        else:
            print(f"[DEBUG] ❌ Ollama library not available for emergency fallback")
        
        print(f"[DEBUG] 🚫 All generation options failed")
        print(f"[DEBUG] Final error to return: Error during generation: {error_detail}")
        print(f"[DEBUG] === generate_text() EXIT (all options failed) ===")
        raise HTTPException(status_code=500, detail=f"Error during generation: {error_detail}")