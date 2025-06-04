#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/process_management.py

import os
import sys
import socket
import signal
import subprocess
import time
import json
import atexit
import traceback
from pathlib import Path
import psutil
import datetime

# Import from process_core for consolidated implementations
from src.lora_training_pipeline.utils.process_core import (
    FileLock, PidFile, PortManager, ProcessVerifier,
    is_process_running, get_python_executable, log_event,
    cleanup_stale_processes, check_process_health,
    acquire_process_lock, release_process_lock
)

# Enable verbose debugging
DEBUG_MODE = os.environ.get("DEBUG_PROCESS_MANAGEMENT", "true").lower() == "true"

def debug_print(*args, **kwargs):
    """Print debug information only when DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        message = " ".join(str(arg) for arg in args)
        print("[PROCESS-MGMT-DEBUG]", *args, **kwargs)

def error_print(*args, **kwargs):
    """Print error information regardless of debug mode."""
    message = " ".join(str(arg) for arg in args)
    print("[PROCESS-MGMT-ERROR]", *args, file=sys.stderr, **kwargs)

# Set up global exception handler for this module
def setup_process_mgmt_exception_handler():
    """Set up a global exception handler to log unhandled exceptions."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't handle keyboard interrupt specially
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        # Format the traceback
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text = ''.join(tb_lines)
        
        # Log the exception
        log_event("UNHANDLED_EXCEPTION", {
            "exception_type": exc_type.__name__,
            "exception_value": str(exc_value),
            "traceback": tb_text
        })
        
        # Print the exception
        print(f"CRITICAL ERROR: Unhandled exception in process management module:", file=sys.stderr)
        print(tb_text, file=sys.stderr)
        
    # Install the handler
    sys.excepthook = handle_exception
    
# Install exception handler
setup_process_mgmt_exception_handler()

# --- Constants ---
INFERENCE_PROCESS_PID_FILE = Path('./inference_process.pid')
DATA_COLLECTION_PID_FILE = Path('./data_collection_ui.pid')
INFERENCE_UI_PID_FILE = Path('./inference_ui.pid')
PROCESS_LOCKS_DIR = Path('./process_locks')
MAX_LOCK_AGE_SECONDS = 3600  # 1 hour max lock age

# Ensure locks directory exists
PROCESS_LOCKS_DIR.mkdir(parents=True, exist_ok=True)

def _create_lock_path(port, process_type):
    """Create a standardized lock file path for a given port and process type."""
    return PROCESS_LOCKS_DIR / f"{process_type}_{port}.lock"

def is_port_in_use(port):
    """
    Check if a port is already in use.

    Args:
        port: Port number to check

    Returns:
        bool: True if the port is in use, False otherwise
    """
    debug_print(f"Checking if port {port} is in use...")
    try:
        # Use the PortManager class from process_core
        port_manager = PortManager()
        port_in_use = port_manager.is_port_in_use(port)
        debug_print(f"Port {port} check result: {'IN USE' if port_in_use else 'AVAILABLE'}")
        return port_in_use
    except Exception as e:
        print(f"[DEBUG] Port check error type: {type(e).__name__}")
        print(f"[DEBUG] Port check error details: {e}")
        error_msg = f"Unexpected error checking port {port}: {e}"
        print(f"WARNING: {error_msg}")
        log_event("PORT_CHECK_ERROR", {"port": port, "error": str(e), "traceback": traceback.format_exc()})
        # If we can't check reliably, assume it might be in use to be safe
        return True

def write_pid_file(pid_file, pid, metadata=None, use_simple_format=False):
    """
    Write a PID file with optional metadata.

    Args:
        pid_file: Path to the PID file
        pid: Process ID to write
        metadata: Optional dictionary of metadata to include
        use_simple_format: If True, writes PID as plain integer (for backward compatibility)
    """
    try:
        # Ensure pid is an integer
        try:
            pid = int(pid)
        except (ValueError, TypeError) as pid_err:
            print(f"[DEBUG] PID conversion error type: {type(pid_err).__name__}")
            print(f"[DEBUG] PID conversion error details: {pid_err}")
            print(f"WARNING: Invalid PID value: {pid}, using current process PID")
            pid = os.getpid()

        # Use the PidFile class from process_core
        pidfile = PidFile(pid_file)

        # Handle legacy simple format mode for backward compatibility
        if use_simple_format:
            debug_print(f"Writing PID {pid} in simple integer format to {pid_file}")
            # Write the PID directly to the file, bypassing the PidFile class
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            with open(pid_file, 'w') as f:
                f.write(str(pid))
            return True
        else:
            # Use the standard JSON format with PidFile class
            if metadata is None:
                metadata = {}

            debug_print(f"Writing PID {pid} in JSON format to {pid_file} with metadata")
            result = pidfile.write(pid, metadata)

            if not result:
                log_event("PID_FILE_WRITE_ERROR", {
                    "pid_file": str(pid_file),
                    "pid": pid,
                    "metadata": metadata
                })

            return result
    except Exception as e:
        print(f"[DEBUG] PID file write error type: {type(e).__name__}")
        print(f"[DEBUG] PID file write error details: {e}")
        error_msg = f"Error writing PID file {pid_file}: {e}"
        print(f"WARNING: {error_msg}")
        log_event("PID_FILE_WRITE_ERROR", {
            "pid_file": str(pid_file),
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        return False

def read_pid_file(pid_file):
    """
    Read a PID file, returning process ID and metadata.
    Handles both JSON and legacy plain-text PID files to fix ValueError issues.

    Args:
        pid_file: Path to the PID file

    Returns:
        dict: Dictionary with pid and metadata, or None if file doesn't exist/read fails
    """
    try:
        # Use the PidFile class from process_core
        pidfile = PidFile(pid_file)
        data = pidfile.read()

        if data:
            debug_print(f"Successfully read PID file {pid_file} with PID {data['pid']}")
        else:
            debug_print(f"Could not read PID file {pid_file} or file is invalid")

        return data
    except Exception as e:
        print(f"[DEBUG] PID file read error type: {type(e).__name__}")
        print(f"[DEBUG] PID file read error details: {e}")
        error_msg = f"Error reading PID file {pid_file}: {e}"
        print(f"WARNING: {error_msg}")
        debug_print(traceback.format_exc())

        # Log the error for diagnostics
        log_event("PID_FILE_READ_ERROR", {
            "pid_file": str(pid_file),
            "error": str(e),
            "traceback": traceback.format_exc()
        })

        return None

# This function is just a wrapper around the one from process_core
# Kept for backward compatibility
def is_process_running(pid):
    """
    Check if a process with the given PID is running.
    
    Args:
        pid: Process ID to check
        
    Returns:
        bool: True if process is running, False otherwise
    """
    # Use the consolidated implementation from process_core
    # Import locally to avoid recursive calls
    from src.lora_training_pipeline.utils.process_core import is_process_running as core_is_process_running
    return core_is_process_running(pid)

def clean_stale_pid_file(pid_file):
    """
    Remove a PID file if it refers to a non-existent process.
    Handles both JSON and legacy plain-text PID files.

    Args:
        pid_file: Path to PID file

    Returns:
        bool: True if file was removed, False otherwise
    """
    try:
        if not pid_file.exists():
            return False

        # Use the PidFile class from process_core
        pidfile = PidFile(pid_file)
        debug_print(f"Checking if PID file {pid_file} is stale")

        # Check if the file is stale and remove it if it is
        if pidfile.is_stale():
            debug_print(f"PID file {pid_file} is stale, removing it")
            result = pidfile.remove()

            if result:
                log_event("PID_FILE_CLEANED", {
                    "pid_file": str(pid_file),
                    "reason": "Process not running"
                })
                return True
            else:
                log_event("PID_FILE_CLEAN_ERROR", {
                    "pid_file": str(pid_file),
                    "reason": "Failed to remove file"
                })
                return False

        # PID file is not stale - process is still running
        debug_print(f"PID file {pid_file} refers to a running process, keeping it")
        return False

    except Exception as e:
        # Something unexpected happened, log it
        error_msg = f"Unexpected error checking PID file {pid_file}: {e}"
        print(f"WARNING: {error_msg}")
        debug_print(f"{error_msg}\n{traceback.format_exc()}")

        log_event("PID_FILE_CHECK_ERROR", {
            "pid_file": str(pid_file),
            "error": str(e),
            "traceback": traceback.format_exc()
        })

        # In case of unexpected errors, it's safer to keep the PID file
        return False

def acquire_process_lock(port, process_type, metadata=None):
    """
    Attempt to acquire a process lock for the given port and process type.
    
    This function implements a robust locking mechanism to prevent multiple
    processes of the same type from running on the same port. It checks:
    1. If a lock file exists and points to a running process
    2. If the port is already in use
    
    Args:
        port: Port number the process will use
        process_type: Type of process (e.g., 'fastapi', 'gradio')
        metadata: Optional dictionary of metadata to include in lock file
        
    Returns:
        bool: True if lock was acquired, False otherwise
    """
    debug_print(f"Attempting to acquire lock for {process_type} on port {port}...")
    
    # Use the consolidated implementation from process_core
    # Import locally to avoid recursive calls
    from src.lora_training_pipeline.utils.process_core import acquire_process_lock as core_acquire_process_lock
    return core_acquire_process_lock(port, process_type, metadata)

def release_process_lock(port, process_type):
    """
    Release a process lock.
    
    Args:
        port: Port number used by the process
        process_type: Type of process
        
    Returns:
        bool: True if lock was released, False otherwise
    """
    debug_print(f"Releasing lock for {process_type} on port {port}...")
    
    # Use the consolidated implementation from process_core
    # Import locally to avoid recursive calls
    from src.lora_training_pipeline.utils.process_core import release_process_lock as core_release_process_lock
    return core_release_process_lock(port, process_type)

def start_fastapi_inference_server(port, model_path, update_signal_file):
    """
    Start a FastAPI inference server with proper process management.
    
    This function implements robust process management to ensure only
    one FastAPI server runs on the specified port. It:
    1. Checks for existing processes via lock file and port check
    2. Creates a proper lock file with process metadata
    3. Records the PID in the standard location
    
    Args:
        port: Port number to use for the server
        model_path: Path to the model to load
        update_signal_file: Path to the signal file for model updates
        
    Returns:
        subprocess.Popen or None: Process object if started, None otherwise
    """
    # Start detailed logging
    debug_print(f"Starting FastAPI inference server on port {port}")
    debug_print(f"Model path: {model_path}")
    debug_print(f"Update signal file: {update_signal_file}")
    
    startup_info = {
        "port": port,
        "model_path": str(model_path),
        "update_signal_file": str(update_signal_file),
        "stage": "initialization",
        "steps": [],
        "start_time": datetime.datetime.now().isoformat()
    }
    
    # Log the startup attempt
    log_event("FASTAPI_STARTUP_ATTEMPT", {
        "port": port,
        "model_path": str(model_path)
    })
    
    try:
        # Clean up stale PID file if it exists
        debug_print(f"Checking for stale PID file: {INFERENCE_PROCESS_PID_FILE}")
        startup_info["steps"].append("check_stale_pid")
        
        was_stale = clean_stale_pid_file(INFERENCE_PROCESS_PID_FILE)
        startup_info["pid_file_was_stale"] = was_stale
        debug_print(f"Stale PID file found and cleaned: {was_stale}")
        
        # Try to acquire process lock with detailed error tracking
        debug_print(f"Attempting to acquire process lock for port {port}")
        startup_info["steps"].append("acquire_process_lock")
        
        lock_metadata = {
            "model_path": str(model_path),
            "update_signal": str(update_signal_file),
            "start_timestamp": datetime.datetime.now().isoformat()
        }
        
        lock_acquired = acquire_process_lock(port, "fastapi_inference", lock_metadata)
        startup_info["lock_acquired"] = lock_acquired
        debug_print(f"Process lock acquired: {lock_acquired}")
        
        if not lock_acquired:
            # Check if there's a process in the PID file
            debug_print(f"Lock acquisition failed, checking PID file: {INFERENCE_PROCESS_PID_FILE}")
            startup_info["steps"].append("check_pid_file_after_lock_failure")
            
            pid_data = read_pid_file(INFERENCE_PROCESS_PID_FILE)
            startup_info["pid_file_data"] = pid_data
            
            if pid_data and "pid" in pid_data:
                pid = pid_data.get("pid")
                debug_print(f"Found PID {pid} in PID file")
                
                # Check if process is running
                proc_running = is_process_running(pid)
                startup_info["process_running"] = proc_running
                debug_print(f"Process {pid} is running: {proc_running}")
                
                if proc_running:
                    msg = f"FastAPI inference server already running with PID {pid}"
                    print(msg)
                    debug_print(msg)
                    
                    # Verify this is actually a FastAPI server by checking process details
                    try:
                        proc = psutil.Process(pid)
                        cmdline = " ".join([str(c) for c in proc.cmdline()])
                        debug_print(f"Process {pid} command line: {cmdline}")
                        
                        is_fastapi = "fastapi_inference" in cmdline and "uvicorn" in cmdline
                        startup_info["is_actually_fastapi"] = is_fastapi
                        
                        if not is_fastapi:
                            warning_msg = f"WARNING: Process {pid} doesn't appear to be a FastAPI server despite PID file"
                            print(warning_msg)
                            debug_print(warning_msg)
                            
                            # Log this conflicting process
                            log_event("PID_FILE_MISMATCH", {
                                "pid": pid,
                                "cmdline": cmdline,
                                "pid_file": str(INFERENCE_PROCESS_PID_FILE),
                                "pid_file_data": pid_data
                            })
                    except Exception as proc_err:
                        debug_print(f"Error checking process details: {proc_err}")
                    
                    # Log the existing server detection
                    log_event("FASTAPI_ALREADY_RUNNING", {
                        "pid": pid,
                        "port": port,
                        "startup_info": startup_info
                    })
                    
                    return None
            
            # Lock acquisition failed but no valid process found
            error_msg = "Failed to acquire lock for FastAPI server, but no process found in PID file"
            print(error_msg)
            debug_print(error_msg)
            print("This indicates a port conflict with another application")
            
            # Log the lock acquisition failure
            log_event("FASTAPI_LOCK_FAILURE", {
                "port": port,
                "startup_info": startup_info
            })
            
            return None
        
        # Prepare to start the server
        debug_print("Preparing to start FastAPI server")
        startup_info["steps"].append("prepare_environment")
        
        # Create a detailed environment
        env = os.environ.copy()
        env["LORA_MODEL_PATH"] = str(model_path)
        env["MODEL_UPDATE_SIGNAL_FILE"] = str(update_signal_file)
        env["PROCESS_NAME"] = "FastAPIInferenceServer"
        env["PYTHONUNBUFFERED"] = "1"  # Ensure Python output is unbuffered for better logging
        
        # Add process management debugging
        if DEBUG_MODE:
            env["DEBUG_PROCESS_MANAGEMENT"] = "true"
        
        # Start the server with extensive error handling
        debug_print("Launching FastAPI server process")
        startup_info["steps"].append("launch_process")
        
        # Use subprocess.Popen with more detailed configuration and standardized Python executable
        python_exe = get_python_executable()
        debug_print(f"Using Python executable for FastAPI server: {python_exe}")

        cmd = [
            python_exe,
            "-m", "uvicorn",
            "src.lora_training_pipeline.inference.fastapi_inference:app",
            "--host", "0.0.0.0",
            "--port", str(port)
        ]
        
        # In debug mode, add extra flags
        if DEBUG_MODE:
            cmd.append("--log-level=debug")
        else:
            cmd.append("--log-level=info")
            
        # In development mode, add reload flag
        if os.environ.get("DEV_MODE", "").lower() == "true":
            cmd.append("--reload")
            debug_print("Running in development mode with --reload enabled")
        
        debug_print(f"Command: {' '.join(cmd)}")
        startup_info["command"] = " ".join(cmd)
        
        try:
            # Capture stdout and stderr in debug mode
            if DEBUG_MODE:
                debug_print("Capturing server output in debug mode")
                fastapi_log = Path('./fastapi_server.log')
                with open(fastapi_log, 'a') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n\n{'='*50}\n{timestamp} - STARTING FASTAPI SERVER\n{'='*50}\n")
                    
                stdout = open(fastapi_log, 'a')
                stderr = open(fastapi_log, 'a')
                try:
                    process = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=stdout,
                        stderr=stderr
                    )
                except Exception as popen_err:
                    print(f"[DEBUG] FastAPI subprocess.Popen error type: {type(popen_err).__name__}")
                    print(f"[DEBUG] FastAPI subprocess.Popen error details: {popen_err}")
                    raise
                # Store file handles to prevent garbage collection
                process._stdout_handle = stdout
                process._stderr_handle = stderr
            else:
                # Without debug mode, redirect to /dev/null
                try:
                    process = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE
                    )
                except Exception as popen_err:
                    print(f"[DEBUG] FastAPI subprocess.Popen (non-debug) error type: {type(popen_err).__name__}")
                    print(f"[DEBUG] FastAPI subprocess.Popen (non-debug) error details: {popen_err}")
                    raise
            
            debug_print(f"Process started with PID: {process.pid}")
            startup_info["process_pid"] = process.pid
            startup_info["steps"].append("process_started")
            
            # Wait a moment for the server to start
            debug_print("Waiting for server initialization (1 second)")
            time.sleep(1)
            
            # Check if process is still running
            is_running = process.poll() is None
            debug_print(f"Process running after 1 second: {is_running}")
            startup_info["process_running_after_1s"] = is_running
            
            if not is_running:
                error_code = process.returncode
                error_msg = f"FastAPI server failed to start (exit code: {error_code})"
                print(f"ERROR: {error_msg}")
                debug_print(error_msg)
                
                # Try to get error output if available
                if hasattr(process, 'stderr') and process.stderr:
                    try:
                        stderr_output = process.stderr.read().decode('utf-8', errors='replace')
                        debug_print(f"Process stderr: {stderr_output}")
                        startup_info["stderr_output"] = stderr_output
                    except Exception as stderr_err:
                        print(f"[DEBUG] FastAPI stderr read error type: {type(stderr_err).__name__}")
                        print(f"[DEBUG] FastAPI stderr read error details: {stderr_err}")
                        debug_print(f"Error reading stderr: {stderr_err}")
                
                # Release lock
                debug_print("Releasing process lock after startup failure")
                startup_info["steps"].append("release_lock_after_failure")
                release_process_lock(port, "fastapi_inference")
                
                # Log the startup failure
                log_event("FASTAPI_STARTUP_FAILURE", {
                    "port": port,
                    "exit_code": error_code,
                    "startup_info": startup_info
                })
                
                return None
            
            # Write PID file with detailed metadata
            debug_print(f"Writing PID file: {INFERENCE_PROCESS_PID_FILE}")
            startup_info["steps"].append("write_pid_file")
            
            pid_metadata = {
                "port": port,
                "model_path": str(model_path),
                "command": " ".join(cmd),
                "process_type": "FastAPIInferenceServer",
                "start_time": datetime.datetime.now().isoformat(),
                "python_executable": sys.executable,
                "hostname": socket.gethostname()
            }
            
            write_pid_file(INFERENCE_PROCESS_PID_FILE, process.pid, pid_metadata)
            debug_print("PID file written successfully")
            startup_info["steps"].append("pid_file_written")
            
            # Wait longer to allow server to fully initialize
            debug_print("Waiting for server to fully initialize (4 seconds)")
            time.sleep(4)
            
            # Verify process is still running after initialization period
            is_running = process.poll() is None
            debug_print(f"Process running after initialization period: {is_running}")
            startup_info["process_running_after_init"] = is_running
            
            if not is_running:
                error_code = process.returncode
                error_msg = f"FastAPI server failed during initialization (exit code: {error_code})"
                print(f"ERROR: {error_msg}")
                debug_print(error_msg)
                
                # Try to get error output if available
                if hasattr(process, 'stderr') and process.stderr:
                    try:
                        stderr_output = process.stderr.read().decode('utf-8', errors='replace')
                        debug_print(f"Process stderr: {stderr_output}")
                        startup_info["stderr_output"] = stderr_output
                    except Exception as stderr_err:
                        print(f"[DEBUG] FastAPI stderr read error type: {type(stderr_err).__name__}")
                        print(f"[DEBUG] FastAPI stderr read error details: {stderr_err}")
                        debug_print(f"Error reading stderr: {stderr_err}")
                
                # Release lock and clean up
                debug_print("Releasing process lock after initialization failure")
                startup_info["steps"].append("release_lock_after_init_failure")
                release_process_lock(port, "fastapi_inference")
                
                if INFERENCE_PROCESS_PID_FILE.exists():
                    try:
                        INFERENCE_PROCESS_PID_FILE.unlink()
                        debug_print("Removed PID file after initialization failure")
                    except Exception as unlink_err:
                        debug_print(f"Error removing PID file: {unlink_err}")
                
                # Log the initialization failure
                log_event("FASTAPI_INIT_FAILURE", {
                    "port": port,
                    "exit_code": error_code,
                    "startup_info": startup_info
                })
                
                return None
            
            # Verify port is actually in use
            debug_print(f"Verifying port {port} is in use")
            startup_info["steps"].append("verify_port")
            
            port_active = is_port_in_use(port)
            debug_print(f"Port {port} active: {port_active}")
            startup_info["port_active"] = port_active
            
            if not port_active:
                warning_msg = f"WARNING: Process is running but port {port} is not in use - possible misconfiguration"
                print(warning_msg)
                debug_print(warning_msg)
                startup_info["port_binding_issue"] = True
                
                # Log the port binding issue
                log_event("FASTAPI_PORT_BINDING_ISSUE", {
                    "port": port,
                    "pid": process.pid,
                    "startup_info": startup_info
                })
            
            # Success - server is running
            success_msg = f"FastAPI inference server started successfully on port {port} (PID: {process.pid})"
            print(success_msg)
            debug_print(success_msg)
            startup_info["steps"].append("server_started_successfully")
            
            # Log the successful startup
            startup_info["end_time"] = datetime.datetime.now().isoformat()
            log_event("FASTAPI_STARTUP_SUCCESS", {
                "port": port,
                "pid": process.pid,
                "startup_info": startup_info
            })
            
            return process
            
        except Exception as e:
            error_msg = f"Error starting FastAPI server: {e}"
            print(f"ERROR: {error_msg}")
            debug_print(f"{error_msg}\n{traceback.format_exc()}")
            startup_info["error"] = str(e)
            startup_info["traceback"] = traceback.format_exc()
            startup_info["steps"].append("error_during_startup")
            
            # Release lock
            debug_print("Releasing process lock after error")
            startup_info["steps"].append("release_lock_after_error")
            
            try:
                release_process_lock(port, "fastapi_inference")
                debug_print("Process lock released successfully")
            except Exception as release_err:
                debug_print(f"Error releasing process lock: {release_err}")
                startup_info["lock_release_error"] = str(release_err)
            
            # Log the startup error
            startup_info["end_time"] = datetime.datetime.now().isoformat()
            log_event("FASTAPI_STARTUP_ERROR", {
                "port": port,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "startup_info": startup_info
            })
            
            return None
            
    except Exception as e:
        # Catch any unexpected errors in the startup process itself
        error_msg = f"Unexpected error in start_fastapi_inference_server: {e}"
        print(f"CRITICAL ERROR: {error_msg}")
        debug_print(f"{error_msg}\n{traceback.format_exc()}")
        
        # Try to clean up if possible
        try:
            if 'lock_acquired' in locals() and lock_acquired:
                debug_print("Attempting to release lock after critical error")
                release_process_lock(port, "fastapi_inference")
        except Exception as cleanup_err:
            debug_print(f"Error during cleanup after critical error: {cleanup_err}")
        
        # Log the critical error
        log_event("CRITICAL_FASTAPI_STARTUP_ERROR", {
            "port": port,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        
        return None

def start_gradio_data_collection(port):
    """
    Start a Gradio data collection UI with proper process management.

    Args:
        port: Port number to use for the UI

    Returns:
        subprocess.Popen or None: Process object if started, None otherwise
    """
    # Clean up stale PID file if it exists
    clean_stale_pid_file(DATA_COLLECTION_PID_FILE)

    # Try to acquire process lock
    if not acquire_process_lock(port, "gradio_data_collection"):
        # Check if there's a process in the PID file
        pid_data = read_pid_file(DATA_COLLECTION_PID_FILE)
        if pid_data and is_process_running(pid_data.get("pid", 0)):
            print(f"Gradio data collection UI already running with PID {pid_data.get('pid')}")
            return None

        print("Failed to acquire lock for Gradio data collection UI, but no process found in PID file")
        print("This indicates a port conflict with another application")
        return None

    # Start the Gradio app
    env = os.environ.copy()
    env["PROCESS_NAME"] = "GradioDataCollection"
    env["GRADIO_PORT"] = str(port)
    # Add a signal handling flag to prevent race conditions with other Gradio processes
    env["GRADIO_DISABLE_SIGNAL_HANDLERS"] = "1"
    # Add debug flag to diagnose issues
    env["GRADIO_DEBUG"] = "1"

    try:
        # Get the standardized Python executable
        python_exe = get_python_executable()
        debug_print(f"Using Python executable for Gradio Data Collection: {python_exe}")

        # Use DETACHED process creation flags to avoid shared signal handling
        creationflags = 0
        if sys.platform == 'win32':
            # On Windows, use CREATE_NEW_PROCESS_GROUP flag
            import subprocess
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        # Additional platform-specific arguments for subprocess.Popen
        proc_kwargs = {
            'env': env,
            'creationflags': creationflags if sys.platform == 'win32' else 0,
            # Redirect output to avoid blocking
            'stdout': subprocess.PIPE if DEBUG_MODE else subprocess.DEVNULL,
            'stderr': subprocess.PIPE if DEBUG_MODE else subprocess.DEVNULL,
            # Set start_new_session flag for Unix-like systems
            'start_new_session': sys.platform != 'win32'
        }

        # Start the process with platform-specific arguments
        try:
            process = subprocess.Popen(
                [python_exe, "-m", "src.lora_training_pipeline.data_collection.gradio_app"],
                **proc_kwargs
            )
        except Exception as popen_err:
            print(f"[DEBUG] Gradio data collection subprocess.Popen error type: {type(popen_err).__name__}")
            print(f"[DEBUG] Gradio data collection subprocess.Popen error details: {popen_err}")
            print(f"[DEBUG] Gradio data collection subprocess.Popen command: {[python_exe, '-m', 'src.lora_training_pipeline.data_collection.gradio_app']}")
            print(f"[DEBUG] Gradio data collection subprocess.Popen kwargs: {proc_kwargs}")
            raise

        # Wait a moment for the process to start
        time.sleep(1)

        # Check if process is still running
        if process.poll() is not None:
            print(f"Gradio data collection UI failed to start (exit code: {process.returncode})")
            # Attempt to get stderr if available
            if DEBUG_MODE and hasattr(process, 'stderr') and process.stderr:
                try:
                    stderr_output = process.stderr.read().decode('utf-8', errors='replace')
                    debug_print(f"Process stderr: {stderr_output}")
                except Exception as stderr_err:
                    print(f"[DEBUG] Gradio data collection stderr read error type: {type(stderr_err).__name__}")
                    print(f"[DEBUG] Gradio data collection stderr read error details: {stderr_err}")
                    debug_print(f"Error reading stderr: {stderr_err}")
            # Release lock
            release_process_lock(port, "gradio_data_collection")
            return None

        # Write PID file with metadata
        write_pid_file(DATA_COLLECTION_PID_FILE, process.pid, {
            "port": port,
            "command": "python -m src.lora_training_pipeline.data_collection.gradio_app",
            "process_type": "GradioDataCollection"
        })

        # Wait a bit longer to allow the UI to fully initialize
        time.sleep(2)
        print(f"Gradio data collection UI started on port {port} (PID: {process.pid})")

        return process
    except Exception as e:
        print(f"Error starting Gradio data collection UI: {e}")
        # Release lock
        release_process_lock(port, "gradio_data_collection")
        return None

def start_gradio_inference_ui(port, api_url):
    """
    Start a Gradio inference UI with proper process management.

    Args:
        port: Port number to use for the UI
        api_url: URL of the FastAPI inference server

    Returns:
        subprocess.Popen or None: Process object if started, None otherwise
    """
    # Clean up stale PID file if it exists
    clean_stale_pid_file(INFERENCE_UI_PID_FILE)

    # Try to acquire process lock
    if not acquire_process_lock(port, "gradio_inference_ui"):
        # Check if there's a process in the PID file
        pid_data = read_pid_file(INFERENCE_UI_PID_FILE)
        if pid_data and is_process_running(pid_data.get("pid", 0)):
            print(f"Gradio inference UI already running with PID {pid_data.get('pid')}")
            return None

        print("Failed to acquire lock for Gradio inference UI, but no process found in PID file")
        print("This indicates a port conflict with another application")
        return None

    # Start the Gradio app
    env = os.environ.copy()
    env["PROCESS_NAME"] = "GradioInferenceUI"
    env["GRADIO_PORT"] = str(port)
    env["INFERENCE_API_URL"] = api_url
    # Add a signal handling flag to prevent race conditions with other Gradio processes
    env["GRADIO_DISABLE_SIGNAL_HANDLERS"] = "1"
    # Add debug flag to diagnose issues
    env["GRADIO_DEBUG"] = "1"
    # Add staggered startup delay to prevent module import race conditions
    env["GRADIO_STARTUP_DELAY"] = "2"

    try:
        # Get the standardized Python executable
        python_exe = get_python_executable()
        debug_print(f"Using Python executable for Gradio Inference UI: {python_exe}")

        # Use DETACHED process creation flags to avoid shared signal handling
        creationflags = 0
        if sys.platform == 'win32':
            # On Windows, use CREATE_NEW_PROCESS_GROUP flag
            import subprocess
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        # Additional platform-specific arguments for subprocess.Popen
        proc_kwargs = {
            'env': env,
            'creationflags': creationflags if sys.platform == 'win32' else 0,
            # Redirect output to avoid blocking
            'stdout': subprocess.PIPE if DEBUG_MODE else subprocess.DEVNULL,
            'stderr': subprocess.PIPE if DEBUG_MODE else subprocess.DEVNULL,
            # Set start_new_session flag for Unix-like systems
            'start_new_session': sys.platform != 'win32'
        }

        # Start the process with platform-specific arguments
        try:
            process = subprocess.Popen(
                [python_exe, "-m", "src.lora_training_pipeline.inference.gradio_inference"],
                **proc_kwargs
            )
        except Exception as popen_err:
            print(f"[DEBUG] Gradio inference UI subprocess.Popen error type: {type(popen_err).__name__}")
            print(f"[DEBUG] Gradio inference UI subprocess.Popen error details: {popen_err}")
            print(f"[DEBUG] Gradio inference UI subprocess.Popen command: {[python_exe, '-m', 'src.lora_training_pipeline.inference.gradio_inference']}")
            print(f"[DEBUG] Gradio inference UI subprocess.Popen kwargs: {proc_kwargs}")
            raise

        # Wait a moment for the process to start
        time.sleep(1)

        # Check if process is still running
        if process.poll() is not None:
            print(f"Gradio inference UI failed to start (exit code: {process.returncode})")
            # Attempt to get stderr if available
            if DEBUG_MODE and hasattr(process, 'stderr') and process.stderr:
                try:
                    stderr_output = process.stderr.read().decode('utf-8', errors='replace')
                    debug_print(f"Process stderr: {stderr_output}")
                except Exception as stderr_err:
                    print(f"[DEBUG] Gradio inference UI stderr read error type: {type(stderr_err).__name__}")
                    print(f"[DEBUG] Gradio inference UI stderr read error details: {stderr_err}")
                    debug_print(f"Error reading stderr: {stderr_err}")
            # Release lock
            release_process_lock(port, "gradio_inference_ui")
            return None

        # Write PID file with metadata
        write_pid_file(INFERENCE_UI_PID_FILE, process.pid, {
            "port": port,
            "api_url": api_url,
            "command": "python -m src.lora_training_pipeline.inference.gradio_inference",
            "process_type": "GradioInferenceUI"
        })

        # Wait a bit longer to allow the UI to fully initialize
        # For the inference UI, we give it more time as it might need to load models
        time.sleep(3)
        print(f"Gradio inference UI started on port {port} (PID: {process.pid})")

        return process
    except Exception as e:
        print(f"Error starting Gradio inference UI: {e}")
        # Release lock
        release_process_lock(port, "gradio_inference_ui")
        return None

def detect_stale_processes(process_type=None):
    """
    Detect stale processes based on PID files.

    Args:
        process_type: Optional filter for specific process type

    Returns:
        list: List of dictionaries containing stale process information
    """
    stale_processes = []

    # Define PID files to check based on process_type
    pid_files = []
    if process_type is None or process_type == "fastapi":
        pid_files.append((INFERENCE_PROCESS_PID_FILE, "FastAPIInferenceServer"))
    if process_type is None or process_type == "data_collection":
        pid_files.append((DATA_COLLECTION_PID_FILE, "GradioDataCollection"))
    if process_type is None or process_type == "inference_ui":
        pid_files.append((INFERENCE_UI_PID_FILE, "GradioInferenceUI"))

    debug_print(f"Checking {len(pid_files)} PID files for stale processes")

    # Check each PID file with robust error handling
    for pid_file, proc_type in pid_files:
        try:
            if not pid_file.exists():
                debug_print(f"PID file {pid_file} does not exist, skipping")
                continue

            # Use the PidFile class from process_core
            pidfile = PidFile(pid_file)

            # Check if the PID file is stale
            if pidfile.is_stale():
                debug_print(f"PID file {pid_file} is stale")
                # Get the data from the file before we mark it as stale
                pid_data = pidfile.read()
                pid = pid_data.get("pid") if pid_data else None

                stale_processes.append({
                    "type": proc_type,
                    "pid": pid,
                    "pid_file": str(pid_file),
                    "reason": "Process not running",
                    "data": pid_data
                })

        except Exception as e:
            # Log any unexpected errors during checking
            error_msg = f"Error checking PID file {pid_file}: {e}"
            print(f"WARNING: {error_msg}")
            debug_print(f"{error_msg}\n{traceback.format_exc()}")

            # Log this error for diagnostics
            log_event("STALE_PROCESS_CHECK_ERROR", {
                "pid_file": str(pid_file),
                "error": str(e),
                "traceback": traceback.format_exc()
            })

            # Consider the file stale if we can't verify it
            stale_processes.append({
                "type": proc_type,
                "pid_file": str(pid_file),
                "reason": f"Error checking file: {e}",
                "error": str(e)
            })

    # Check process lock directory for stale locks with robust error handling
    try:
        debug_print(f"Checking process lock directory: {PROCESS_LOCKS_DIR}")
        if not PROCESS_LOCKS_DIR.exists():
            debug_print(f"Process lock directory does not exist, creating it")
            PROCESS_LOCKS_DIR.mkdir(parents=True, exist_ok=True)

        # Use glob with error handling to find lock files
        lock_files = list(PROCESS_LOCKS_DIR.glob("*.lock"))
        debug_print(f"Found {len(lock_files)} lock files to check")

        for lock_file in lock_files:
            try:
                debug_print(f"Checking lock file: {lock_file}")

                # Use the FileLock and PidFile classes to check if the lock is stale
                file_lock = FileLock(lock_file)
                # If we can acquire the lock, it means it's stale
                if file_lock.acquire(timeout=0):
                    debug_print(f"Lock file {lock_file} is stale (could acquire lock)")
                    file_lock.release()

                    # Try to get any data from the lock file for reporting
                    pidfile = PidFile(lock_file)
                    pid_data = pidfile.read()
                    pid = pid_data.get("pid") if pid_data else None
                    proc_type = pid_data.get("process_type", os.path.basename(lock_file).split("_")[0]) if pid_data else "Unknown"

                    # Skip if we're filtering and this isn't the type we want
                    if process_type and process_type != proc_type.lower():
                        debug_print(f"Skipping process type {proc_type} due to filter {process_type}")
                        continue

                    stale_processes.append({
                        "type": proc_type,
                        "pid": pid,
                        "lock_file": str(lock_file),
                        "reason": "Lock is stale",
                        "data": pid_data
                    })
                else:
                    debug_print(f"Lock file {lock_file} is still active (could not acquire lock)")

            except Exception as e:
                # Log any unexpected errors during lock file checking
                error_msg = f"Error checking lock file {lock_file}: {e}"
                print(f"WARNING: {error_msg}")
                debug_print(f"{error_msg}\n{traceback.format_exc()}")

                # Log this error for diagnostics
                log_event("LOCK_FILE_CHECK_ERROR", {
                    "lock_file": str(lock_file),
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })

                # Consider the lock file stale if we can't verify it
                stale_processes.append({
                    "type": "Unknown",
                    "lock_file": str(lock_file),
                    "reason": f"Error checking file: {e}",
                    "error": str(e)
                })

    except Exception as e:
        # Handle errors in the lock directory iteration itself
        error_msg = f"Error checking process lock directory: {e}"
        print(f"WARNING: {error_msg}")
        debug_print(f"{error_msg}\n{traceback.format_exc()}")

        # Log this error for diagnostics
        log_event("PROCESS_LOCK_DIR_ERROR", {
            "error": str(e),
            "traceback": traceback.format_exc()
        })

    return stale_processes

def cleanup_stale_processes():
    """
    Clean up stale processes and their associated files.
    
    Returns:
        dict: Information about cleanup results
    """
    # Use the consolidated implementation from process_core
    # Import locally to avoid recursive calls
    from src.lora_training_pipeline.utils.process_core import cleanup_stale_processes as core_cleanup_stale_processes
    return core_cleanup_stale_processes([
        INFERENCE_PROCESS_PID_FILE,
        DATA_COLLECTION_PID_FILE,
        INFERENCE_UI_PID_FILE
    ])

def check_process_health():
    """
    Check the health of all managed processes.
    
    Returns:
        dict: Dictionary of process status information
    """
    # Use the consolidated implementation from process_core
    # Import locally to avoid recursive calls
    from src.lora_training_pipeline.utils.process_core import check_process_health as core_check_process_health
    return core_check_process_health([
        INFERENCE_PROCESS_PID_FILE,
        DATA_COLLECTION_PID_FILE,
        INFERENCE_UI_PID_FILE
    ])