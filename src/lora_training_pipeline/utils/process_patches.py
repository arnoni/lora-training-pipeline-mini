#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/process_patches.py

"""
Patch module for the existing LoRA Training Pipeline code to fix process management issues.
This module provides drop-in replacement functions for zenml_pipeline.py and run_pipeline.py.
"""

import os
import sys
import time
import signal
import subprocess
import traceback
from pathlib import Path

# Enable debug mode
DEBUG_MODE = os.environ.get("DEBUG_PROCESS_MANAGEMENT", "true").lower() == "true"

# Add a flag for controlling sequential startup
SEQUENTIAL_START = os.environ.get("SEQUENTIAL_PROCESS_START", "true").lower() == "true"

# Setup error logging
def log_process_error(error_type, details, error=None):
    """Log process errors to file for debugging"""
    try:
        error_log = Path('./process_errors.log')
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create error entry
        error_entry = {
            "timestamp": timestamp,
            "type": error_type,
            "details": details
        }
        
        # Add error info if provided
        if error:
            error_entry["error"] = str(error)
            error_entry["traceback"] = traceback.format_exc()
            
        # Write to log file
        with open(error_log, 'a') as f:
            import json
            f.write(json.dumps(error_entry) + "\n")
            
        print(f"ERROR: {error_type} - {details}" + (f" - {error}" if error else ""))
    except Exception as e:
        print(f"Failed to log error: {e}")

# Import process management utilities with error handling
try:
    from src.lora_training_pipeline.utils.process_management import (
        start_fastapi_inference_server,
        start_gradio_data_collection,
        start_gradio_inference_ui,
        cleanup_stale_processes,
        check_process_health,
        is_process_running,
        read_pid_file,
        debug_print,
        log_event
    )
    
    # Log successful import
    if DEBUG_MODE:
        debug_print("Successfully imported process management modules")
except ImportError as e:
    error_msg = f"Failed to import process management utilities: {e}"
    print(f"CRITICAL ERROR: {error_msg}")
    log_process_error("IMPORT_ERROR", "Failed to import process management modules", e)
    
    # Define minimal versions of needed functions
    def debug_print(*args, **kwargs):
        if DEBUG_MODE:
            print("[DEBUG]", *args, **kwargs)
            
    def log_event(event_type, details):
        debug_print(f"Event: {event_type} - {details}")
        
# Set up global exception handler for this module
def setup_exception_handler():
    """Set up global exception handler for debugging"""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't handle keyboard interrupt specially
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        # Log the exception
        log_process_error(
            "UNHANDLED_EXCEPTION", 
            f"Unhandled {exc_type.__name__} in process patches module",
            exc_value
        )
        
    # Install the handler
    sys.excepthook = handle_exception
    
# Install exception handler
setup_exception_handler()

# Log module initialization
log_event("PATCHES_MODULE_LOADED", {"timestamp": time.time()})

# --- Constants ---
INFERENCE_PROCESS_PID_FILE = Path('./inference_process.pid')
MODEL_UPDATE_SIGNAL_FILE = Path('./.model_update')
INFERENCE_SERVER_PORT = 8001

def patch_start_inference_server(model_path):
    """
    Patched version of the start_inference_server function in zenml_pipeline.py.
    This version uses the enhanced process management utilities.
    
    Args:
        model_path: Path to the model to load
        
    Returns:
        None
    """
    print("\n" + "="*80)
    print("PATCHED: Starting FastAPI inference server with enhanced process management")
    print("="*80)
    
    print(f"Model path: {model_path}")
    print(f"Signal file: {MODEL_UPDATE_SIGNAL_FILE}")
    print(f"Port: {INFERENCE_SERVER_PORT}")
    
    # Cleanup any stale processes first
    cleanup_result = cleanup_stale_processes()
    if cleanup_result["stale_processes"] > 0:
        print(f"Cleaned up {cleanup_result['stale_processes']} stale processes")
        print(f"Removed {cleanup_result['cleaned_pid_files']} stale PID files")
        print(f"Removed {cleanup_result['cleaned_lock_files']} stale lock files")
        if cleanup_result["errors"]:
            print(f"Encountered {len(cleanup_result['errors'])} errors during cleanup")
    
    # Check if there's already a running server process
    health_info = check_process_health()
    fastapi_info = health_info.get("fastapi_server")
    
    if fastapi_info and fastapi_info.get("status") == "running":
        pid = fastapi_info.get("pid")
        port = fastapi_info.get("port")
        port_active = fastapi_info.get("port_active", False)
        
        print(f"Found existing FastAPI server (PID: {pid}, Port: {port}, Active: {port_active})")
        
        if port_active:
            print("Server port is active, sending model update signal")
            # Signal model update instead of starting a new server
            MODEL_UPDATE_SIGNAL_FILE.write_text(str(model_path))
            print(f"✅ Model update signaled. New model path: {model_path}")
            return
        else:
            print("Server process exists but port is not active - possible zombie process")
            print("Will attempt to start a new server")
    
    # Start a new server
    process = start_fastapi_inference_server(
        port=INFERENCE_SERVER_PORT,
        model_path=model_path,
        update_signal_file=MODEL_UPDATE_SIGNAL_FILE
    )
    
    if process:
        print(f"✅ FastAPI server started successfully on port {INFERENCE_SERVER_PORT}")
        print(f"PID: {process.pid}")
    else:
        print("❌ Failed to start FastAPI server")
        print("This could indicate a port conflict or other system issue")
        
    print("="*80)

def patch_start_gradio_data_collection(port):
    """
    Patched version of the start_gradio_data_collection function.
    
    Args:
        port: Port number to use for the UI
        
    Returns:
        subprocess.Popen or None: Process object if started, None otherwise
    """
    print("\n" + "="*80)
    print(f"PATCHED: Starting Gradio Data Collection UI on port {port}")
    print("="*80)
    
    # Cleanup any stale processes first
    cleanup_result = cleanup_stale_processes()
    if cleanup_result["stale_processes"] > 0:
        print(f"Cleaned up {cleanup_result['stale_processes']} stale processes")
    
    # Check existing processes
    health_info = check_process_health()
    ui_info = health_info.get("data_collection_ui")
    
    if ui_info and ui_info.get("status") == "running":
        pid = ui_info.get("pid")
        current_port = ui_info.get("port")
        port_active = ui_info.get("port_active", False)
        
        print(f"Found existing Gradio Data Collection UI (PID: {pid}, Port: {current_port}, Active: {port_active})")
        
        if port_active and str(current_port) == str(port):
            print("UI is already running on the requested port")
            return None
    
    # Start a new UI instance
    process = start_gradio_data_collection(port=port)
    
    if process:
        print(f"✅ Gradio Data Collection UI started successfully on port {port}")
        print(f"PID: {process.pid}")
        return process
    else:
        print("❌ Failed to start Gradio Data Collection UI")
        print("This could indicate a port conflict or other system issue")
        return None

def patch_start_gradio_inference_ui(port, api_url):
    """
    Patched version of the start_gradio_inference_ui function.
    
    Args:
        port: Port number to use for the UI
        api_url: URL of the FastAPI inference server
        
    Returns:
        subprocess.Popen or None: Process object if started, None otherwise
    """
    print("\n" + "="*80)
    print(f"PATCHED: Starting Gradio Inference UI on port {port}")
    print("="*80)
    
    print(f"API URL: {api_url}")
    
    # Cleanup any stale processes first
    cleanup_result = cleanup_stale_processes()
    if cleanup_result["stale_processes"] > 0:
        print(f"Cleaned up {cleanup_result['stale_processes']} stale processes")
    
    # Check existing processes
    health_info = check_process_health()
    ui_info = health_info.get("inference_ui")
    
    if ui_info and ui_info.get("status") == "running":
        pid = ui_info.get("pid")
        current_port = ui_info.get("port")
        port_active = ui_info.get("port_active", False)
        
        print(f"Found existing Gradio Inference UI (PID: {pid}, Port: {current_port}, Active: {port_active})")
        
        if port_active and str(current_port) == str(port):
            print("UI is already running on the requested port")
            return None
    
    # Start a new UI instance
    process = start_gradio_inference_ui(port=port, api_url=api_url)
    
    if process:
        print(f"✅ Gradio Inference UI started successfully on port {port}")
        print(f"PID: {process.pid}")
        return process
    else:
        print("❌ Failed to start Gradio Inference UI")
        print("This could indicate a port conflict or other system issue")
        return None

def signal_model_update(model_path):
    """
    Patched version of the signal_model_update function.
    This version checks if the FastAPI server is actually running before sending the signal.
    
    Args:
        model_path: Path to the model to load
        
    Returns:
        bool: True if signal was sent successfully, False otherwise
    """
    # Check if the server is running
    health_info = check_process_health()
    fastapi_info = health_info.get("fastapi_server")
    
    if not fastapi_info or fastapi_info.get("status") != "running":
        print("❌ No running FastAPI server found to signal")
        return False
    
    # Check if the port is active
    if not fastapi_info.get("port_active", False):
        print("❌ FastAPI server process exists but port is not active")
        print("The server may not be responsive to signals")
    
    # Send the signal
    try:
        MODEL_UPDATE_SIGNAL_FILE.write_text(str(model_path))
        print(f"✅ Model update signaled. New model path: {model_path}")
        return True
    except Exception as e:
        print(f"❌ Error signaling model update: {e}")
        return False

def patch_launch_or_update_inference(model_path):
    """
    Patched version of the launch_or_update_inference function in zenml_pipeline.py.
    
    Args:
        model_path: Path to the model to load
        
    Returns:
        None
    """
    print("\n" + "="*80)
    print("PATCHED: Launching or updating inference server")
    print("="*80)
    
    # Check if the server is running
    health_info = check_process_health()
    fastapi_info = health_info.get("fastapi_server")
    
    if fastapi_info and fastapi_info.get("status") == "running" and fastapi_info.get("port_active", False):
        # Server is running - just signal model update
        print(f"FastAPI server already running (PID: {fastapi_info.get('pid')})")
        signal_model_update(model_path)
    else:
        # Server is not running or not responding - start a new one
        if fastapi_info and fastapi_info.get("status") == "running":
            print(f"Server process exists (PID: {fastapi_info.get('pid')}) but is not responding")
            print("Starting a new server instance")
        else:
            print("No running FastAPI server found")
            print("Starting a new server instance")
            
        patch_start_inference_server(model_path)
        
    print("="*80)

# Utility functions for run_pipeline.py
def run_pipeline_with_enhanced_process_management():
    """
    Function to be called from run_pipeline.py to start all components
    with enhanced process management.

    Returns:
        tuple: (data_collection_process, inference_ui_process, fastapi_process)
    """
    # Log the startup
    log_event("PIPELINE_STARTUP", {
        "timestamp": time.time(),
        "pid": os.getpid()
    })

    # Track startup progress for detailed error reporting
    startup_info = {
        "start_time": time.time(),
        "steps": [],
        "errors": [],
        "components_started": [],
        "components_failed": [],
        "sequential_start": SEQUENTIAL_START
    }

    try:
        # Define ports
        data_collection_port = 7862
        inference_ui_port = 7861
        fastapi_port = 8001

        startup_info["ports"] = {
            "data_collection": data_collection_port,
            "inference_ui": inference_ui_port,
            "fastapi": fastapi_port
        }

        # Clean up stale processes first
        debug_print("Cleaning up stale processes")
        startup_info["steps"].append("cleanup_stale_processes")

        print("\n" + "="*80)
        print("ENHANCED PROCESS MANAGEMENT: Cleaning up stale processes")
        print("="*80)

        try:
            cleanup_result = cleanup_stale_processes()
            startup_info["cleanup_result"] = cleanup_result
            print(f"Cleaned up {cleanup_result['stale_processes']} stale processes")

            if cleanup_result["errors"]:
                error_count = len(cleanup_result["errors"])
                print(f"Encountered {error_count} errors during cleanup")
                debug_print(f"Cleanup errors: {cleanup_result['errors']}")
                startup_info["errors"].append(f"{error_count} cleanup errors")
        except Exception as cleanup_err:
            error_msg = f"Error during stale process cleanup: {cleanup_err}"
            print(f"WARNING: {error_msg}")
            debug_print(f"{error_msg}\n{traceback.format_exc()}")
            startup_info["errors"].append(error_msg)
            startup_info["cleanup_error"] = str(cleanup_err)
            startup_info["cleanup_traceback"] = traceback.format_exc()

            # Log the error but continue - cleanup is helpful but not critical
            log_event("CLEANUP_ERROR", {
                "error": str(cleanup_err),
                "traceback": traceback.format_exc()
            })

        # Initialize process variables
        fastapi_process = None
        inference_ui_process = None
        data_collection_process = None

        # Check if model directory exists
        model_dir = Path("./output/best_model")
        model_exists = model_dir.exists()
        model_path = str(model_dir) if model_exists else None

        startup_info["model_path"] = model_path
        startup_info["model_exists"] = model_exists

        debug_print(f"Model directory check: exists={model_exists}, path={model_path}")

        print("\n" + "="*80)
        print("STARTING COMPONENTS WITH ENHANCED PROCESS MANAGEMENT")
        if SEQUENTIAL_START:
            print("Using SEQUENTIAL startup mode to prevent module import race conditions")
        else:
            print("Using PARALLEL startup mode")
        print("="*80)

        # Create required environment variables to control sub-processes
        # These are critical for preventing the KeyboardInterrupt during module loading
        os.environ["GRADIO_DISABLE_SIGNAL_HANDLERS"] = "1"
        os.environ["PYTHONUNBUFFERED"] = "1"

        # Start FastAPI server if model exists
        if model_path:
            startup_info["steps"].append("start_fastapi_server")
            debug_print(f"Starting FastAPI server with model: {model_path}")
            print(f"Starting FastAPI server with model: {model_path}")

            try:
                # Set up signal file path
                update_signal_file = Path("./.model_update")
                startup_info["update_signal_file"] = str(update_signal_file)

                # Start the server
                fastapi_process = start_fastapi_inference_server(
                    port=fastapi_port,
                    model_path=model_path,
                    update_signal_file=update_signal_file
                )

                # Check if server started successfully
                if fastapi_process:
                    debug_print(f"FastAPI server started successfully with PID: {fastapi_process.pid}")
                    startup_info["fastapi_pid"] = fastapi_process.pid
                    startup_info["components_started"].append("fastapi_server")

                    # Give it time to start - longer for sequential startup
                    time.sleep(5 if SEQUENTIAL_START else 2)
                else:
                    debug_print("FastAPI server failed to start")
                    startup_info["components_failed"].append("fastapi_server")
                    startup_info["errors"].append("FastAPI server failed to start")
            except Exception as fastapi_err:
                error_msg = f"Error starting FastAPI server: {fastapi_err}"
                print(f"ERROR: {error_msg}")
                debug_print(f"{error_msg}\n{traceback.format_exc()}")
                startup_info["components_failed"].append("fastapi_server")
                startup_info["errors"].append(error_msg)
                startup_info["fastapi_error"] = str(fastapi_err)
                startup_info["fastapi_traceback"] = traceback.format_exc()

                # Log the error but continue with other components
                log_event("FASTAPI_START_ERROR", {
                    "error": str(fastapi_err),
                    "traceback": traceback.format_exc()
                })
        else:
            debug_print("Skipping FastAPI server - no model available")

        # Add delay for sequential startup
        if SEQUENTIAL_START:
            print("Sequential startup: waiting 5 seconds before starting next component...")
            time.sleep(5)

        # Start Gradio Data Collection UI
        startup_info["steps"].append("start_data_collection")
        debug_print(f"Starting Gradio Data Collection UI on port {data_collection_port}")
        print(f"Starting Gradio Data Collection UI on port {data_collection_port}")

        # Set environment variables specifically for Data Collection UI
        os.environ["GRADIO_STARTUP_DELAY"] = "3"  # Give it time to initialize

        try:
            data_collection_process = start_gradio_data_collection(
                port=data_collection_port
            )

            # Check if UI started successfully
            if data_collection_process:
                debug_print(f"Data Collection UI started successfully with PID: {data_collection_process.pid}")
                startup_info["data_collection_pid"] = data_collection_process.pid
                startup_info["components_started"].append("data_collection_ui")

                # Wait longer in sequential mode
                if SEQUENTIAL_START:
                    print("Sequential startup: waiting 10 seconds to ensure Data Collection UI is fully initialized...")
                    time.sleep(10)
            else:
                debug_print("Data Collection UI failed to start")
                startup_info["components_failed"].append("data_collection_ui")
                startup_info["errors"].append("Data Collection UI failed to start")
        except Exception as data_err:
            error_msg = f"Error starting Data Collection UI: {data_err}"
            print(f"ERROR: {error_msg}")
            debug_print(f"{error_msg}\n{traceback.format_exc()}")
            startup_info["components_failed"].append("data_collection_ui")
            startup_info["errors"].append(error_msg)
            startup_info["data_collection_error"] = str(data_err)
            startup_info["data_collection_traceback"] = traceback.format_exc()

            # Log the error but continue with other components
            log_event("DATA_COLLECTION_START_ERROR", {
                "error": str(data_err),
                "traceback": traceback.format_exc()
            })

        # Only proceed to Inference UI after Data Collection UI is fully initialized in sequential mode
        if SEQUENTIAL_START:
            print("Sequential startup: waiting another 5 seconds before starting Inference UI...")
            time.sleep(5)

        # Start Gradio Inference UI with enhanced environment variables
        startup_info["steps"].append("start_inference_ui")
        api_url = f"http://localhost:{fastapi_port}"
        debug_print(f"Starting Gradio Inference UI on port {inference_ui_port} with API URL: {api_url}")
        print(f"Starting Gradio Inference UI on port {inference_ui_port}")

        # Set environment variables specifically for Inference UI
        os.environ["GRADIO_STARTUP_DELAY"] = "5"  # Longer delay for inference UI

        try:
            inference_ui_process = start_gradio_inference_ui(
                port=inference_ui_port,
                api_url=api_url
            )

            # Check if UI started successfully
            if inference_ui_process:
                debug_print(f"Inference UI started successfully with PID: {inference_ui_process.pid}")
                startup_info["inference_ui_pid"] = inference_ui_process.pid
                startup_info["components_started"].append("inference_ui")
            else:
                debug_print("Inference UI failed to start")
                startup_info["components_failed"].append("inference_ui")
                startup_info["errors"].append("Inference UI failed to start")
        except Exception as inference_err:
            error_msg = f"Error starting Inference UI: {inference_err}"
            print(f"ERROR: {error_msg}")
            debug_print(f"{error_msg}\n{traceback.format_exc()}")
            startup_info["components_failed"].append("inference_ui")
            startup_info["errors"].append(error_msg)
            startup_info["inference_ui_error"] = str(inference_err)
            startup_info["inference_ui_traceback"] = traceback.format_exc()

            # Log the error but continue with other components
            log_event("INFERENCE_UI_START_ERROR", {
                "error": str(inference_err),
                "traceback": traceback.format_exc()
            })
        
        # Validate all processes started correctly
        startup_info["steps"].append("process_validation")
        debug_print("Validating processes")
        
        print("\n" + "="*80)
        print("PROCESS STARTUP SUMMARY")
        print("="*80)
        
        # Check if each process is still running
        if fastapi_process:
            is_running = fastapi_process.poll() is None
            debug_print(f"FastAPI server running: {is_running}")
            startup_info["fastapi_still_running"] = is_running
            
            if is_running:
                print(f"✅ FastAPI server started on port {fastapi_port} (PID: {fastapi_process.pid})")
            else:
                print(f"❌ FastAPI server started but terminated unexpectedly (exit code: {fastapi_process.returncode})")
                startup_info["components_failed"].append("fastapi_server_terminated")
                startup_info["errors"].append(f"FastAPI server terminated with code {fastapi_process.returncode}")
        elif model_path:
            print(f"❌ Failed to start FastAPI server")
        else:
            print(f"ℹ️ FastAPI server not started (no model available)")
            
        if data_collection_process:
            is_running = data_collection_process.poll() is None
            debug_print(f"Data Collection UI running: {is_running}")
            startup_info["data_collection_still_running"] = is_running
            
            if is_running:
                print(f"✅ Gradio Data Collection UI started on port {data_collection_port} (PID: {data_collection_process.pid})")
            else:
                print(f"❌ Gradio Data Collection UI started but terminated unexpectedly (exit code: {data_collection_process.returncode})")
                startup_info["components_failed"].append("data_collection_ui_terminated")
                startup_info["errors"].append(f"Data Collection UI terminated with code {data_collection_process.returncode}")
        else:
            print(f"❌ Failed to start Gradio Data Collection UI")
            
        if inference_ui_process:
            is_running = inference_ui_process.poll() is None
            debug_print(f"Inference UI running: {is_running}")
            startup_info["inference_ui_still_running"] = is_running
            
            if is_running:
                print(f"✅ Gradio Inference UI started on port {inference_ui_port} (PID: {inference_ui_process.pid})")
            else:
                print(f"❌ Gradio Inference UI started but terminated unexpectedly (exit code: {inference_ui_process.returncode})")
                startup_info["components_failed"].append("inference_ui_terminated")
                startup_info["errors"].append(f"Inference UI terminated with code {inference_ui_process.returncode}")
        else:
            print(f"❌ Failed to start Gradio Inference UI")
        
        # Display access points
        print("\n" + "="*80)
        print("ACCESS POINTS")
        print("="*80)
        print(f"Data Collection UI: http://localhost:{data_collection_port}")
        print(f"Inference UI: http://localhost:{inference_ui_port}")
        print(f"FastAPI Inference API: http://localhost:{fastapi_port}/docs")
        print("="*80)
        
        # Check port connectivity for all components as final verification
        startup_info["steps"].append("port_verification")
        debug_print("Verifying port connectivity")
        
        try:
            # Verify each port actually works
            fastapi_accessible = is_port_in_use(fastapi_port)
            data_ui_accessible = is_port_in_use(data_collection_port)
            inference_ui_accessible = is_port_in_use(inference_ui_port)
            
            startup_info["port_verification"] = {
                "fastapi": fastapi_accessible,
                "data_collection": data_ui_accessible,
                "inference_ui": inference_ui_accessible
            }
            
            debug_print(f"Port verification: FastAPI={fastapi_accessible}, Data UI={data_ui_accessible}, Inference UI={inference_ui_accessible}")
            
            # Log warnings for any port issues
            if not fastapi_accessible and fastapi_process and fastapi_process.poll() is None:
                warning_msg = f"WARNING: FastAPI server process is running but port {fastapi_port} is not accessible"
                print(warning_msg)
                debug_print(warning_msg)
                startup_info["warnings"] = startup_info.get("warnings", []) + [warning_msg]
                
            if not data_ui_accessible and data_collection_process and data_collection_process.poll() is None:
                warning_msg = f"WARNING: Data Collection UI process is running but port {data_collection_port} is not accessible"
                print(warning_msg)
                debug_print(warning_msg)
                startup_info["warnings"] = startup_info.get("warnings", []) + [warning_msg]
                
            if not inference_ui_accessible and inference_ui_process and inference_ui_process.poll() is None:
                warning_msg = f"WARNING: Inference UI process is running but port {inference_ui_port} is not accessible"
                print(warning_msg)
                debug_print(warning_msg)
                startup_info["warnings"] = startup_info.get("warnings", []) + [warning_msg]
        except Exception as verify_err:
            debug_print(f"Error during port verification: {verify_err}")
            startup_info["port_verification_error"] = str(verify_err)
        
        # Log successful startup
        startup_info["completion_time"] = time.time()
        startup_info["duration"] = startup_info["completion_time"] - startup_info["start_time"]
        
        # Check for any errors
        if startup_info["errors"]:
            log_event("PIPELINE_STARTUP_WITH_ERRORS", startup_info)
            print(f"\nWARNING: Pipeline started with {len(startup_info['errors'])} errors")
        else:
            log_event("PIPELINE_STARTUP_SUCCESS", startup_info)
            print(f"\nINFO: Pipeline components started successfully")
        
        return data_collection_process, inference_ui_process, fastapi_process
        
    except Exception as e:
        # Handle any unexpected errors in the startup process
        error_msg = f"Critical error during pipeline startup: {e}"
        print(f"CRITICAL ERROR: {error_msg}")
        debug_print(f"{error_msg}\n{traceback.format_exc()}")
        
        # Log the critical error
        startup_info["critical_error"] = str(e)
        startup_info["critical_traceback"] = traceback.format_exc()
        startup_info["completion_time"] = time.time()
        startup_info["duration"] = startup_info["completion_time"] - startup_info["start_time"]
        
        log_event("PIPELINE_STARTUP_CRITICAL_ERROR", startup_info)
        
        # Return whatever processes we managed to start, or None values
        data_collection_process = data_collection_process if 'data_collection_process' in locals() else None
        inference_ui_process = inference_ui_process if 'inference_ui_process' in locals() else None
        fastapi_process = fastapi_process if 'fastapi_process' in locals() else None
        
        return data_collection_process, inference_ui_process, fastapi_process