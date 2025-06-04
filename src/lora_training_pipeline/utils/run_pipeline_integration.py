#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/run_pipeline_integration.py
"""
Integration Module for Parallel Pipeline Startup

This module provides the necessary hooks to integrate the parallel service startup
capability into the main run_pipeline.py script. It's designed to be imported from
run_pipeline.py to replace the sequential service startup with parallel execution.
"""

import os
import sys
import time
import traceback
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./pipeline_integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('pipeline_integration')

# Set debug level for more verbose output
debug_level = os.environ.get('LORA_DEBUG_LEVEL', 'INFO').upper()
if debug_level == 'DEBUG':
    logger.setLevel(logging.DEBUG)
    print(f"🐛 DEBUG logging enabled for pipeline_integration")
elif debug_level == 'TRACE':
    logger.setLevel(logging.DEBUG)  # Python doesn't have TRACE, use DEBUG
    print(f"🔍 TRACE logging enabled for pipeline_integration")
else:
    logger.setLevel(logging.INFO)

# Import core utilities
from src.lora_training_pipeline.utils.process_watchdog import ProcessWatchdog

# Import the parallel startup module
try:
    from src.lora_training_pipeline.utils.run_pipeline_parallel import (
        run_pipeline_with_parallel_services,
        stop_parallel_services,
        PARALLEL_STARTUP_AVAILABLE
    )
    logger.info("✅ Parallel startup module loaded successfully")
except ImportError as e:
    PARALLEL_STARTUP_AVAILABLE = False
    logger.error(f"❌ Parallel startup module not available: {e}")
    # Get detailed import error information
    try:
        import importlib.util
        spec = importlib.util.find_spec("src.lora_training_pipeline.utils.run_pipeline_parallel")
        if spec is None:
            logger.error("Module not found in Python path")
            
            # Check if file exists
            parallel_file = Path("src/lora_training_pipeline/utils/run_pipeline_parallel.py")
            if parallel_file.exists():
                logger.error(f"File exists at {parallel_file.absolute()} but cannot be imported")
            else:
                logger.error(f"File does not exist at expected path: {parallel_file.absolute()}")
                
            # Check PYTHONPATH
            logger.error(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
            logger.error(f"sys.path: {sys.path}")
    except Exception as check_err:
        logger.error(f"Error checking module details: {check_err}")
        
    logger.error("\nTo fix this issue, make sure the package is installed in development mode:")
    logger.error("uv venv .venv && . .venv/bin/activate && uv pip install -e .")

# Global variable to store the service manager
service_manager = None

def run_pipeline_with_parallel_execution(
    data_collection_port: int = 7862,
    inference_ui_port: int = 7861,
    fastapi_inference_port: int = 8001,
    dashboard_port: int = 7863,
    session_id: str = None
) -> Tuple[Any, Any, Any]:
    """
    Start all pipeline services in parallel and return process references compatible
    with the original run_pipeline.py script.
    
    Args:
        data_collection_port: Port for the data collection Gradio UI
        inference_ui_port: Port for the inference Gradio UI
        fastapi_inference_port: Port for the FastAPI inference server
        dashboard_port: Port for the dashboard UI
        session_id: Current session ID for logging and tracking
        
    Returns:
        Tuple containing (data_collection_process, inference_ui_process, fastapi_process)
        to maintain compatibility with the original script
    """
    global service_manager
    start_time = time.time()
    
    if not PARALLEL_STARTUP_AVAILABLE:
        logger.error("❌ ERROR: Parallel startup module not available")
        logger.error("Falling back to sequential startup. Please install with:")
        logger.error("uv venv .venv && . .venv/bin/activate && uv pip install -e .")
        return None, None, None
    
    try:
        logger.info("\n" + "="*80)
        logger.info("STARTING PIPELINE WITH PARALLEL EXECUTION")
        logger.info("="*80)
        
        # Log all parameters for debugging
        logger.info(f"Input parameters:")
        logger.info(f"  Data Collection Port: {data_collection_port}")
        logger.info(f"  Inference UI Port: {inference_ui_port}")
        logger.info(f"  FastAPI Inference Port: {fastapi_inference_port}")
        logger.info(f"  Dashboard Port: {dashboard_port}")
        logger.info(f"  Session ID: {session_id}")
        
        # First check if there are any port conflicts
        logger.info("Checking for port conflicts...")
        
        # Wrap each watchdog creation in try-except to prevent failures
        watchdogs = {}
        
        try:
            watchdogs["Data Collection UI"] = ProcessWatchdog("GradioDataCollection", data_collection_port)
            logger.debug(f"Created watchdog for Data Collection UI on port {data_collection_port}")
        except Exception as e:
            logger.error(f"❌ Error creating watchdog for Data Collection UI: {e}")
            logger.debug(traceback.format_exc())
            
        try:
            watchdogs["Inference UI"] = ProcessWatchdog("GradioInferenceUI", inference_ui_port)
            logger.debug(f"Created watchdog for Inference UI on port {inference_ui_port}")
        except Exception as e:
            logger.error(f"❌ Error creating watchdog for Inference UI: {e}")
            logger.debug(traceback.format_exc())
            
        try:
            watchdogs["FastAPI Inference"] = ProcessWatchdog("FastAPIInferenceServer", fastapi_inference_port)
            logger.debug(f"Created watchdog for FastAPI Inference on port {fastapi_inference_port}")
        except Exception as e:
            logger.error(f"❌ Error creating watchdog for FastAPI Inference: {e}")
            logger.debug(traceback.format_exc())
            
        try:
            watchdogs["Dashboard"] = ProcessWatchdog("PipelineDashboard", dashboard_port)
            logger.debug(f"Created watchdog for Dashboard on port {dashboard_port}")
        except Exception as e:
            logger.error(f"❌ Error creating watchdog for Dashboard: {e}")
            logger.debug(traceback.format_exc())
        
        # Check all ports together
        port_checks = {}
        for name, watchdog in watchdogs.items():
            try:
                can_start = watchdog.can_start()
                port_checks[name] = can_start
                logger.debug(f"Port check for {name}: {'✅ Available' if can_start else '❌ Conflict'}")
            except Exception as e:
                logger.error(f"❌ Error checking port for {name}: {e}")
                logger.debug(traceback.format_exc())
                port_checks[name] = False
        
        # If any port check fails, report it
        failed_ports = {}
        for name, success in port_checks.items():
            if not success and name in watchdogs:
                try:
                    failed_ports[name] = watchdogs[name].error_message
                except Exception:
                    failed_ports[name] = "Unknown error (could not get error message)"
        
        if failed_ports:
            logger.error("❌ Cannot start some services due to port conflicts:")
            for name, error in failed_ports.items():
                logger.error(f"  - {name}: {error}")
            logger.warning("\nPipeline will start with limited functionality.")
            
            # Check if we can run port diagnostics
            for name, error in failed_ports.items():
                port = None
                if name == "Data Collection UI":
                    port = data_collection_port
                elif name == "Inference UI":
                    port = inference_ui_port
                elif name == "FastAPI Inference":
                    port = fastapi_inference_port
                elif name == "Dashboard":
                    port = dashboard_port
                    
                if port:
                    try:
                        import socket
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        result = sock.connect_ex(('127.0.0.1', port))
                        if result == 0:
                            logger.error(f"Diagnostic: Port {port} is currently in use")
                            
                            # Try to get the process using the port on Unix-like systems
                            if sys.platform != 'win32':
                                try:
                                    # Use lsof to find the process using the port
                                    lsof_output = subprocess.check_output(
                                        ["lsof", "-i", f":{port}"], 
                                        stderr=subprocess.STDOUT,
                                        text=True
                                    )
                                    logger.error(f"Process using port {port}:\n{lsof_output}")
                                except Exception as lsof_err:
                                    logger.debug(f"Could not get process using port: {lsof_err}")
                        else:
                            logger.error(f"Diagnostic: Port {port} appears to be free (socket check returned {result})")
                            logger.error(f"This is inconsistent with the watchdog result and might indicate a bug")
                        sock.close()
                    except Exception as sock_err:
                        logger.error(f"Error checking port {port}: {sock_err}")
            
            # You might want to abort here or continue with available ports
            logger.warning("Continuing with available ports only.")
        
        # Start services in parallel
        logger.info("Starting services in parallel...")
        pids, manager = run_pipeline_with_parallel_services(
            data_collection_port=data_collection_port,
            inference_ui_port=inference_ui_port,
            fastapi_inference_port=fastapi_inference_port,
            dashboard_port=dashboard_port,
            session_id=session_id
        )
        
        if not manager:
            logger.error("❌ Failed to get service manager from parallel startup")
            return None, None, None
            
        if not pids:
            logger.error("❌ No PIDs returned from parallel startup")
            logger.error("Services may have failed to start properly")
            # Continue anyway as some services might still be running
        else:
            logger.info(f"✅ Started {len(pids)} services with PIDs: {pids}")
        
        # Store manager for later use
        service_manager = manager
        
        # Create dummy process objects for compatibility with the original script
        # These won't be real subprocess.Popen objects, but they'll have the necessary attributes
        class DummyProcess:
            def __init__(self, pid=None, service_name=None):
                self.pid = pid
                self.service_name = service_name
                self._returncode = None
                self.creation_time = time.time()
                logger.debug(f"Created DummyProcess for {service_name} with PID {pid}")
                
            def poll(self):
                # Check if the process is still running
                if self.pid is None:
                    logger.debug(f"DummyProcess.poll(): {self.service_name} has no PID, returning 1")
                    return 1  # Process not started
                
                try:
                    import psutil
                    if psutil.pid_exists(self.pid):
                        logger.debug(f"DummyProcess.poll(): {self.service_name} (PID {self.pid}) is running, returning None")
                        return None  # Process is running
                    else:
                        self._returncode = 1
                        logger.warning(f"⚠️ DummyProcess.poll(): {self.service_name} (PID {self.pid}) is not running, returning 1")
                        return 1  # Process has exited
                except Exception as e:
                    logger.warning(f"⚠️ DummyProcess.poll(): Error checking {self.service_name} (PID {self.pid}): {e}")
                    # If we can't check, assume it's still running
                    return None
                    
            def terminate(self):
                # This is a no-op, as termination is handled by the service manager
                logger.info(f"Request to terminate process {self.service_name} (PID: {self.pid})")
                logger.info("Termination will be handled by the service manager during shutdown")
                
                # Actually use the service manager to stop the service
                if service_manager and self.service_name in ["data_collection", "inference_ui", "fastapi_inference"]:
                    service_map = {
                        "data_collection": "data_collection",
                        "inference_ui": "inference_ui",
                        "fastapi_inference": "fastapi_inference"
                    }
                    actual_name = service_map.get(self.service_name)
                    if actual_name:
                        try:
                            logger.info(f"Forwarding terminate request to service manager for {actual_name}")
                            service_manager.stop_service(actual_name)
                        except Exception as e:
                            logger.error(f"Error stopping service {actual_name}: {e}")
                
                return
                
            def kill(self):
                # This is a no-op, as termination is handled by the service manager
                logger.info(f"Request to kill process {self.service_name} (PID: {self.pid})")
                logger.info("Termination will be handled by the service manager during shutdown")
                
                # Use the terminate method which will use the service manager
                self.terminate()
                return
        
        # Get the status of each service
        try:
            statuses = manager.get_all_statuses()
            logger.debug(f"Service statuses: {statuses}")
            
            # Create dummy processes for compatibility
            data_collection_process = DummyProcess(
                pid=statuses.get("data_collection", {}).get("pid"),
                service_name="data_collection"
            )
            
            inference_ui_process = DummyProcess(
                pid=statuses.get("inference_ui", {}).get("pid"),
                service_name="inference_ui"
            )
            
            fastapi_process = DummyProcess(
                pid=statuses.get("fastapi_inference", {}).get("pid"),
                service_name="fastapi_inference"
            )
            
            # Log timing information
            duration = time.time() - start_time
            logger.info(f"✅ Parallel execution startup completed in {duration:.2f} seconds")
            logger.info(f"Data Collection UI: PID={data_collection_process.pid}")
            logger.info(f"Inference UI: PID={inference_ui_process.pid}")
            logger.info(f"FastAPI Inference: PID={fastapi_process.pid}")
            
            return data_collection_process, inference_ui_process, fastapi_process
        except Exception as status_err:
            logger.error(f"❌ Error getting service statuses: {status_err}")
            logger.error(traceback.format_exc())
            return None, None, None
    
    except Exception as e:
        logger.error(f"❌ Failed to start services with parallel execution: {e}")
        logger.error(traceback.format_exc())
        return None, None, None

def stop_all_parallel_services():
    """Stop all services that were started with parallel execution."""
    global service_manager
    
    if service_manager:
        logger.info("Stopping all services with parallel execution...")
        try:
            # Get service statuses before stopping for diagnostic purposes
            try:
                statuses = service_manager.get_all_statuses()
                running_services = []
                for name, status in statuses.items():
                    if status.get("running", False):
                        running_services.append(f"{name} (PID: {status.get('pid')})")
                
                if running_services:
                    logger.info(f"Found {len(running_services)} running services to stop:")
                    for service in running_services:
                        logger.info(f"  - {service}")
                else:
                    logger.warning("No running services found, but service manager exists")
            except Exception as status_err:
                logger.error(f"Error getting service statuses: {status_err}")
                logger.debug(traceback.format_exc())
            
            # Actually stop the services
            start_time = time.time()
            results = stop_parallel_services(service_manager)
            duration = time.time() - start_time
            
            # Log results
            successful = sum(1 for result in results.values() if result)
            total = len(results)
            logger.info(f"✅ Stopped {successful}/{total} services in {duration:.2f} seconds")
            
            # Reset the service manager
            service_manager = None
            
            return results
        except Exception as e:
            logger.error(f"❌ Error stopping parallel services: {e}")
            logger.error(traceback.format_exc())
            service_manager = None
            return {}
    else:
        logger.info("No service manager found, no services to stop")
        return {}

def register_cleanup_handlers():
    """Register cleanup handlers for proper shutdown."""
    import atexit
    import signal
    import threading
    
    # Set a global flag for cleaner shutdown detection
    global _shutdown_in_progress
    _shutdown_in_progress = False
    
    # Create a lock to prevent re-entrant cleanup calls
    cleanup_lock = threading.Lock()
    
    # Register atexit handler
    def atexit_handler():
        """Handle atexit cleanup with locking."""
        global _shutdown_in_progress
        
        # Use non-blocking acquire to prevent deadlocks
        if cleanup_lock.acquire(blocking=False):
            try:
                if not _shutdown_in_progress:
                    _shutdown_in_progress = True
                    logger.info("atexit handler triggered, performing cleanup")
                    stop_all_parallel_services()
                else:
                    logger.debug("atexit handler: shutdown already in progress, skipping duplicate cleanup")
            finally:
                cleanup_lock.release()
        else:
            logger.warning("atexit handler: cleanup lock is held, skipping duplicate cleanup")
    
    # Register the handler
    atexit.register(atexit_handler)
    logger.debug("Registered atexit handler for parallel services cleanup")
    
    # Save original signal handlers
    try:
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        logger.debug(f"Saved original signal handlers: SIGINT={original_sigint}, SIGTERM={original_sigterm}")
    except Exception as e:
        logger.error(f"Error saving original signal handlers: {e}")
        original_sigint = signal.SIG_DFL  # Default handler
        original_sigterm = signal.SIG_DFL  # Default handler
    
    # Create signal handler
    def signal_handler(sig, frame):
        """Handle signals by stopping all services and calling the original handler."""
        global _shutdown_in_progress
        signal_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else f"signal {sig}"
        
        # Use non-blocking acquire to prevent deadlocks
        if cleanup_lock.acquire(blocking=False):
            try:
                if not _shutdown_in_progress:
                    _shutdown_in_progress = True
                    logger.info(f"Received {signal_name}, stopping all services")
                    stop_all_parallel_services()
                    logger.info(f"Cleanup complete, restoring original handlers")
                else:
                    logger.debug(f"Signal handler: shutdown already in progress, skipping duplicate cleanup")
            finally:
                cleanup_lock.release()
        else:
            logger.warning(f"Signal handler: cleanup lock is held, skipping duplicate cleanup")
        
        # Restore original handlers (always do this to prevent infinite recursion)
        try:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            logger.debug(f"Restored original signal handlers")
        except Exception as restore_err:
            logger.error(f"Error restoring signal handlers: {restore_err}")
        
        # Re-raise the signal to let the original handler deal with it
        logger.debug(f"Re-raising {signal_name} to original handler")
        try:
            os.kill(os.getpid(), sig)
        except Exception as kill_err:
            logger.error(f"Error re-raising signal {sig}: {kill_err}")
            # If we can't re-raise, exit directly as a fallback
            logger.error("Forcing exit as fallback")
            os._exit(128 + sig)
    
    # Register our handlers with proper error checking
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("✅ Registered cleanup handlers for parallel execution")
    except Exception as sig_err:
        logger.error(f"❌ Error registering signal handlers: {sig_err}")
        logger.error(traceback.format_exc())
        logger.warning("Process cleanup may be incomplete on shutdown")

if __name__ == "__main__":
    # Test functionality
    logger.info("Testing parallel pipeline startup integration...")
    
    # Set debug level for more detailed logging during testing
    os.environ['LORA_DEBUG_LEVEL'] = 'DEBUG'
    logger.setLevel(logging.DEBUG)
    logger.info("🐛 DEBUG logging enabled for test")
    
    # Log system information for diagnostics
    logger.info("System information:")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # Register cleanup handlers
    register_cleanup_handlers()
    
    # Start services with diagnostic timers
    logger.info("Starting services with parallel execution...")
    start_time = time.time()
    data_collection, inference_ui, fastapi = run_pipeline_with_parallel_execution()
    duration = time.time() - start_time
    
    # Log the results
    if all([data_collection, inference_ui, fastapi]):
        logger.info(f"✅ All services started successfully in {duration:.2f} seconds")
        logger.info(f"Data Collection UI: PID={data_collection.pid}")
        logger.info(f"Inference UI: PID={inference_ui.pid}")
        logger.info(f"FastAPI Inference: PID={fastapi.pid}")
    else:
        logger.error(f"❌ Some services failed to start properly")
        logger.error(f"Data Collection UI: {'✅' if data_collection else '❌'}")
        logger.error(f"Inference UI: {'✅' if inference_ui else '❌'}")
        logger.error(f"FastAPI Inference: {'✅' if fastapi else '❌'}")
    
    logger.info("\nServices are running. Press Ctrl+C to stop...")
    
    try:
        # Keep checking service health periodically
        check_interval = 10  # seconds
        while True:
            try:
                time.sleep(check_interval)
                
                # Check if services are still running
                if service_manager:
                    try:
                        statuses = service_manager.get_all_statuses()
                        running = sum(1 for status in statuses.values() if status.get("running", False))
                        logger.info(f"Health check: {running}/{len(statuses)} services running")
                        
                        # Log any non-running services
                        for name, status in statuses.items():
                            if not status.get("running", False):
                                exit_code = status.get("exit_code")
                                logger.warning(f"⚠️ Service {name} is not running (exit code: {exit_code})")
                    except Exception as health_err:
                        logger.error(f"Error checking service health: {health_err}")
            except KeyboardInterrupt:
                # Handle Ctrl+C in the nested loop
                raise
            except Exception as loop_err:
                logger.error(f"Error in main loop: {loop_err}")
                logger.debug(traceback.format_exc())
    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt received, stopping services...")
        start_time = time.time()
        stop_all_parallel_services()
        duration = time.time() - start_time
        logger.info(f"All services stopped in {duration:.2f} seconds.")
    except Exception as e:
        logger.error(f"\nUnexpected error in main loop: {e}")
        logger.error(traceback.format_exc())
        stop_all_parallel_services()
        logger.info("All services stopped after error.")
        sys.exit(1)