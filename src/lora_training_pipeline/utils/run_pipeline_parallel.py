#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/run_pipeline_parallel.py
"""
Parallel Service Integration Module for run_pipeline.py

This module provides integration between the parallel_startup.py module and
the main run_pipeline.py script, enabling services to be started in parallel
rather than sequentially.
"""

import os
import sys
import time
import signal
import logging
import traceback
import subprocess
from pathlib import Path
import psutil
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./parallel_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('run_pipeline_parallel')

# Set debug level for more verbose output
debug_level = os.environ.get('LORA_DEBUG_LEVEL', 'INFO').upper()
if debug_level == 'DEBUG':
    logger.setLevel(logging.DEBUG)
    print(f"🐛 DEBUG logging enabled for run_pipeline_parallel")
elif debug_level == 'TRACE':
    logger.setLevel(logging.DEBUG)  # Python doesn't have TRACE, use DEBUG
    print(f"🔍 TRACE logging enabled for run_pipeline_parallel")
else:
    logger.setLevel(logging.INFO)

# Import the parallel startup module
try:
    from src.lora_training_pipeline.utils.parallel_startup import (
        ServiceManager, DEFAULT_SERVICES
    )
    PARALLEL_STARTUP_AVAILABLE = True
    logger.info("✅ Parallel startup module loaded successfully")
except ImportError as e:
    PARALLEL_STARTUP_AVAILABLE = False
    logger.error(f"❌ Parallel startup module not available: {e}")
    logger.error("Install with: uv pip install -e .")
    
    # Get detailed import error information
    try:
        import importlib.util
        spec = importlib.util.find_spec("src.lora_training_pipeline.utils.parallel_startup")
        if spec is None:
            logger.error("Module not found in Python path")
            
            # Check if file exists
            parallel_file = Path("src/lora_training_pipeline/utils/parallel_startup.py")
            if parallel_file.exists():
                logger.error(f"File exists at {parallel_file.absolute()} but cannot be imported")
            else:
                logger.error(f"File does not exist at expected path: {parallel_file.absolute()}")
                
            # Check PYTHONPATH
            logger.error(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
            logger.error(f"sys.path: {sys.path}")
    except Exception as check_err:
        logger.error(f"Error checking module details: {check_err}")
        logger.debug(traceback.format_exc())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./parallel_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('parallel_pipeline')

def run_pipeline_with_parallel_services(
    data_collection_port: int = 7862,
    inference_ui_port: int = 7861,
    fastapi_inference_port: int = 8001,
    dashboard_port: int = 7863,
    session_id: str = None
) -> Tuple[List[int], ServiceManager]:
    """
    Start all pipeline services in parallel using the ServiceManager.
    
    Args:
        data_collection_port: Port for the data collection Gradio UI
        inference_ui_port: Port for the inference Gradio UI
        fastapi_inference_port: Port for the FastAPI inference server
        dashboard_port: Port for the dashboard UI
        session_id: Current session ID for logging and tracking
        
    Returns:
        Tuple containing:
        - List of PIDs for the started processes
        - ServiceManager instance for controlling the services
    """
    start_time = time.time()
    logger.info("\n" + "="*80)
    logger.info("STARTING PIPELINE SERVICES IN PARALLEL")
    logger.info("="*80)
    
    # Log input parameters
    logger.info(f"Parallel startup configuration:")
    logger.info(f"  Data Collection Port: {data_collection_port}")
    logger.info(f"  Inference UI Port: {inference_ui_port}")
    logger.info(f"  FastAPI Inference Port: {fastapi_inference_port}")
    logger.info(f"  Dashboard Port: {dashboard_port}")
    logger.info(f"  Session ID: {session_id or 'Not provided'}")
    logger.info(f"  Working Directory: {os.getcwd()}")
    
    if not PARALLEL_STARTUP_AVAILABLE:
        logger.error("❌ ERROR: Parallel startup module not available.")
        logger.error("Please install the package with: uv pip install -e .")
        return [], None
    
    try:
        # Build custom service configuration with the provided ports
        services = {
            "dashboard": {
                "command": ["python", "-m", "src.lora_training_pipeline.utils.dashboard"],
                "port": dashboard_port,
                "env": {
                    "PROCESS_NAME": "PipelineDashboard",
                    "DASHBOARD_PORT": str(dashboard_port),
                    "DASHBOARD_API_URL": f"http://localhost:{dashboard_port}",
                    "PYTHONPATH": "${PWD}:${PYTHONPATH}",
                    "SESSION_ID": session_id or "",
                    "PYTHONUNBUFFERED": "1",
                    "LORA_DEBUG_LEVEL": debug_level
                },
                "health_check": {
                    "type": "port",
                    "timeout": 15,
                    "interval": 1,
                    "retries": 15
                },
                "dependencies": []  # No dependencies
            },
            "fastapi_inference": {
                "command": ["python", "-m", "src.lora_training_pipeline.inference.fastapi_inference"],
                "port": fastapi_inference_port,
                "env": {
                    "FASTAPI_INFERENCE_PORT": str(fastapi_inference_port),
                    "PYTHONPATH": "${PWD}:${PYTHONPATH}",
                    "PROCESS_NAME": "FastAPIInferenceServer",
                    "SESSION_ID": session_id or "",
                    "PYTHONUNBUFFERED": "1",
                    "LORA_DEBUG_LEVEL": debug_level
                },
                "health_check": {
                    "url": f"http://localhost:{fastapi_inference_port}/health",
                    "timeout": 20,
                    "interval": 1,
                    "retries": 20
                },
                "dependencies": []  # No dependencies
            },
            "data_collection": {
                "command": ["python", "-m", "src.lora_training_pipeline.data_collection.gradio_app"],
                "port": data_collection_port,
                "env": {
                    "GRADIO_PORT": str(data_collection_port),
                    "GRADIO_DATA_COLLECTION_PORT": str(data_collection_port),
                    "DATA_COLLECTION_API_URL": f"http://localhost:{data_collection_port}",
                    "PYTHONPATH": "${PWD}:${PYTHONPATH}",
                    "PROCESS_NAME": "GradioDataCollection",
                    "SESSION_ID": session_id or "",
                    "GRADIO_DISABLE_SIGNAL_HANDLERS": "1",
                    "PYTHONUNBUFFERED": "1",
                    "LORA_DEBUG_LEVEL": debug_level
                },
                "health_check": {
                    "type": "port",
                    "timeout": 15,
                    "interval": 1,
                    "retries": 15
                },
                "dependencies": []  # No dependencies
            },
            "inference_ui": {
                "command": ["python", "-m", "src.lora_training_pipeline.inference.gradio_inference"],
                "port": inference_ui_port,
                "env": {
                    "GRADIO_PORT": str(inference_ui_port),
                    "INFERENCE_API_URL": f"http://localhost:{fastapi_inference_port}",
                    "PYTHONPATH": "${PWD}:${PYTHONPATH}",
                    "PROCESS_NAME": "GradioInferenceUI",
                    "SESSION_ID": session_id or "",
                    "GRADIO_DISABLE_SIGNAL_HANDLERS": "1",
                    "PYTHONUNBUFFERED": "1",
                    "LORA_DEBUG_LEVEL": debug_level
                },
                "health_check": {
                    "type": "port",
                    "timeout": 15,
                    "interval": 1,
                    "retries": 15
                },
                "dependencies": ["fastapi_inference"]  # Depends on the FastAPI server
            }
        }
        
        # Check Python executable to ensure it's correct
        python_executable = sys.executable
        logger.info(f"Using Python executable: {python_executable}")
        
        # Update commands to use the correct Python executable
        for service_name, config in services.items():
            if config["command"][0] == "python":
                config["command"][0] = python_executable
                logger.debug(f"Updated command for {service_name} to use Python: {python_executable}")
        
        # Initialize the service manager with our configuration
        try:
            logger.info("Initializing ServiceManager...")
            manager = ServiceManager(services)
            logger.info("✅ ServiceManager initialized successfully")
        except Exception as init_err:
            logger.error(f"❌ Error initializing ServiceManager: {init_err}")
            logger.error(traceback.format_exc())
            return [], None
        
        # Start all services in parallel
        logger.info("Starting all services in parallel...")
        start_services_time = time.time()
        try:
            results = manager.start_all_services()
            services_duration = time.time() - start_services_time
            logger.info(f"Service startup completed in {services_duration:.2f} seconds")
        except Exception as start_err:
            logger.error(f"❌ Error starting services: {start_err}")
            logger.error(traceback.format_exc())
            return [], manager  # Return manager for cleanup even if startup failed
        
        # Print results
        successful = sum(1 for result in results.values() if result)
        total = len(services)
        logger.info(f"Started {successful}/{total} services successfully")
        
        # If no services started successfully, return early
        if successful == 0:
            logger.error("❌ No services started successfully")
            return [], manager
        
        # Print detailed status
        try:
            statuses = manager.get_all_statuses()
            logger.info("\nService Status:")
            logger.info("==============")
            
            pids = []
            failed_services = []
            
            for service_name, status in statuses.items():
                running = "✅ Running" if status["running"] else "❌ Stopped"
                pid = status["pid"] or "N/A"
                port = status["port"] or "N/A"
                exit_code = f" (exit code: {status['exit_code']})" if status['exit_code'] is not None else ""
                
                logger.info(f"\n{service_name}:")
                logger.info(f"  Status: {running}{exit_code}")
                logger.info(f"  PID: {pid}")
                logger.info(f"  Port: {port}")
                logger.info(f"  URL: {status['url'] or 'N/A'}")
                
                if status['uptime']:
                    hours, remainder = divmod(int(status['uptime']), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    logger.info(f"  Uptime: {hours}h {minutes}m {seconds}s")
                
                # Add PID to the list if the service is running
                if status["running"] and status["pid"]:
                    pids.append(status["pid"])
                else:
                    failed_services.append(service_name)
            
            # If some services failed to start, log more details
            if failed_services:
                logger.warning(f"⚠️ The following services failed to start: {', '.join(failed_services)}")
                
                # Check for common issues
                for service_name in failed_services:
                    # Look for log files
                    log_glob = f"{service_name}_*_stderr.log"
                    try:
                        import glob
                        log_files = glob.glob(log_glob)
                        if log_files:
                            newest_log = max(log_files, key=os.path.getmtime)
                            logger.warning(f"Checking error log for {service_name}: {newest_log}")
                            try:
                                with open(newest_log, 'r') as f:
                                    # Read last 20 lines
                                    lines = f.readlines()
                                    if len(lines) > 20:
                                        logger.warning(f"Last 20 lines of error log:\n{''.join(lines[-20:])}")
                                    else:
                                        logger.warning(f"Error log content:\n{''.join(lines)}")
                            except Exception as read_err:
                                logger.error(f"Error reading log file: {read_err}")
                    except Exception as glob_err:
                        logger.error(f"Error looking for log files: {glob_err}")
        except Exception as status_err:
            logger.error(f"❌ Error getting service statuses: {status_err}")
            logger.error(traceback.format_exc())
        
        # Update PID files for each service for compatibility with the original script
        try:
            pid_files = {
                "fastapi_inference": Path('./inference_process.pid'),
                "data_collection": Path('./data_collection_ui.pid'),
                "inference_ui": Path('./inference_ui.pid')
            }
            
            for service_name, pid_file in pid_files.items():
                if service_name in statuses and statuses[service_name]["running"]:
                    pid = statuses[service_name]["pid"]
                    if pid:
                        logger.info(f"Writing PID {pid} to file: {pid_file}")
                        
                        # Use atomic write operation
                        temp_file = Path(f"{pid_file}.tmp")
                        try:
                            temp_file.write_text(str(pid))
                            temp_file.replace(pid_file)  # Atomic operation
                            logger.debug(f"PID file written successfully: {pid_file}")
                        except Exception as write_err:
                            logger.error(f"Error writing PID file: {write_err}")
                            # Fallback to direct write
                            try:
                                pid_file.write_text(str(pid))
                                logger.debug(f"PID file written successfully using fallback method: {pid_file}")
                            except Exception as fallback_err:
                                logger.error(f"Fallback write also failed: {fallback_err}")
        except Exception as pid_err:
            logger.error(f"⚠️ Warning: Could not update PID files: {pid_err}")
            logger.error(traceback.format_exc())
        
        # Set environment variables for URLs
        try:
            os.environ["DATA_COLLECTION_API_URL"] = f"http://localhost:{data_collection_port}"
            os.environ["INFERENCE_API_URL"] = f"http://localhost:{fastapi_inference_port}"
            os.environ["DASHBOARD_API_URL"] = f"http://localhost:{dashboard_port}"
            logger.info("✅ Environment variables set for service URLs")
        except Exception as env_err:
            logger.error(f"⚠️ Warning: Could not set environment variables: {env_err}")
            logger.error(traceback.format_exc())
        
        # Calculate total duration
        total_duration = time.time() - start_time
        logger.info(f"\n✅ Parallel service startup complete with {len(pids)} processes in {total_duration:.2f} seconds")
        
        return pids, manager
    except Exception as e:
        logger.error(f"❌ Unexpected error in parallel service startup: {e}")
        logger.error(traceback.format_exc())
        return [], None

def stop_parallel_services(manager: ServiceManager = None) -> Dict[str, bool]:
    """
    Stop all services that were started in parallel.
    
    Args:
        manager: ServiceManager instance to use for stopping services
        
    Returns:
        Dictionary with status of each service stop operation
    """
    start_time = time.time()
    logger.info("\n" + "="*80)
    logger.info("STOPPING PARALLEL PIPELINE SERVICES")
    logger.info("="*80)
    
    if not manager:
        logger.warning("No service manager provided, cannot stop services")
        return {}
    
    try:
        # Get status before stopping for diagnostic purposes
        try:
            statuses = manager.get_all_statuses()
            running = sum(1 for status in statuses.values() if status.get("running", False))
            total = len(statuses)
            logger.info(f"Found {running}/{total} services running before shutdown")
            
            # Log details of running services
            for name, status in statuses.items():
                if status.get("running", False):
                    pid = status.get("pid", "N/A")
                    port = status.get("port", "N/A")
                    uptime = status.get("uptime", 0)
                    if uptime:
                        hours, remainder = divmod(int(uptime), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        uptime_str = f"{hours}h {minutes}m {seconds}s"
                    else:
                        uptime_str = "N/A"
                    logger.debug(f"Service {name}: PID={pid}, Port={port}, Uptime={uptime_str}")
        except Exception as status_err:
            logger.error(f"Error getting service statuses before shutdown: {status_err}")
            logger.debug(traceback.format_exc())
        
        # Stop all services
        logger.info("Stopping all services...")
        stop_start_time = time.time()
        results = manager.stop_all_services()
        stop_duration = time.time() - stop_start_time
        
        # Print results
        successful = sum(1 for result in results.values() if result)
        total = len(results)
        logger.info(f"Stopped {successful}/{total} services successfully in {stop_duration:.2f} seconds")
        
        # Handle any failed stops
        if successful < total:
            failed = [name for name, result in results.items() if not result]
            logger.warning(f"⚠️ Failed to stop the following services: {', '.join(failed)}")
            
            # Try force killing if some services failed to stop
            logger.info("Attempting to force kill remaining processes...")
            force_kill_count = 0
            
            try:
                # Get updated status
                statuses = manager.get_all_statuses()
                
                for name in failed:
                    if name in statuses and statuses[name].get("running", False):
                        pid = statuses[name].get("pid")
                        if pid:
                            logger.warning(f"Force killing service {name} with PID {pid}")
                            try:
                                if sys.platform == 'win32':
                                    # Windows - use taskkill /F
                                    subprocess.run(['taskkill', '/F', '/PID', str(pid)], 
                                                 check=False, capture_output=True, text=True)
                                else:
                                    # Unix - use kill -9
                                    os.kill(pid, signal.SIGKILL)
                                force_kill_count += 1
                            except Exception as kill_err:
                                logger.error(f"Error force killing process {pid}: {kill_err}")
            except Exception as force_err:
                logger.error(f"Error during force kill: {force_err}")
                logger.debug(traceback.format_exc())
                
            if force_kill_count > 0:
                logger.info(f"Force killed {force_kill_count} processes")
        
        # Remove PID files
        try:
            pid_files = [
                Path('./inference_process.pid'),
                Path('./data_collection_ui.pid'),
                Path('./inference_ui.pid')
            ]
            
            for pid_file in pid_files:
                if pid_file.exists():
                    try:
                        pid_file.unlink()
                        logger.debug(f"Removed PID file: {pid_file}")
                    except Exception as unlink_err:
                        logger.warning(f"Could not remove PID file {pid_file}: {unlink_err}")
        except Exception as pid_file_err:
            logger.error(f"Error cleaning up PID files: {pid_file_err}")
        
        # Calculate total duration
        total_duration = time.time() - start_time
        logger.info(f"\n✅ Parallel service shutdown complete in {total_duration:.2f} seconds")
        
        return results
    except Exception as e:
        logger.error(f"❌ Unexpected error stopping parallel services: {e}")
        logger.error(traceback.format_exc())
        return {}

if __name__ == "__main__":
    # Test functionality by starting services directly
    logger.info("Testing parallel pipeline startup by direct invocation...")
    
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
    
    # Start services with diagnostic timers
    start_time = time.time()
    pids, manager = run_pipeline_with_parallel_services()
    duration = time.time() - start_time
    
    if not manager:
        logger.error("❌ Failed to get service manager")
        sys.exit(1)
    
    if not pids:
        logger.error("❌ No services started successfully")
        logger.error("Check the logs for errors and try again")
        sys.exit(1)
    
    logger.info(f"✅ Started {len(pids)} services in {duration:.2f} seconds")
    logger.info("\nServices are running. Press Ctrl+C to stop...")
    
    try:
        # Keep checking service health periodically
        check_interval = 10  # seconds
        while True:
            try:
                time.sleep(check_interval)
                
                # Check if services are still running
                if manager:
                    try:
                        statuses = manager.get_all_statuses()
                        running = sum(1 for status in statuses.values() if status.get("running", False))
                        logger.info(f"Health check: {running}/{len(statuses)} services running")
                        
                        # Log any non-running services
                        for name, status in statuses.items():
                            if not status.get("running", False):
                                exit_code = status.get("exit_code")
                                logger.warning(f"⚠️ Service {name} is not running (exit code: {exit_code})")
                    except Exception as health_err:
                        logger.error(f"Error checking service health: {health_err}")
                        logger.debug(traceback.format_exc())
            except KeyboardInterrupt:
                # Handle Ctrl+C in the nested loop
                raise
            except Exception as loop_err:
                logger.error(f"Error in main loop: {loop_err}")
                logger.debug(traceback.format_exc())
    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt received, stopping services...")
        start_time = time.time()
        stop_parallel_services(manager)
        duration = time.time() - start_time
        logger.info(f"All services stopped in {duration:.2f} seconds.")
    except Exception as e:
        logger.error(f"\nUnexpected error in main loop: {e}")
        logger.error(traceback.format_exc())
        stop_parallel_services(manager)
        logger.info("All services stopped after error.")
        sys.exit(1)