#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/parallel_startup.py
"""
Parallel Service Startup

This module provides utilities to start services in parallel, avoiding the
sequential startup that can lead to race conditions and stalled processes.
It manages proper initialization, resource allocation, and health checks
for all services.
"""

import os
import sys
import time
import signal
import logging
import threading
from pathlib import Path
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
from typing import Dict, List, Tuple, Any, Optional, Callable, Set

# Import process management utilities
try:
    from src.lora_training_pipeline.utils.process_watchdog import (
        ProcessWatchdog, get_services_status, cleanup_all_services
    )
except ImportError:
    # If ProcessWatchdog is not available, use simplified implementations
    print("WARNING: ProcessWatchdog not available, using simplified process management")
    
    def get_services_status():
        """Get status of all services."""
        return {}
        
    def cleanup_all_services():
        """Clean up all services."""
        return {"cleaned_services": 0, "errors": []}
        
    class ProcessWatchdog:
        """Dummy ProcessWatchdog."""
        def __init__(self, service_type, port=None):
            self.service_type = service_type
            self.port = port
            self.error_message = None
            
        def can_start(self):
            """Check if service can start."""
            return True
            
        def register_pid(self, pid, metadata=None):
            """Register service PID."""
            return True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./parallel_startup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('parallel_startup')

# Set debug level for more verbose output
debug_level = os.environ.get('LORA_DEBUG_LEVEL', 'INFO').upper()
if debug_level == 'DEBUG':
    logger.setLevel(logging.DEBUG)
    print(f"🐛 DEBUG logging enabled for parallel_startup")
elif debug_level == 'TRACE':
    logger.setLevel(logging.DEBUG)  # Python doesn't have TRACE, use DEBUG
    print(f"🔍 TRACE logging enabled for parallel_startup")
else:
    logger.setLevel(logging.INFO)

# Default service configuration
DEFAULT_SERVICES = {
    "fastapi_inference": {
        "command": ["python", "-m", "src.lora_training_pipeline.inference.fastapi_inference"],
        "port": 8001,
        "env": {
            "FASTAPI_INFERENCE_PORT": "8001",
            "PYTHONPATH": "${PWD}:${PYTHONPATH}",
            "PROCESS_NAME": "FastAPIInferenceServer"
        },
        "health_check": {
            "url": "http://localhost:8001/health",
            "timeout": 20,  # seconds
            "interval": 1,  # seconds
            "retries": 20
        },
        "dependencies": []  # No dependencies
    },
    "data_collection": {
        "command": ["python", "-m", "src.lora_training_pipeline.data_collection.gradio_app"],
        "port": 7862,
        "env": {
            "GRADIO_PORT": "7862",
            "PYTHONPATH": "${PWD}:${PYTHONPATH}",
            "PROCESS_NAME": "GradioDataCollection",
            "GRADIO_DISABLE_SIGNAL_HANDLERS": "1"
        },
        "health_check": {
            "type": "port",  # Simple port check instead of HTTP
            "timeout": 15,  # seconds
            "interval": 1,  # seconds
            "retries": 15
        },
        "dependencies": []  # No dependencies
    },
    "inference_ui": {
        "command": ["python", "-m", "src.lora_training_pipeline.inference.gradio_inference"],
        "port": 7861,
        "env": {
            "GRADIO_PORT": "7861",
            "INFERENCE_API_URL": "http://localhost:8001",
            "PYTHONPATH": "${PWD}:${PYTHONPATH}",
            "PROCESS_NAME": "GradioInferenceUI",
            "GRADIO_DISABLE_SIGNAL_HANDLERS": "1"
        },
        "health_check": {
            "type": "port",  # Simple port check instead of HTTP
            "timeout": 15,  # seconds
            "interval": 1,  # seconds
            "retries": 15
        },
        "dependencies": ["fastapi_inference"]  # Depends on the FastAPI server
    }
}

class ServiceManager:
    """
    Manages the startup and shutdown of multiple services in parallel.
    
    This class handles:
    - Starting services in the correct order based on dependencies
    - Checking service health
    - Handling service shutdown
    - Resource cleanup
    """
    def __init__(self, services: Optional[Dict[str, Dict[str, Any]]] = None, max_workers: int = 4):
        """
        Initialize service manager.
        
        Args:
            services: Dictionary of service configurations (uses DEFAULT_SERVICES if None)
            max_workers: Maximum number of concurrent workers for parallel operations
        """
        self.services = services or DEFAULT_SERVICES
        self.max_workers = max_workers
        self.processes = {}
        self.watchdogs = {}
        self.stop_event = threading.Event()
        self.service_locks = {name: threading.Lock() for name in self.services}
        
        # Validate service configurations
        self._validate_services()
        
        # Register signal handlers for cleanup
        self._register_signal_handlers()
        
        # Register atexit handler
        import atexit
        atexit.register(self.stop_all_services)
        
    def _validate_services(self):
        """Validate service configurations and resolve dependencies."""
        # Check for circular dependencies
        visited = set()
        temp_visited = set()
        
        def check_circular_deps(service_name, path=None):
            if path is None:
                path = []
                
            if service_name in temp_visited:
                raise ValueError(f"Circular dependency detected: {' -> '.join(path + [service_name])}")
                
            if service_name in visited:
                return
                
            temp_visited.add(service_name)
            path.append(service_name)
            
            # Check each dependency
            for dep in self.services[service_name].get("dependencies", []):
                if dep not in self.services:
                    raise ValueError(f"Service {service_name} depends on unknown service {dep}")
                    
                check_circular_deps(dep, path.copy())
                
            temp_visited.remove(service_name)
            visited.add(service_name)
            
        # Check each service
        for service_name in self.services:
            check_circular_deps(service_name)
            
        logger.info("Service configurations validated successfully")
        
    def _register_signal_handlers(self):
        """Register signal handlers for clean shutdown."""
        # Save original handlers
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)
        
        def signal_handler(sig, frame):
            """Handle signals by stopping all services and exiting."""
            logger.info(f"Received signal {signal.Signals(sig).name}, stopping all services")
            self.stop_all_services()
            
            # Restore original handlers
            signal.signal(signal.SIGINT, self.original_sigint)
            signal.signal(signal.SIGTERM, self.original_sigterm)
            
            # Re-raise the signal to let the original handler deal with it
            os.kill(os.getpid(), sig)
            
        # Register our handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def prepare_environment(self, service_name: str) -> Dict[str, str]:
        """
        Prepare environment variables for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            dict: Environment variables for the service
        """
        try:
            # Start with current environment
            env = os.environ.copy()
            
            # Add service-specific variables
            service_env = self.services[service_name].get("env", {})
            logger.debug(f"Preparing environment for {service_name} with {len(service_env)} custom variables")
            
            for key, value in service_env.items():
                # Handle variable expansion like ${PWD}
                orig_value = value
                if "${" in value:
                    # Simple expansion (not a full shell implementation)
                    for var_name, var_value in env.items():
                        if f"${{{var_name}}}" in value:
                            value = value.replace(f"${{{var_name}}}", var_value)
                            logger.debug(f"  Expanded ${{{var_name}}} in {key} to '{var_value}'")
                    
                    # Check if all variables were expanded
                    if "${" in value:
                        unexpanded = [x for x in value.split() if "${" in x]
                        logger.warning(f"⚠️ Warning: Could not expand all variables in {key}={orig_value}")
                        logger.warning(f"  Unexpanded variables: {unexpanded}")
                        
                env[key] = value
            
            # Add debug environment variable if enabled
            if logger.level <= logging.DEBUG:
                env["LORA_DEBUG_LEVEL"] = "DEBUG"
            
            # Ensure Python output is unbuffered for better logging
            env["PYTHONUNBUFFERED"] = "1"
            
            # Add a unique service identifier for easier process tracking
            env["SERVICE_ID"] = f"{service_name}_{os.getpid()}_{int(time.time())}"
            
            logger.debug(f"Environment prepared for {service_name} with {len(env)} total variables")
            return env
            
        except Exception as e:
            logger.error(f"❌ Error preparing environment for {service_name}: {e}")
            logger.error(traceback.format_exc())
            # Return basic environment to avoid complete failure
            basic_env = os.environ.copy()
            basic_env["PYTHONUNBUFFERED"] = "1"
            basic_env["SERVICE_STARTUP_ERROR"] = f"Error preparing environment: {str(e)}"
            return basic_env
        
    def start_service(self, service_name: str) -> bool:
        """
        Start a service.
        
        Args:
            service_name: Name of the service to start
            
        Returns:
            bool: True if service started successfully, False otherwise
        """
        start_time = time.time()
        logger.info(f"⏳ Starting service {service_name}...")
        
        try:
            with self.service_locks[service_name]:
                # Check if service is already running
                if service_name in self.processes and self.processes[service_name].poll() is None:
                    logger.info(f"✅ Service {service_name} is already running")
                    return True
                    
                # Get service configuration
                service_config = self.services.get(service_name)
                if not service_config:
                    logger.error(f"❌ No configuration for service {service_name}")
                    return False
                    
                # Log service details
                logger.debug(f"Service {service_name} configuration:")
                logger.debug(f"  Command: {service_config.get('command')}")
                logger.debug(f"  Port: {service_config.get('port')}")
                logger.debug(f"  Dependencies: {service_config.get('dependencies', [])}")
                
                # Check if dependencies are running
                dependencies = service_config.get("dependencies", [])
                if dependencies:
                    logger.info(f"Checking {len(dependencies)} dependencies for {service_name}")
                    
                for dep in dependencies:
                    if dep not in self.processes:
                        logger.error(f"❌ Dependency {dep} for service {service_name} is missing from process list")
                        return False
                    
                    if self.processes[dep].poll() is not None:
                        exit_code = self.processes[dep].poll()
                        logger.error(f"❌ Dependency {dep} for service {service_name} exited with code {exit_code}")
                        return False
                    
                    logger.debug(f"✓ Dependency {dep} for {service_name} is running (PID: {self.processes[dep].pid})")
                        
                # Check if we can start the service (port available, etc.)
                port = service_config.get("port")
                logger.info(f"Creating ProcessWatchdog for {service_name} on port {port}")
                watchdog = ProcessWatchdog(service_name, port)
                
                # Try to start the service
                if not watchdog.can_start():
                    logger.error(f"❌ Cannot start service {service_name}: {watchdog.error_message}")
                    
                    # Try to get more detailed information about port conflicts
                    if port:
                        try:
                            import socket
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            result = sock.connect_ex(('127.0.0.1', port))
                            if result == 0:
                                logger.error(f"⚠️ Port {port} is already in use by another process")
                                
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
                                logger.error(f"⚠️ Port {port} check returned {result}, but watchdog reports it's unavailable")
                            sock.close()
                        except Exception as sock_err:
                            logger.error(f"Error checking port {port}: {sock_err}")
                    
                    return False
                    
                # Store the watchdog for later use
                self.watchdogs[service_name] = watchdog
                    
                # Prepare command and environment
                command = service_config["command"]
                logger.info(f"Preparing environment for {service_name}")
                env = self.prepare_environment(service_name)
                
                # Generate unique log file names with timestamps
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                stdout_log = f"{service_name}_{timestamp}_stdout.log"
                stderr_log = f"{service_name}_{timestamp}_stderr.log"
                
                # Start the process
                try:
                    logger.info(f"🚀 Starting service {service_name} with command: {' '.join(command)}")
                    logger.info(f"📝 Logs: stdout={stdout_log}, stderr={stderr_log}")
                    
                    # Use platform-specific arguments for subprocess.Popen
                    # Create file handles with debug logging
                    try:
                        stdout_handle = open(stdout_log, "a")
                    except Exception as stdout_err:
                        print(f"[DEBUG] Parallel startup stdout open error type: {type(stdout_err).__name__}")
                        print(f"[DEBUG] Parallel startup stdout open error details: {stdout_err}")
                        print(f"[DEBUG] Parallel startup stdout path: {stdout_log}")
                        raise
                    
                    try:
                        stderr_handle = open(stderr_log, "a")
                    except Exception as stderr_err:
                        print(f"[DEBUG] Parallel startup stderr open error type: {type(stderr_err).__name__}")
                        print(f"[DEBUG] Parallel startup stderr open error details: {stderr_err}")
                        print(f"[DEBUG] Parallel startup stderr path: {stderr_log}")
                        stdout_handle.close()  # Clean up stdout handle if stderr fails
                        raise
                    
                    proc_kwargs = {
                        'env': env,
                        # Redirect output to files
                        'stdout': stdout_handle,
                        'stderr': stderr_handle,
                        # Set start_new_session flag for Unix-like systems
                        'start_new_session': sys.platform != 'win32'
                    }
                    
                    # Add Windows-specific flags
                    if sys.platform == 'win32':
                        # On Windows, use CREATE_NEW_PROCESS_GROUP flag
                        proc_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
                        logger.debug(f"Using Windows-specific creationflags for {service_name}")
                    
                    # Start the process
                    try:
                        process = subprocess.Popen(command, **proc_kwargs)
                    except Exception as popen_err:
                        print(f"[DEBUG] Parallel startup subprocess.Popen error type: {type(popen_err).__name__}")
                        print(f"[DEBUG] Parallel startup subprocess.Popen error details: {popen_err}")
                        print(f"[DEBUG] Parallel startup subprocess.Popen command: {command}")
                        print(f"[DEBUG] Parallel startup subprocess.Popen kwargs keys: {list(proc_kwargs.keys())}")
                        # Clean up file handles
                        try:
                            stdout_handle.close()
                            stderr_handle.close()
                        except Exception as close_err:
                            print(f"[DEBUG] Error closing file handles: {close_err}")
                        raise
                    
                    # Verify process started
                    if process.pid <= 0:
                        logger.error(f"❌ Process for {service_name} has invalid PID: {process.pid}")
                        return False
                    
                    # Check if process is immediately dead (poll() returns non-None for a dead process)
                    if process.poll() is not None:
                        exit_code = process.poll()
                        logger.error(f"❌ Process for {service_name} exited immediately with code {exit_code}")
                        
                        # Try to read stderr for clues
                        try:
                            with open(stderr_log, "r") as err_file:
                                err_content = err_file.read(2000)  # Read first 2000 chars
                                if err_content:
                                    logger.error(f"Error output from {service_name}:\n{err_content}")
                        except Exception as read_err:
                            print(f"[DEBUG] Parallel startup stderr read error type: {type(read_err).__name__}")
                            print(f"[DEBUG] Parallel startup stderr read error details: {read_err}")
                            print(f"[DEBUG] Parallel startup stderr path: {stderr_log}")
                            logger.error(f"Could not read error log: {read_err}")
                            
                        return False
                    
                    # Store process info
                    self.processes[service_name] = process
                    
                    # Register PID with watchdog
                    process_metadata = {
                        "command": " ".join(command),
                        "start_time": time.strftime('%Y-%m-%dT%H:%M:%S'),
                        "stdout_log": stdout_log,
                        "stderr_log": stderr_log,
                        "service_name": service_name,
                        "port": port
                    }
                    
                    if not watchdog.register_pid(process.pid, process_metadata):
                        logger.warning(f"⚠️ Failed to register PID with watchdog for {service_name}")
                    
                    # Create a symlink to the latest log files for easier access
                    try:
                        if os.path.exists(f"{service_name}_latest_stdout.log"):
                            os.unlink(f"{service_name}_latest_stdout.log")
                        if os.path.exists(f"{service_name}_latest_stderr.log"):
                            os.unlink(f"{service_name}_latest_stderr.log")
                            
                        if sys.platform != 'win32':
                            os.symlink(stdout_log, f"{service_name}_latest_stdout.log")
                            os.symlink(stderr_log, f"{service_name}_latest_stderr.log")
                        else:
                            # On Windows, just copy the files
                            import shutil
                            shutil.copy2(stdout_log, f"{service_name}_latest_stdout.log")
                            shutil.copy2(stderr_log, f"{service_name}_latest_stderr.log")
                    except Exception as link_err:
                        logger.warning(f"⚠️ Could not create log symlinks: {link_err}")
                    
                    duration = time.time() - start_time
                    logger.info(f"✅ Started service {service_name} with PID {process.pid} in {duration:.2f}s")
                    
                    # Start health check in a separate thread
                    health_thread = threading.Thread(
                        target=self._health_check_thread,
                        args=(service_name,),
                        daemon=True,
                        name=f"health_check_{service_name}"
                    )
                    health_thread.start()
                    logger.debug(f"Started health check thread for {service_name}")
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ Error starting service {service_name}: {e}")
                    logger.error(traceback.format_exc())
                    return False
        except Exception as outer_e:
            logger.error(f"❌ Unexpected error in start_service for {service_name}: {outer_e}")
            logger.error(traceback.format_exc())
            return False
                
    def start_all_services(self) -> Dict[str, bool]:
        """
        Start all services in parallel, respecting dependencies.
        
        Returns:
            dict: Status of each service (True if started successfully, False otherwise)
        """
        # First determine dependency levels
        dependency_levels = self._get_dependency_levels()
        
        # Start services level by level
        results = {}
        
        for level, services in enumerate(dependency_levels):
            logger.info(f"Starting services at dependency level {level}: {', '.join(services)}")
            
            # Start services in this level in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.start_service, service_name): service_name
                    for service_name in services
                }
                
                for future in as_completed(futures):
                    service_name = futures[future]
                    try:
                        result = future.result()
                        results[service_name] = result
                        logger.info(f"Service {service_name} {'started successfully' if result else 'failed to start'}")
                    except Exception as e:
                        results[service_name] = False
                        logger.error(f"Error starting service {service_name}: {e}")
            
            # Check if any service in this level failed to start
            if not all(results.get(service, False) for service in services):
                logger.error(f"Some services at level {level} failed to start, cannot continue to next level")
                break
        
        # Log summary
        successful = sum(1 for result in results.values() if result)
        total = len(self.services)
        logger.info(f"Started {successful}/{total} services successfully")
        
        return results
        
    def stop_service(self, service_name: str) -> bool:
        """
        Stop a service.
        
        Args:
            service_name: Name of the service to stop
            
        Returns:
            bool: True if service stopped successfully, False otherwise
        """
        with self.service_locks[service_name]:
            # Check if service is running
            if service_name not in self.processes:
                logger.info(f"Service {service_name} is not running")
                return True
                
            process = self.processes[service_name]
            
            if process.poll() is not None:
                logger.info(f"Service {service_name} is already stopped")
                return True
                
            # Stop the process
            try:
                logger.info(f"Stopping service {service_name} (PID {process.pid})")
                
                # Try to terminate gracefully
                process.terminate()
                
                # Wait for termination
                for _ in range(10):  # 10 second timeout
                    if process.poll() is not None:
                        logger.info(f"Service {service_name} terminated gracefully")
                        return True
                    time.sleep(1)
                    
                # Force kill if still running
                logger.warning(f"Service {service_name} did not terminate gracefully, forcing kill")
                process.kill()
                
                # Wait again
                for _ in range(5):  # 5 second timeout
                    if process.poll() is not None:
                        logger.info(f"Service {service_name} forcefully killed")
                        return True
                    time.sleep(1)
                    
                logger.error(f"Failed to kill service {service_name}")
                return False
                
            except Exception as e:
                logger.error(f"Error stopping service {service_name}: {e}")
                return False
                
    def stop_all_services(self) -> Dict[str, bool]:
        """
        Stop all services in parallel, in reverse dependency order.
        
        Returns:
            dict: Status of each service (True if stopped successfully, False otherwise)
        """
        # Set stop event
        self.stop_event.set()
        
        # Determine dependency levels
        dependency_levels = self._get_dependency_levels()
        
        # Stop in reverse order
        results = {}
        
        for level in reversed(range(len(dependency_levels))):
            services = dependency_levels[level]
            logger.info(f"Stopping services at dependency level {level}: {', '.join(services)}")
            
            # Stop services in this level in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.stop_service, service_name): service_name
                    for service_name in services
                }
                
                for future in as_completed(futures):
                    service_name = futures[future]
                    try:
                        result = future.result()
                        results[service_name] = result
                    except Exception as e:
                        results[service_name] = False
                        logger.error(f"Error stopping service {service_name}: {e}")
        
        # Clean up resources
        self._cleanup_resources()
        
        # Log summary
        successful = sum(1 for result in results.values() if result)
        total = len(self.processes)
        logger.info(f"Stopped {successful}/{total} services successfully")
        
        return results
        
    def restart_service(self, service_name: str) -> bool:
        """
        Restart a service.
        
        Args:
            service_name: Name of the service to restart
            
        Returns:
            bool: True if service restarted successfully, False otherwise
        """
        # Stop the service
        if not self.stop_service(service_name):
            logger.error(f"Failed to stop service {service_name} for restart")
            return False
            
        # Start the service
        return self.start_service(service_name)
        
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """
        Get status of a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            dict: Status information for the service
        """
        status = {
            "name": service_name,
            "running": False,
            "pid": None,
            "uptime": None,
            "exit_code": None,
            "port": self.services[service_name].get("port"),
            "url": None
        }
        
        # Check if process exists
        if service_name in self.processes:
            process = self.processes[service_name]
            
            # Check if process is running
            exit_code = process.poll()
            
            if exit_code is None:
                # Process is running
                status["running"] = True
                status["pid"] = process.pid
                
                # Get process info
                try:
                    proc = psutil.Process(process.pid)
                    create_time = proc.create_time()
                    uptime = time.time() - create_time
                    status["uptime"] = uptime
                    status["create_time"] = time.strftime(
                        '%Y-%m-%d %H:%M:%S', 
                        time.localtime(create_time)
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            else:
                # Process has exited
                status["exit_code"] = exit_code
        
        # Add URL if available
        port = status["port"]
        if port:
            status["url"] = f"http://localhost:{port}"
            
        return status
        
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all services.
        
        Returns:
            dict: Status information for all services
        """
        return {
            service_name: self.get_service_status(service_name)
            for service_name in self.services
        }
        
    def _health_check_thread(self, service_name: str):
        """
        Thread function for checking service health.
        
        Args:
            service_name: Name of the service to check
        """
        start_time = time.time()
        logger.info(f"⏳ Starting health check for service {service_name}")
        
        try:
            service_config = self.services[service_name]
            health_config = service_config.get("health_check", {})
            
            # Get health check parameters
            check_type = health_config.get("type", "http")
            timeout = health_config.get("timeout", 30)
            interval = health_config.get("interval", 1)
            retries = health_config.get("retries", 30)
            url = health_config.get("url", f"http://localhost:{service_config.get('port', 8000)}/health")
            
            logger.info(f"Health check configuration for {service_name}:")
            logger.info(f"  Type: {check_type}")
            logger.info(f"  Timeout: {timeout}s")
            logger.info(f"  Interval: {interval}s")
            logger.info(f"  Retries: {retries}")
            if check_type == "http":
                logger.info(f"  URL: {url}")
            elif check_type == "port":
                logger.info(f"  Port: {service_config.get('port')}")
            
            # Import requests if needed
            if check_type == "http":
                try:
                    import requests
                    from requests.exceptions import RequestException
                    logger.debug(f"Successfully imported requests for HTTP health check")
                except ImportError:
                    logger.error(f"❌ Cannot import requests module for HTTP health check")
                    logger.error(f"Please install requests: pip install requests")
                    return
            
            # Wait for service to start
            healthy = False
            status_messages = []
            
            for attempt in range(retries):
                # Check if stop event is set
                if self.stop_event.is_set():
                    logger.info(f"Health check for {service_name} cancelled due to stop event")
                    return
                    
                # Check if process is still running
                if service_name in self.processes:
                    process = self.processes[service_name]
                    exit_code = process.poll()
                    if exit_code is not None:
                        # Try to read stderr for clues
                        error_details = ""
                        try:
                            if service_name in self.watchdogs:
                                watchdog = self.watchdogs[service_name]
                                pid_data = watchdog.get_pid_data()
                                if pid_data and "stderr_log" in pid_data:
                                    stderr_log = pid_data["stderr_log"]
                                    with open(stderr_log, "r") as err_file:
                                        error_details = err_file.read(2000)  # Read first 2000 chars
                        except Exception as read_err:
                            print(f"[DEBUG] Parallel startup health check stderr read error type: {type(read_err).__name__}")
                            print(f"[DEBUG] Parallel startup health check stderr read error details: {read_err}")
                            logger.debug(f"Could not read error log: {read_err}")
                            
                        logger.error(f"❌ Service {service_name} exited with code {exit_code} during health check")
                        if error_details:
                            logger.error(f"Error output from {service_name}:\n{error_details}")
                        return
                else:
                    logger.error(f"❌ Service {service_name} not found in processes during health check")
                    return
                    
                # Perform health check based on type
                try:
                    if check_type == "http":
                        # HTTP health check
                        logger.debug(f"Attempt {attempt+1}/{retries}: HTTP health check for {service_name} at {url}")
                        response = requests.get(url, timeout=timeout)
                        
                        if response.status_code < 300:
                            # Successful health check
                            duration = time.time() - start_time
                            logger.info(f"✅ Service {service_name} is healthy (HTTP {response.status_code}) after {duration:.2f}s")
                            
                            # Try to get more detailed health info if available
                            try:
                                health_data = response.json()
                                logger.debug(f"Health data from {service_name}: {health_data}")
                            except Exception:
                                # JSON parsing failed, use text response
                                try:
                                    if len(response.text) > 0:
                                        logger.debug(f"Health response from {service_name}: {response.text[:200]}...")
                                except Exception:
                                    pass
                            
                            healthy = True
                            break
                        else:
                            # Non-successful HTTP status
                            message = f"Health check returned HTTP {response.status_code}"
                            if message not in status_messages:
                                status_messages.append(message)
                                logger.warning(f"⚠️ {message}")
                            
                            # Try to get response content for more details
                            try:
                                content = response.text[:200]
                                logger.debug(f"Response from {service_name}: {content}")
                            except Exception:
                                pass
                    elif check_type == "port":
                        # Simple port check
                        port = service_config.get("port")
                        if port:
                            logger.debug(f"Attempt {attempt+1}/{retries}: Port health check for {service_name} on port {port}")
                            import socket
                            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                                s.settimeout(timeout)
                                result = s.connect_ex(('localhost', port))
                                if result == 0:
                                    duration = time.time() - start_time
                                    logger.info(f"✅ Service {service_name} is healthy (port {port} is open) after {duration:.2f}s")
                                    healthy = True
                                    break
                                else:
                                    message = f"Port {port} is not open (code: {result})"
                                    if message not in status_messages:
                                        status_messages.append(message)
                                        logger.warning(f"⚠️ {message}")
                        else:
                            logger.error(f"❌ No port specified for service {service_name} port health check")
                            return
                    else:
                        logger.error(f"❌ Unknown health check type: {check_type}")
                        return
                except Exception as e:
                    message = f"Health check error: {type(e).__name__}: {str(e)}"
                    if message not in status_messages:
                        status_messages.append(message)
                        logger.warning(f"⚠️ {message}")
                    
                    if logger.level <= logging.DEBUG:
                        logger.debug(traceback.format_exc())
                
                # Show progress every 5 attempts
                if (attempt + 1) % 5 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"⏳ Still waiting for {service_name} to be healthy (attempt {attempt+1}/{retries}, elapsed: {elapsed:.1f}s)")
                
                # Wait before next attempt
                time.sleep(interval)
                
            # Check if health check succeeded
            if healthy:
                return
            else:
                # Health check failed after all retries
                logger.error(f"❌ Health check for service {service_name} failed after {retries} retries ({time.time() - start_time:.1f}s)")
                logger.error(f"Status messages: {', '.join(status_messages) if status_messages else 'No additional information'}")
                
                # Check if process is still running and get additional diagnostics
                if service_name in self.processes:
                    process = self.processes[service_name]
                    if process.poll() is None:
                        logger.error(f"Process is still running (PID: {process.pid}) but health check failed")
                        
                        # Try to get more process info
                        try:
                            import psutil
                            proc = psutil.Process(process.pid)
                            logger.error(f"Process details: Status={proc.status()}, CPU={proc.cpu_percent()}%, Memory={proc.memory_info().rss / (1024*1024):.1f}MB")
                            
                            # Try to read stderr log
                            if service_name in self.watchdogs:
                                watchdog = self.watchdogs[service_name]
                                pid_data = watchdog.get_pid_data()
                                if pid_data and "stderr_log" in pid_data:
                                    stderr_log = pid_data["stderr_log"]
                                    try:
                                        with open(stderr_log, "r") as err_file:
                                            last_lines = err_file.readlines()[-20:]  # Last 20 lines
                                            logger.error(f"Last lines from stderr:\n{''.join(last_lines)}")
                                    except Exception as read_err:
                                        print(f"[DEBUG] Parallel startup diagnostics stderr read error type: {type(read_err).__name__}")
                                        print(f"[DEBUG] Parallel startup diagnostics stderr read error details: {read_err}")
                                        logger.error(f"Could not read stderr log: {read_err}")
                        except Exception as diag_err:
                            logger.error(f"Error getting diagnostics: {diag_err}")
        except Exception as e:
            logger.error(f"❌ Unexpected error in health check thread for {service_name}: {e}")
            logger.error(traceback.format_exc())
        
    def _get_dependency_levels(self) -> List[List[str]]:
        """
        Group services by dependency level for parallel execution.
        
        Returns:
            list: List of lists, where each inner list contains services at the same dependency level
        """
        # First, get all dependency relationships
        dependencies = {
            service_name: set(self.services[service_name].get("dependencies", []))
            for service_name in self.services
        }
        
        # Group services by level
        levels = []
        remaining = set(self.services.keys())
        
        while remaining:
            # Find services with no remaining dependencies
            current_level = {
                service for service in remaining
                if not dependencies[service] & remaining  # No dependencies in remaining services
            }
            
            if not current_level:
                # Circular dependency detected
                logger.error(f"Circular dependency detected among remaining services: {remaining}")
                # Break the cycle by taking the first remaining service
                current_level = {next(iter(remaining))}
                
            # Add current level to levels
            levels.append(sorted(current_level))
            
            # Remove current level from remaining
            remaining -= current_level
            
        return levels
        
    def _cleanup_resources(self):
        """Clean up resources used by services."""
        # Close log file handles
        for service_name in self.processes:
            process = self.processes[service_name]
            
            # Close stdout/stderr files if they exist
            for attr in ['stdout', 'stderr']:
                if hasattr(process, attr) and getattr(process, attr) is not None:
                    try:
                        getattr(process, attr).close()
                    except Exception as e:
                        logger.error(f"Error closing {attr} for {service_name}: {e}")
        
        # Clear process dictionary
        self.processes.clear()
        
        # Clear watchdogs
        self.watchdogs.clear()
        
def start_services_parallel(services=None):
    """
    Start all services in parallel.
    
    Args:
        services: Dictionary of service configurations (uses DEFAULT_SERVICES if None)
        
    Returns:
        ServiceManager: The service manager instance
    """
    manager = ServiceManager(services)
    manager.start_all_services()
    return manager
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel Service Startup Utility")
    parser.add_argument("--start", action="store_true", help="Start all services in parallel")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    parser.add_argument("--restart", action="store_true", help="Restart all services")
    parser.add_argument("--status", action="store_true", help="Show status of all services")
    parser.add_argument("--service", help="Specify a single service to act on")
    
    args = parser.parse_args()
    
    # Create service manager
    manager = ServiceManager()
    
    if args.start:
        if args.service:
            # Start a single service
            success = manager.start_service(args.service)
            print(f"Service {args.service} {'started successfully' if success else 'failed to start'}")
        else:
            # Start all services
            results = manager.start_all_services()
            successful = sum(1 for result in results.values() if result)
            total = len(results)
            print(f"Started {successful}/{total} services successfully")
            
    elif args.stop:
        if args.service:
            # Stop a single service
            success = manager.stop_service(args.service)
            print(f"Service {args.service} {'stopped successfully' if success else 'failed to stop'}")
        else:
            # Stop all services
            results = manager.stop_all_services()
            successful = sum(1 for result in results.values() if result)
            total = len(results)
            print(f"Stopped {successful}/{total} services successfully")
            
    elif args.restart:
        if args.service:
            # Restart a single service
            success = manager.restart_service(args.service)
            print(f"Service {args.service} {'restarted successfully' if success else 'failed to restart'}")
        else:
            # Restart all services
            manager.stop_all_services()
            results = manager.start_all_services()
            successful = sum(1 for result in results.values() if result)
            total = len(results)
            print(f"Restarted {successful}/{total} services successfully")
            
    elif args.status or not any([args.start, args.stop, args.restart, args.service]):
        # Show status of all services
        if args.service:
            # Show status of a single service
            status = manager.get_service_status(args.service)
            print(f"\nStatus of {args.service}:")
            print(f"  Running: {'Yes' if status['running'] else 'No'}")
            print(f"  PID: {status['pid'] or 'N/A'}")
            print(f"  Port: {status['port'] or 'N/A'}")
            print(f"  URL: {status['url'] or 'N/A'}")
            if status['uptime']:
                hours, remainder = divmod(int(status['uptime']), 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"  Uptime: {hours}h {minutes}m {seconds}s")
            if status['exit_code'] is not None:
                print(f"  Exit Code: {status['exit_code']}")
        else:
            # Show status of all services
            statuses = manager.get_all_statuses()
            
            print("\nService Status:")
            print("==============")
            
            for service_name, status in statuses.items():
                running = "✅ Running" if status["running"] else "❌ Stopped"
                pid = status["pid"] or "N/A"
                port = status["port"] or "N/A"
                exit_code = f" (exit code: {status['exit_code']})" if status['exit_code'] is not None else ""
                
                print(f"\n{service_name}:")
                print(f"  Status: {running}{exit_code}")
                print(f"  PID: {pid}")
                print(f"  Port: {port}")
                print(f"  URL: {status['url'] or 'N/A'}")
                
                if status['uptime']:
                    hours, remainder = divmod(int(status['uptime']), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(f"  Uptime: {hours}h {minutes}m {seconds}s")
                    print(f"  Started: {status['create_time']}")