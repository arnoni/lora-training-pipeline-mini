#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/service_manager.py

"""
Service Manager Module

Provides a unified solution for managing services, ensuring only one instance
runs per port, with proper cross-platform locking, PID file management,
and process health monitoring.

This module resolves the following issues:
1. Multiple FastAPI servers starting on the same port
2. Duplicate Gradio UIs running simultaneously
3. Inconsistent PID file formats causing parsing errors
4. Cross-platform process locking without requiring fcntl
5. Port binding reliability and verification
6. Process tracking and cleanup

Usage:
    from service_manager import ServiceManager
    
    # FastAPI server
    with ServiceManager("fastapi", 8001) as manager:
        if manager.can_start():
            # Start FastAPI server safely
            start_fastapi_server(port=8001)
        else:
            print(f"Server already running: {manager.error_message}")
            
    # Gradio UI
    with ServiceManager("gradio_inference", 7861) as manager:
        if manager.can_start():
            # Start Gradio UI safely
            start_gradio_ui(port=7861)
        else:
            print(f"UI already running: {manager.error_message}")
"""

import os
import sys
import json
import time
import socket
import psutil
import atexit
import logging
import tempfile
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

# Import the standardized Python executable utility
from src.lora_training_pipeline.utils.process_executable import get_python_executable as get_standardized_python_executable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('service_manager')

# Enable debug logging if environment variable is set
DEBUG_MODE = os.environ.get("DEBUG_SERVICE_MANAGER", "false").lower() == "true"
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)

# Path constants
BASE_DIR = Path('.')
LOG_DIR = BASE_DIR / 'logs'
PROCESS_LOCKS_DIR = BASE_DIR / 'process_locks'
PID_FILES_DIR = BASE_DIR

# Ensure directories exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
PROCESS_LOCKS_DIR.mkdir(parents=True, exist_ok=True)

# Common service definitions with their default ports
SERVICE_TYPES = {
    "fastapi": {
        "port": 8001,
        "pid_file": PID_FILES_DIR / "inference_process.pid",
        "display_name": "FastAPI Inference Server"
    },
    "gradio_inference": {
        "port": 7861,
        "pid_file": PID_FILES_DIR / "inference_ui.pid",
        "display_name": "Gradio Inference UI"
    },
    "gradio_data_collection": {
        "port": 7862,
        "pid_file": PID_FILES_DIR / "data_collection_ui.pid",
        "display_name": "Gradio Data Collection UI"
    },
    "dashboard": {
        "port": 7863,
        "pid_file": PID_FILES_DIR / "dashboard.pid",
        "display_name": "Dashboard"
    }
}

# Cross-platform file locking
# We'll use an atomic file operation approach that works on all platforms
class FileLock:
    """Cross-platform file lock implementation using atomic file operations."""
    
    def __init__(self, lock_file: Path):
        self.lock_file = lock_file
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.temp_file = None
        self.acquired = False
        self.pid = os.getpid()
        
    def acquire(self, timeout: float = 10.0, retry_interval: float = 0.1) -> bool:
        """
        Acquire lock with timeout and retries.
        
        Args:
            timeout: Maximum time to wait for lock (seconds)
            retry_interval: Time between retries (seconds)
            
        Returns:
            bool: True if lock acquired, False otherwise
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self._try_acquire():
                self.acquired = True
                return True
                
            # Wait before retry
            time.sleep(retry_interval)
        
        return False
        
    def _try_acquire(self) -> bool:
        """
        Try to acquire the lock once.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if lock file exists and is valid
        if self.lock_file.exists():
            # Try to verify if the lock is stale
            try:
                lock_data = self._read_lock_file()
                if lock_data and "pid" in lock_data:
                    pid = lock_data["pid"]
                    # Check if process is still running
                    if _is_process_running(pid):
                        # Lock is still valid
                        return False
                    else:
                        # Process doesn't exist, lock is stale - try to remove it
                        try:
                            self.lock_file.unlink()
                            logger.debug(f"Removed stale lock from PID {pid}")
                        except Exception as e:
                            logger.warning(f"Could not remove stale lock: {e}")
                            return False
                else:
                    # Invalid lock file - try to remove it
                    try:
                        self.lock_file.unlink()
                        logger.debug("Removed invalid lock file")
                    except Exception as e:
                        logger.warning(f"Could not remove invalid lock file: {e}")
                        return False
            except Exception as e:
                logger.warning(f"Error reading lock file: {e}")
                return False
        
        # Try to create lock file atomically
        try:
            # Prepare temp file with our lock data
            self.temp_file = self.lock_file.with_suffix(f".{self.pid}.tmp")
            lock_data = {
                "pid": self.pid,
                "timestamp": datetime.now().isoformat(),
                "hostname": socket.gethostname(),
                "process_name": Path(sys.argv[0]).name
            }
            
            # Write to temp file
            with open(self.temp_file, 'w') as f:
                json.dump(lock_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Ensure file is written to disk
                
            # Try atomic rename - this will fail if lock file was created meanwhile
            try:
                # Use replace() which is atomic on both Windows and Unix
                self.temp_file.replace(self.lock_file)
                logger.debug(f"Acquired lock file: {self.lock_file}")
                return True
            except Exception as e:
                # Another process likely created the lock file first
                logger.debug(f"Failed to acquire lock (rename failed): {e}")
                try:
                    if self.temp_file.exists():
                        self.temp_file.unlink()
                except Exception as cleanup_err:
                    print(f"[DEBUG] Error cleaning up temp file during lock acquisition: {type(cleanup_err).__name__}: {cleanup_err}")
                    pass
                return False
                
        except Exception as e:
            logger.warning(f"Error acquiring lock: {e}")
            try:
                if self.temp_file and self.temp_file.exists():
                    self.temp_file.unlink()
            except Exception as cleanup_err2:
                print(f"[DEBUG] Error cleaning up temp file during lock error: {type(cleanup_err2).__name__}: {cleanup_err2}")
                pass
            return False
            
    def release(self) -> bool:
        """
        Release the lock if held.
        
        Returns:
            bool: True if successfully released, False otherwise
        """
        if not self.acquired:
            return True
            
        try:
            # Verify it's our lock before removing
            lock_data = self._read_lock_file()
            if lock_data and lock_data.get("pid") == self.pid:
                self.lock_file.unlink()
                logger.debug(f"Released lock file: {self.lock_file}")
                self.acquired = False
                return True
            else:
                logger.warning("Not releasing lock that belongs to another process")
                return False
        except Exception as e:
            logger.warning(f"Error releasing lock: {e}")
            return False
            
    def _read_lock_file(self) -> Optional[Dict[str, Any]]:
        """
        Read lock file contents.
        
        Returns:
            dict: Lock data if readable, None otherwise
        """
        try:
            if not self.lock_file.exists():
                return None
                
            with open(self.lock_file, 'r') as f:
                content = f.read().strip()
                
            if not content:
                return None
                
            try:
                # Try to parse as JSON
                data = json.loads(content)
                return data
            except json.JSONDecodeError:
                # Try simple integer format (legacy)
                try:
                    pid = int(content)
                    return {"pid": pid, "legacy_format": True}
                except ValueError:
                    logger.warning(f"Invalid lock file content: {content[:50]}")
                    return None
        except Exception as e:
            logger.warning(f"Error reading lock file: {e}")
            return None
            
    def __enter__(self):
        self.acquire()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False  # Don't suppress exceptions

def _is_port_in_use(port: int, host: str = "127.0.0.1", timeout: float = 1.0, retries: int = 2) -> bool:
    """
    Check if a port is in use with proper error handling and retries.
    
    Args:
        port: Port number to check
        host: Host address to check
        timeout: Socket connection timeout
        retries: Number of connection attempts
        
    Returns:
        bool: True if port is in use, False otherwise
    """
    # Validate port
    try:
        port = int(port)
        if port < 1 or port > 65535:
            logger.error(f"Invalid port number: {port}")
            raise ValueError(f"Invalid port number: {port}")
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid port parameter: {e}")
        raise ValueError(f"Invalid port parameter: {e}")
        
    logger.debug(f"Checking if port {port} is in use on {host}")
    
    for attempt in range(retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                result = s.connect_ex((host, port))
                port_in_use = (result == 0)
                
                logger.debug(f"Port {port} check (attempt {attempt+1}/{retries}): "
                           f"{'IN USE' if port_in_use else 'AVAILABLE'} (code {result})")
                
                if port_in_use:
                    return True
                    
                # For better reliability, wait before retry
                if attempt < retries - 1:
                    time.sleep(0.5)
                    continue
                    
                # If we get here on last retry, port is available
                return False
                
        except socket.error as e:
            logger.warning(f"Socket error checking port {port}: {e}")
            # Try again if we have retries left
            if attempt < retries - 1:
                continue
            else:
                # If we can't check reliably, assume it might be in use (safer)
                logger.error(f"Could not reliably check port {port}")
                return True
                
        except Exception as e:
            logger.error(f"Unexpected error checking port {port}: {e}")
            # If we can't check reliably, assume it might be in use (safer)
            return True
            
    # Should never get here, but just in case
    return True

def _is_process_running(pid: int) -> bool:
    """
    Check if a process is running and not a zombie.
    
    Args:
        pid: Process ID to check
        
    Returns:
        bool: True if process is running, False otherwise
    """
    try:
        # Validate pid
        pid = int(pid)
        if pid <= 0:
            logger.error(f"Invalid process ID: {pid}")
            return False
    except (TypeError, ValueError):
        logger.error(f"Invalid process ID parameter: {pid}")
        return False
        
    try:
        process = psutil.Process(pid)
        
        # Check if process is zombie
        if process.status() == psutil.STATUS_ZOMBIE:
            logger.debug(f"Process {pid} is a zombie")
            return False
            
        # Check if process responds to signals
        try:
            os.kill(pid, 0)  # This doesn't actually send a signal
            logger.debug(f"Process {pid} is running and responsive")
            return True
        except OSError:
            logger.debug(f"Process {pid} doesn't respond to signals")
            return False
            
    except psutil.NoSuchProcess:
        logger.debug(f"Process {pid} doesn't exist")
        return False
    except psutil.AccessDenied:
        # If we can't access process info, it might still be running
        logger.debug(f"Access denied for process {pid}, assuming it's running")
        return True

def _write_pid_file(pid_file: Path, pid: int, metadata: Dict[str, Any]) -> bool:
    """
    Write PID file with metadata in a standardized format.
    
    Args:
        pid_file: Path to PID file
        pid: Process ID
        metadata: Additional metadata to include
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure parent directory exists
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create standard data structure
        data = {
            "pid": pid,
            "timestamp": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "process_name": Path(sys.argv[0]).name
        }
        
        # Add metadata
        data.update(metadata)
        
        # Write to temporary file first for atomic operation
        temp_file = pid_file.with_suffix(f".{pid}.tmp")
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk
            
        # Atomic replace
        temp_file.replace(pid_file)
        logger.debug(f"Wrote PID file: {pid_file}")
        return True
    except Exception as e:
        logger.error(f"Error writing PID file {pid_file}: {e}")
        logger.debug(traceback.format_exc())
        return False

def _read_pid_file(pid_file: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Read PID file with robust error handling.
    
    Args:
        pid_file: Path to PID file
        
    Returns:
        Tuple[Optional[Dict], Optional[str]]: 
            - PID data or None if not readable
            - Error message or None if successful
    """
    try:
        if not pid_file.exists():
            return None, f"PID file {pid_file} does not exist"
            
        with open(pid_file, 'r') as f:
            content = f.read().strip()
            
        if not content:
            return None, f"PID file {pid_file} is empty"
            
        # Try parsing as JSON
        try:
            data = json.loads(content)
            
            # Validate PID
            if "pid" not in data:
                return None, f"PID file {pid_file} missing 'pid' field"
                
            try:
                data["pid"] = int(data["pid"])
                if data["pid"] <= 0:
                    return None, f"Invalid PID value: {data['pid']}"
            except (TypeError, ValueError):
                return None, f"Invalid PID value: {data.get('pid')}"
                
            return data, None
        except json.JSONDecodeError:
            # Try to parse as plain integer (legacy format)
            try:
                pid = int(content)
                if pid <= 0:
                    return None, f"Invalid PID value: {pid}"
                    
                return {"pid": pid, "legacy_format": True}, None
            except ValueError:
                return None, f"Invalid PID file format (not JSON or integer)"
    except Exception as e:
        return None, f"Error reading PID file: {e}"

def _find_process_by_port(port: int) -> Optional[Dict[str, Any]]:
    """
    Find process using a specific port.
    
    Args:
        port: Port number
        
    Returns:
        dict: Process information or None if not found
    """
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                connections = proc.connections(kind='inet')
                for conn in connections:
                    if conn.laddr.port == port and conn.status == 'LISTEN':
                        return {
                            "pid": proc.info['pid'],
                            "name": proc.info['name'],
                            "cmdline": " ".join(proc.info.get('cmdline', [])),
                            "port": port
                        }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.error(f"Error finding process by port {port}: {e}")
        
    return None

class ServiceManager:
    """
    Service Manager for ensuring only one instance of a service runs.
    
    This class provides robust service management with:
    - Cross-platform process locking
    - Port availability checking
    - PID file management
    - Service health verification
    """
    
    def __init__(
        self, 
        service_type: str, 
        port: Optional[int] = None,
        pid_file: Optional[Union[str, Path]] = None,
        timeout: float = 10.0
    ):
        """
        Initialize the Service Manager.
        
        Args:
            service_type: Type of service ('fastapi', 'gradio_inference', etc.)
            port: Port number (optional, uses default from SERVICE_TYPES)
            pid_file: Custom PID file path (optional, uses default from SERVICE_TYPES)
            timeout: Lock acquisition timeout in seconds
        """
        # Validate service type
        if service_type not in SERVICE_TYPES and not pid_file:
            raise ValueError(f"Unknown service type: {service_type}. "
                           f"Must be one of {list(SERVICE_TYPES.keys())} or provide a pid_file")
            
        self.service_type = service_type
        self.display_name = SERVICE_TYPES.get(service_type, {}).get("display_name", service_type)
        self.timeout = timeout
        self.error_message = None
        self.pid = os.getpid()
        
        # Get port (from parameter, SERVICE_TYPES, or default to None)
        self.port = port or SERVICE_TYPES.get(service_type, {}).get("port", None)
        
        # Get PID file path
        if pid_file:
            self.pid_file = Path(pid_file)
        else:
            self.pid_file = SERVICE_TYPES.get(service_type, {}).get("pid_file", 
                          PID_FILES_DIR / f"{service_type}.pid")
            
        # Create lock path
        if self.port:
            self.lock_file = PROCESS_LOCKS_DIR / f"{service_type}_{self.port}.lock"
        else:
            self.lock_file = PROCESS_LOCKS_DIR / f"{service_type}.lock"
            
        # Initialize lock
        self.lock = FileLock(self.lock_file)
        self.lock_acquired = False
        
        logger.debug(f"Initialized ServiceManager for {service_type} "
                   f"(port: {self.port}, pid_file: {self.pid_file})")
        
    def can_start(self) -> bool:
        """
        Check if the service can start (no other instance running).
        
        Returns:
            bool: True if service can start, False otherwise
        """
        logger.info(f"Checking if {self.display_name} can start")
        
        # Check if port is already in use
        if self.port:
            if _is_port_in_use(self.port):
                process_info = _find_process_by_port(self.port)
                if process_info:
                    self.error_message = (f"Port {self.port} is already in use by "
                                        f"process {process_info['pid']} ({process_info['name']})")
                else:
                    self.error_message = f"Port {self.port} is already in use"
                    
                logger.warning(self.error_message)
                return False
        
        # Try to acquire lock
        if not self.lock.acquire(timeout=self.timeout):
            self.error_message = (f"Another instance of {self.display_name} "
                                f"is already running (could not acquire lock)")
            logger.warning(self.error_message)
            return False
            
        # We got the lock, now check PID file
        self.lock_acquired = True
        
        if self.pid_file.exists():
            pid_data, error = _read_pid_file(self.pid_file)
            
            if error:
                logger.warning(f"Error reading PID file: {error}")
                # Continue, we'll rewrite the PID file
            elif pid_data:
                pid = pid_data.get("pid")
                if _is_process_running(pid):
                    # Process from PID file is still running
                    self.error_message = (f"Another instance of {self.display_name} "
                                        f"is already running (PID: {pid})")
                    logger.warning(self.error_message)
                    self.release()
                    return False
        
        # Write our PID file
        metadata = {
            "service_type": self.service_type,
            "display_name": self.display_name
        }
        
        if self.port:
            metadata["port"] = self.port
            
        if not _write_pid_file(self.pid_file, self.pid, metadata):
            logger.warning(f"Failed to write PID file {self.pid_file}")
            # Continue anyway, this is not critical
            
        # Register cleanup
        self._register_cleanup()
        
        logger.info(f"{self.display_name} can start (acquired lock and wrote PID file)")
        return True
        
    def release(self):
        """Release lock and clean up PID file when service is stopping."""
        if not self.lock_acquired:
            return
            
        logger.debug(f"Releasing service lock for {self.display_name}")
        
        # Clean up PID file if it's ours
        if self.pid_file.exists():
            pid_data, _ = _read_pid_file(self.pid_file)
            if pid_data and pid_data.get("pid") == self.pid:
                try:
                    self.pid_file.unlink()
                    logger.debug(f"Removed PID file {self.pid_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove PID file {self.pid_file}: {e}")
        
        # Release lock
        self.lock.release()
        self.lock_acquired = False
        
    def _register_cleanup(self):
        """Register cleanup handlers for proper shutdown."""
        # Register atexit handler
        atexit.register(self.release)
        
        # Register signal handlers
        try:
            import signal
            
            def signal_handler(sig, frame):
                logger.info(f"Received signal {sig}, releasing lock")
                self.release()
                sys.exit(0)
                
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except Exception as e:
            logger.warning(f"Failed to register signal handlers: {e}")
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False  # Don't suppress exceptions
        
    @staticmethod
    def check_port_availability(port: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if a port is available for binding.
        
        Args:
            port: Port number to check
            
        Returns:
            Tuple[bool, Optional[Dict]]: (port_available, process_info)
                - port_available: True if port is available, False if in use
                - process_info: Information about process using the port, or None
        """
        if _is_port_in_use(port):
            process_info = _find_process_by_port(port)
            return False, process_info
        return True, None
        
    @staticmethod
    def check_service_status(service_type: str) -> Dict[str, Any]:
        """
        Check if a service is running.
        
        Args:
            service_type: Type of service
            
        Returns:
            dict: Service status information
        """
        result = {
            "service_type": service_type,
            "display_name": SERVICE_TYPES.get(service_type, {}).get("display_name", service_type),
            "running": False,
            "port_active": False,
            "pid": None,
            "port": None,
            "pid_file_exists": False,
            "pid_file_valid": False,
            "lock_file_exists": False
        }
        
        # Get expected values
        service_info = SERVICE_TYPES.get(service_type, {})
        pid_file = service_info.get("pid_file")
        port = service_info.get("port")
        
        if not pid_file:
            return result
            
        # Check PID file
        pid_file = Path(pid_file)
        result["pid_file_exists"] = pid_file.exists()
        
        if result["pid_file_exists"]:
            pid_data, error = _read_pid_file(pid_file)
            result["pid_file_valid"] = error is None and pid_data is not None
            
            if result["pid_file_valid"]:
                result["pid"] = pid_data.get("pid")
                result["port"] = pid_data.get("port", port)
                result["running"] = _is_process_running(result["pid"])
                
                # Check if port is actually bound by this process
                if result["port"] and result["running"]:
                    try:
                        process = psutil.Process(result["pid"])
                        for conn in process.connections(kind='inet'):
                            if conn.laddr.port == result["port"] and conn.status == 'LISTEN':
                                result["port_active"] = True
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Fall back to simple port check
                        result["port_active"] = _is_port_in_use(result["port"])
        
        # Check lock file
        if port:
            lock_file = PROCESS_LOCKS_DIR / f"{service_type}_{port}.lock"
        else:
            lock_file = PROCESS_LOCKS_DIR / f"{service_type}.lock"
            
        result["lock_file_exists"] = lock_file.exists()
        
        return result
        
    @staticmethod
    def cleanup_stale_services() -> Dict[str, Any]:
        """
        Cleanup stale services by removing invalid PID files and locks.
        
        Returns:
            dict: Results of cleanup operation
        """
        result = {
            "stale_pid_files_removed": 0,
            "stale_locks_removed": 0,
            "service_status": {}
        }
        
        # Check all service types
        for service_type, info in SERVICE_TYPES.items():
            status = ServiceManager.check_service_status(service_type)
            result["service_status"][service_type] = status
            
            # Check for stale PID file
            if status["pid_file_exists"] and (not status["pid_file_valid"] or not status["running"]):
                try:
                    Path(info["pid_file"]).unlink()
                    result["stale_pid_files_removed"] += 1
                    logger.info(f"Removed stale PID file for {service_type}")
                except Exception as e:
                    logger.warning(f"Failed to remove stale PID file for {service_type}: {e}")
            
            # Check for stale lock file
            lock_file = PROCESS_LOCKS_DIR / f"{service_type}_{info.get('port', '')}.lock"
            if lock_file.exists():
                lock = FileLock(lock_file)
                if lock.acquire(timeout=0.1):
                    # We got the lock, meaning it was stale
                    lock.release()  # Also removes the file
                    result["stale_locks_removed"] += 1
                    logger.info(f"Removed stale lock file for {service_type}")
                    
        return result

# Helper functions to find the correct Python executable
def get_python_executable() -> str:
    """
    Get the correct Python executable path based on the current environment.

    This function implements a standardized approach to finding the Python executable
    that should be used for spawning subprocesses. It handles:

    1. Windows vs. Linux/Unix environments
    2. Virtual environments
    3. Hard-coded paths in WSL environments
    4. Configuration from environment variables

    Returns:
        str: Path to the Python executable
    """
    # Check if an override is explicitly set in environment variables
    if "PYTHON_EXECUTABLE_OVERRIDE" in os.environ:
        python_exe = os.environ["PYTHON_EXECUTABLE_OVERRIDE"]
        logger.info(f"Using override Python executable from environment: {python_exe}")
        return python_exe

    # Handle WSL-specific path correction
    # This matches the logic in run_pipeline.py that overrides sys.executable
    if os.path.exists('/proc/version'):
        # Check if running in WSL
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    # Check for hard-coded Windows path from run_pipeline.py
                    win_path = r"C:\Users\arnon\Documents\dev\projects\incoming\LoRA_Training_Pipeline\.venv\Scripts\python.exe"
                    logger.info(f"Running in WSL, checking for Windows Python path: {win_path}")
                    if os.path.exists(win_path):
                        return win_path
        except Exception as e:
            logger.warning(f"Error checking WSL environment: {e}")

    # Use sys.executable as the primary source - the same Python that's running now
    logger.debug(f"Using current Python executable: {sys.executable}")
    return sys.executable

# Helper functions for process execution
def start_process(command: List[str], env: Optional[Dict[str, str]] = None) -> int:
    """
    Start a process with the given command.

    Args:
        command: Command to execute
        env: Environment variables

    Returns:
        int: Process ID
    """
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)

    # Start the process
    try:
        process = subprocess.Popen(
            command,
            env=env_vars,
            start_new_session=True  # Detach from parent
        )
    except Exception as popen_err:
        print(f"[DEBUG] Service manager subprocess.Popen error type: {type(popen_err).__name__}")
        print(f"[DEBUG] Service manager subprocess.Popen error details: {popen_err}")
        print(f"[DEBUG] Service manager subprocess.Popen command: {command}")
        print(f"[DEBUG] Service manager subprocess.Popen env_vars keys: {list(env_vars.keys())}")
        raise

    return process.pid

# Specialized service start functions
def start_fastapi_server(port: int = 8001, model_path: Optional[str] = None) -> int:
    """
    Start FastAPI server with proper service management.

    Args:
        port: Port number
        model_path: Path to model

    Returns:
        int: Process ID if started, 0 otherwise
    """
    with ServiceManager("fastapi", port) as manager:
        if not manager.can_start():
            logger.warning(f"Cannot start FastAPI server: {manager.error_message}")
            return 0

        # Prepare environment
        env = {
            "FASTAPI_PORT": str(port)
        }

        if model_path:
            env["MODEL_PATH"] = str(model_path)

        # Get the appropriate Python executable
        python_exe = get_standardized_python_executable()
        logger.debug(f"Using Python executable for FastAPI server: {python_exe}")

        # Start the process
        command = [
            python_exe,
            "-m",
            "uvicorn",
            "src.lora_training_pipeline.inference.fastapi_inference:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port)
        ]

        pid = start_process(command, env)
        logger.info(f"Started FastAPI server on port {port} with PID {pid}")
        return pid

def start_gradio_inference_ui(port: int = 7861, api_url: Optional[str] = None) -> int:
    """
    Start Gradio inference UI with proper service management.

    Args:
        port: Port number
        api_url: URL to FastAPI server

    Returns:
        int: Process ID if started, 0 otherwise
    """
    with ServiceManager("gradio_inference", port) as manager:
        if not manager.can_start():
            logger.warning(f"Cannot start Gradio Inference UI: {manager.error_message}")
            return 0

        # Prepare environment
        env = {
            "GRADIO_PORT": str(port)
        }

        if api_url:
            env["INFERENCE_API_URL"] = api_url

        # Get the appropriate Python executable
        python_exe = get_standardized_python_executable()
        logger.debug(f"Using Python executable for Gradio Inference UI: {python_exe}")

        # Start the process
        command = [
            python_exe,
            "-m",
            "src.lora_training_pipeline.inference.gradio_inference"
        ]

        pid = start_process(command, env)
        logger.info(f"Started Gradio Inference UI on port {port} with PID {pid}")
        return pid

def start_gradio_data_collection(port: int = 7862) -> int:
    """
    Start Gradio data collection with proper service management.

    Args:
        port: Port number

    Returns:
        int: Process ID if started, 0 otherwise
    """
    with ServiceManager("gradio_data_collection", port) as manager:
        if not manager.can_start():
            logger.warning(f"Cannot start Gradio Data Collection: {manager.error_message}")
            return 0

        # Prepare environment
        env = {
            "GRADIO_PORT": str(port)
        }

        # Get the appropriate Python executable
        python_exe = get_standardized_python_executable()
        logger.debug(f"Using Python executable for Gradio Data Collection: {python_exe}")

        # Start the process
        command = [
            python_exe,
            "-m",
            "src.lora_training_pipeline.data_collection.gradio_app"
        ]

        pid = start_process(command, env)
        logger.info(f"Started Gradio Data Collection on port {port} with PID {pid}")
        return pid

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Service Manager for LoRA Training Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument("--status", action="store_true", 
                     help="Check status of all services")
    group.add_argument("--cleanup", action="store_true", 
                     help="Clean up stale services")
    group.add_argument("--start", choices=list(SERVICE_TYPES.keys()),
                     help="Start a specific service")
    group.add_argument("--restart", choices=list(SERVICE_TYPES.keys()),
                     help="Restart a specific service")
    group.add_argument("--check-port", type=int,
                     help="Check if a specific port is available")
    
    parser.add_argument("--port", type=int,
                      help="Port for the service (overrides default)")
    parser.add_argument("--model-path", 
                      help="Model path for FastAPI server")
    parser.add_argument("--api-url", 
                      help="API URL for Gradio inference UI")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug mode if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    if args.status:
        print("Checking service status...")
        for service_type in SERVICE_TYPES:
            status = ServiceManager.check_service_status(service_type)
            name = status["display_name"]
            
            if status["running"]:
                print(f"✅ {name} is RUNNING (PID: {status['pid']})")
                if status["port_active"]:
                    print(f"   Port {status['port']} is active")
                else:
                    print(f"   Port {status['port']} is NOT active")
            else:
                print(f"❌ {name} is NOT running")
                
                if status["pid_file_exists"]:
                    if status["pid_file_valid"]:
                        print(f"   PID file exists with PID {status['pid']} (process not running)")
                    else:
                        print(f"   PID file exists but is invalid")
                        
                if status["lock_file_exists"]:
                    print(f"   Lock file exists (likely stale)")
    
    elif args.cleanup:
        print("Cleaning up stale services...")
        result = ServiceManager.cleanup_stale_services()
        
        print(f"Removed {result['stale_pid_files_removed']} stale PID files")
        print(f"Removed {result['stale_locks_removed']} stale lock files")
        
        # Show current status after cleanup
        print("\nService status after cleanup:")
        for service_type, status in result["service_status"].items():
            name = status["display_name"]
            if status["running"]:
                print(f"✅ {name} is RUNNING (PID: {status['pid']})")
            else:
                print(f"❌ {name} is NOT running")
    
    elif args.check_port:
        port = args.check_port
        available, process_info = ServiceManager.check_port_availability(port)
        
        if available:
            print(f"✅ Port {port} is AVAILABLE")
        else:
            print(f"❌ Port {port} is IN USE")
            if process_info:
                print(f"   Used by PID {process_info['pid']} ({process_info['name']})")
    
    elif args.start:
        service_type = args.start
        port = args.port or SERVICE_TYPES[service_type]["port"]
        
        print(f"Starting {SERVICE_TYPES[service_type]['display_name']} on port {port}...")
        
        if service_type == "fastapi":
            pid = start_fastapi_server(port, args.model_path)
            if pid > 0:
                print(f"✅ Started FastAPI server on port {port} (PID: {pid})")
            else:
                print("❌ Failed to start FastAPI server")
                
        elif service_type == "gradio_inference":
            pid = start_gradio_inference_ui(port, args.api_url)
            if pid > 0:
                print(f"✅ Started Gradio Inference UI on port {port} (PID: {pid})")
            else:
                print("❌ Failed to start Gradio Inference UI")
                
        elif service_type == "gradio_data_collection":
            pid = start_gradio_data_collection(port)
            if pid > 0:
                print(f"✅ Started Gradio Data Collection on port {port} (PID: {pid})")
            else:
                print("❌ Failed to start Gradio Data Collection")
                
        elif service_type == "dashboard":
            # Custom dashboard start logic
            print("Dashboard service not yet implemented")
    
    elif args.restart:
        service_type = args.restart
        port = args.port or SERVICE_TYPES[service_type]["port"]

        print(f"Restarting {SERVICE_TYPES[service_type]['display_name']}...")

        # Check current status
        status = ServiceManager.check_service_status(service_type)
        if status["running"]:
            # Try to terminate the process
            try:
                print(f"Terminating existing process (PID: {status['pid']})")
                process = psutil.Process(status["pid"])
                process.terminate()

                # Wait for termination
                try:
                    process.wait(timeout=5)
                    print("Process terminated gracefully")
                except psutil.TimeoutExpired:
                    print("Process did not terminate gracefully, sending SIGKILL")
                    process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"Error terminating process: {e}")

            # Clean up PID and lock files
            print("Cleaning up PID and lock files")
            try:
                pid_file = SERVICE_TYPES[service_type]["pid_file"]
                if Path(pid_file).exists():
                    Path(pid_file).unlink()
            except Exception as e:
                print(f"Error removing PID file: {e}")

            # Try to remove lock file
            if status["lock_file_exists"]:
                try:
                    if port:
                        lock_file = PROCESS_LOCKS_DIR / f"{service_type}_{port}.lock"
                    else:
                        lock_file = PROCESS_LOCKS_DIR / f"{service_type}.lock"

                    if lock_file.exists():
                        lock_file.unlink()
                except Exception as e:
                    print(f"Error removing lock file: {e}")

        # Get and display the Python executable that will be used
        python_exe = get_standardized_python_executable()
        print(f"Using Python executable: {python_exe}")

        # Start the service again
        print(f"Starting {SERVICE_TYPES[service_type]['display_name']} on port {port}")

        if service_type == "fastapi":
            pid = start_fastapi_server(port, args.model_path)
            if pid > 0:
                print(f"✅ Started FastAPI server on port {port} (PID: {pid})")
            else:
                print("❌ Failed to start FastAPI server")

        elif service_type == "gradio_inference":
            pid = start_gradio_inference_ui(port, args.api_url)
            if pid > 0:
                print(f"✅ Started Gradio Inference UI on port {port} (PID: {pid})")
            else:
                print("❌ Failed to start Gradio Inference UI")

        elif service_type == "gradio_data_collection":
            pid = start_gradio_data_collection(port)
            if pid > 0:
                print(f"✅ Started Gradio Data Collection on port {port} (PID: {pid})")
            else:
                print("❌ Failed to start Gradio Data Collection")

        elif service_type == "dashboard":
            # Custom dashboard start logic
            print("Dashboard service not yet implemented")