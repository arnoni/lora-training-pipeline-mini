#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/process_watchdog.py
"""
Process Watchdog Module

This module provides a pre-flight "process watchdog" that runs before each UI/server launch.
It implements robust process management to:

1. Validate PID files against live processes using psutil.
2. Automatically kill or allow reuse of stray/stale PIDs and clear PID files atomically.
3. Programmatically select free ports for services.
4. Utilize file-locks to enforce single instances of services.

Usage:
    # To ensure a service can start cleanly
    from src.lora_training_pipeline.utils.process_watchdog import ProcessWatchdog
    
    # Create a watchdog for a specific service
    watchdog = ProcessWatchdog("fastapi_inference", 8001)
    
    # Check if the service can start
    if watchdog.can_start():
        # Start the service
        start_service()
        
        # Register the service PID
        watchdog.register_pid(os.getpid())
    else:
        print(f"Cannot start service: {watchdog.error_message}")
"""

import os
import sys
import time
import socket
import logging
import signal
import datetime
import threading
import atexit
from pathlib import Path
import psutil
import json
import fcntl
import uuid
import traceback
from typing import Dict, List, Optional, Any, Tuple, Union, Set

# Import existing utilities from process_core
try:
    from src.lora_training_pipeline.utils.process_core import (
        FileLock, PidFile, PortManager, ProcessVerifier, is_process_running,
        cleanup_stale_processes, log_event
    )
except ImportError:
    # Fallback implementations if process_core is not available
    from src.lora_training_pipeline.utils.port_conflict_resolver import (
        is_port_in_use, check_process_running as is_process_running
    )
    
    # Define simplified versions of required classes
    class FileLock:
        """Simplified FileLock implementation."""
        def __init__(self, lock_file):
            self.lock_file = Path(lock_file)
            self.lock_acquired = False
            
        def acquire(self, timeout=0):
            """Acquire lock with timeout."""
            if self.lock_file.exists():
                return False
                
            try:
                with open(self.lock_file, 'w') as f:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    f.write(str(os.getpid()))
                    self.lock_acquired = True
                    return True
            except (IOError, OSError):
                return False
                
        def release(self):
            """Release lock."""
            if self.lock_acquired and self.lock_file.exists():
                try:
                    self.lock_file.unlink()
                    self.lock_acquired = False
                    return True
                except (IOError, OSError):
                    return False
            return True
    
    class PidFile:
        """Simplified PidFile implementation."""
        def __init__(self, pid_file):
            self.pid_file = Path(pid_file)
            
        def read(self):
            """Read PID file."""
            if not self.pid_file.exists():
                return None
                
            try:
                with open(self.pid_file, 'r') as f:
                    content = f.read().strip()
                    
                    try:
                        # Try JSON format first
                        data = json.loads(content)
                        if "pid" in data:
                            return data
                    except json.JSONDecodeError:
                        # Try plain integer format
                        try:
                            pid = int(content)
                            return {"pid": pid, "timestamp": datetime.datetime.now().isoformat()}
                        except ValueError:
                            return None
            except Exception:
                return None
                
        def write(self, pid, metadata=None):
            """Write PID file."""
            try:
                data = {
                    "pid": int(pid),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                if metadata:
                    data.update(metadata)
                    
                with open(self.pid_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                return True
            except Exception:
                return False
                
        def is_stale(self):
            """Check if PID file is stale."""
            data = self.read()
            if not data:
                return True
                
            pid = data.get("pid")
            if not pid:
                return True
                
            return not is_process_running(pid)
            
        def remove(self):
            """Remove PID file."""
            if self.pid_file.exists():
                try:
                    self.pid_file.unlink()
                    return True
                except Exception:
                    return False
            return True
    
    # Log event function
    def log_event(event_type, details):
        """Log event to console."""
        timestamp = datetime.datetime.now().isoformat()
        print(f"[{timestamp}] EVENT: {event_type} - {details}")
        
    # Cleanup stale processes function
    def cleanup_stale_processes(pidfiles=None):
        """Clean up stale processes."""
        if not pidfiles:
            return {"stale_processes": 0, "cleaned_pid_files": 0, "errors": []}
            
        result = {"stale_processes": 0, "cleaned_pid_files": 0, "errors": []}
        
        for pid_file in pidfiles:
            try:
                pidfile = PidFile(pid_file)
                if pidfile.is_stale():
                    if pidfile.remove():
                        result["stale_processes"] += 1
                        result["cleaned_pid_files"] += 1
            except Exception as e:
                result["errors"].append(str(e))
                
        return result

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./process_watchdog.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('process_watchdog')

# Standard PID file paths
INFERENCE_PROCESS_PID_FILE = Path('./inference_process.pid')
DATA_COLLECTION_PID_FILE = Path('./data_collection_ui.pid')
INFERENCE_UI_PID_FILE = Path('./inference_ui.pid')
PROCESS_LOCKS_DIR = Path('./process_locks')

# Ensure locks directory exists
PROCESS_LOCKS_DIR.mkdir(parents=True, exist_ok=True)

# Service type to PID file mapping
SERVICE_PID_FILES = {
    "fastapi_inference": INFERENCE_PROCESS_PID_FILE,
    "data_collection": DATA_COLLECTION_PID_FILE,
    "inference_ui": INFERENCE_UI_PID_FILE
}

# Default ports for services
DEFAULT_PORTS = {
    "fastapi_inference": 8001,
    "data_collection": 7862,
    "inference_ui": 7861
}

class ProcessWatchdog:
    """
    Process watchdog for robust service management.
    
    This class provides pre-flight checks before launching services to ensure
    clean startup and avoid port/PID conflicts.
    """
    def __init__(self, service_type: str, port: Optional[int] = None, 
                 pid_file: Optional[Union[str, Path]] = None,
                 auto_cleanup: bool = True):
        """
        Initialize process watchdog.
        
        Args:
            service_type: Type of service (e.g., 'fastapi_inference', 'data_collection')
            port: Port number for the service (uses default if not specified)
            pid_file: Custom PID file path (uses standard path if not specified)
            auto_cleanup: Whether to automatically clean up stale processes
        """
        self.service_type = service_type
        self.port = port if port is not None else DEFAULT_PORTS.get(service_type)
        self.pid_file = Path(pid_file) if pid_file else SERVICE_PID_FILES.get(service_type)
        self.auto_cleanup = auto_cleanup
        self.error_message = None
        self.process_lock = None
        self.instance_id = str(uuid.uuid4())
        
        # Validation
        if not self.port:
            self.error_message = f"No port specified for service type: {service_type}"
            logger.error(self.error_message)
            
        if not self.pid_file:
            self.error_message = f"No PID file path for service type: {service_type}"
            logger.error(self.error_message)
            
        logger.info(f"Initialized ProcessWatchdog for {service_type} on port {self.port}")
        logger.info(f"PID file: {self.pid_file}")
        
    def can_start(self) -> bool:
        """
        Check if the service can start.
        
        Performs a series of checks to determine if the service can start cleanly:
        1. Validates PID file against live processes
        2. Checks if port is in use
        3. Tries to acquire process lock
        
        Returns:
            bool: True if service can start, False otherwise
        """
        if self.error_message:
            logger.error(f"Cannot start due to initialization error: {self.error_message}")
            return False
            
        logger.info(f"Checking if {self.service_type} can start on port {self.port}")
        
        # First, clean up stale processes if enabled
        if self.auto_cleanup:
            self._cleanup_stale_processes()
            
        # Check if PID file exists and points to a running process
        pid_data = self._check_pid_file()
        if pid_data and pid_data.get("running", False):
            self.error_message = (f"Service {self.service_type} is already running "
                                 f"with PID {pid_data.get('pid')}")
            logger.warning(self.error_message)
            return False
            
        # Check if port is in use
        if self._is_port_in_use():
            # Get process info for better error message
            proc_info = self._get_process_by_port()
            if proc_info:
                self.error_message = (f"Port {self.port} is already in use by PID {proc_info.get('pid')} "
                                     f"({proc_info.get('name', 'unknown')})")
            else:
                self.error_message = f"Port {self.port} is already in use by an unknown process"
                
            logger.warning(self.error_message)
            return False
            
        # Try to acquire process lock
        if not self._acquire_process_lock():
            self.error_message = f"Could not acquire process lock for {self.service_type} on port {self.port}"
            logger.warning(self.error_message)
            return False
            
        logger.info(f"Service {self.service_type} can start on port {self.port}")
        return True
        
    def register_pid(self, pid: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register the service PID.
        
        Args:
            pid: Process ID to register
            metadata: Optional metadata to include in PID file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not metadata:
            metadata = {}
            
        # Add standard metadata
        full_metadata = {
            "service_type": self.service_type,
            "port": self.port,
            "timestamp": datetime.datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "instance_id": self.instance_id
        }
        full_metadata.update(metadata)
        
        # Write PID file
        try:
            # Ensure parent directory exists
            self.pid_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write PID file with metadata
            pidfile = PidFile(self.pid_file)
            result = pidfile.write(pid, full_metadata)
            
            if result:
                logger.info(f"Registered PID {pid} for service {self.service_type}")
                
                # Register cleanup handler
                atexit.register(self.cleanup)
                
                # Register signal handlers
                self._register_signal_handlers()
                
                return True
            else:
                self.error_message = f"Failed to write PID file {self.pid_file}"
                logger.error(self.error_message)
                return False
                
        except Exception as e:
            self.error_message = f"Error registering PID: {e}"
            logger.error(self.error_message)
            logger.error(traceback.format_exc())
            return False
            
    def cleanup(self) -> bool:
        """
        Clean up process resources.
        
        Returns:
            bool: True if successful, False otherwise
        """
        success = True
        logger.info(f"Cleaning up resources for {self.service_type}")
        
        # Release process lock
        if self.process_lock:
            try:
                logger.info("Releasing process lock")
                success = self.process_lock.release() and success
            except Exception as e:
                logger.error(f"Error releasing process lock: {e}")
                success = False
                
        # Remove PID file if it belongs to this process
        try:
            if self.pid_file.exists():
                pidfile = PidFile(self.pid_file)
                pid_data = pidfile.read()
                
                if pid_data and pid_data.get("instance_id") == self.instance_id:
                    logger.info(f"Removing PID file {self.pid_file}")
                    success = pidfile.remove() and success
                else:
                    logger.info(f"PID file {self.pid_file} belongs to another instance")
        except Exception as e:
            logger.error(f"Error removing PID file: {e}")
            success = False
            
        return success
        
    def _check_pid_file(self) -> Optional[Dict[str, Any]]:
        """
        Check if PID file exists and points to a running process.
        
        Returns:
            dict: PID file data with 'running' flag, or None if file doesn't exist
        """
        if not self.pid_file.exists():
            logger.info(f"PID file {self.pid_file} does not exist")
            return None
            
        try:
            # Read PID file
            pidfile = PidFile(self.pid_file)
            pid_data = pidfile.read()
            
            if not pid_data:
                logger.warning(f"PID file {self.pid_file} exists but is invalid")
                if self.auto_cleanup:
                    logger.info(f"Removing invalid PID file {self.pid_file}")
                    pidfile.remove()
                return None
                
            # Check if process is running
            pid = pid_data.get("pid")
            if not pid:
                logger.warning(f"PID file {self.pid_file} does not contain a PID")
                if self.auto_cleanup:
                    logger.info(f"Removing invalid PID file {self.pid_file}")
                    pidfile.remove()
                return None
                
            running = is_process_running(pid)
            pid_data["running"] = running
            
            if running:
                logger.info(f"Process with PID {pid} is running")
            else:
                logger.info(f"Process with PID {pid} is not running")
                if self.auto_cleanup:
                    logger.info(f"Removing stale PID file {self.pid_file}")
                    pidfile.remove()
                    
            return pid_data
            
        except Exception as e:
            logger.error(f"Error checking PID file {self.pid_file}: {e}")
            logger.error(traceback.format_exc())
            return None
            
    def _is_port_in_use(self) -> bool:
        """
        Check if port is in use.
        
        Returns:
            bool: True if port is in use, False otherwise
        """
        try:
            result = is_port_in_use(self.port)
            logger.info(f"Port {self.port} is {'in use' if result else 'available'}")
            return result
        except Exception as e:
            logger.error(f"Error checking if port {self.port} is in use: {e}")
            logger.error(traceback.format_exc())
            # Assume port is in use if we can't check
            return True
            
    def _get_process_by_port(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the process using the port.
        
        Returns:
            dict: Process information, or None if no process is using the port
        """
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    connections = proc.connections(kind='inet')
                    for conn in connections:
                        if conn.laddr.port == self.port:
                            # Found process using this port
                            return {
                                "pid": proc.pid,
                                "name": proc.name(),
                                "cmdline": " ".join(proc.cmdline())
                            }
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
        except Exception as e:
            logger.error(f"Error getting process by port {self.port}: {e}")
            logger.error(traceback.format_exc())
            
        return None
        
    def _acquire_process_lock(self) -> bool:
        """
        Acquire process lock.
        
        Returns:
            bool: True if lock was acquired, False otherwise
        """
        try:
            # Create lock file path
            lock_file = PROCESS_LOCKS_DIR / f"{self.service_type}_{self.port}.lock"
            
            # Create file lock
            self.process_lock = FileLock(lock_file)
            
            # Try to acquire lock
            result = self.process_lock.acquire(timeout=2.0)  # Allow a short timeout
            
            if result:
                logger.info(f"Acquired process lock for {self.service_type} on port {self.port}")
                
                # Register cleanup handler
                atexit.register(self._release_process_lock)
                
                return True
            else:
                logger.warning(f"Could not acquire process lock for {self.service_type} on port {self.port}")
                self.process_lock = None
                return False
                
        except Exception as e:
            logger.error(f"Error acquiring process lock: {e}")
            logger.error(traceback.format_exc())
            self.process_lock = None
            return False
            
    def _release_process_lock(self) -> bool:
        """
        Release process lock.
        
        Returns:
            bool: True if lock was released, False otherwise
        """
        if self.process_lock:
            try:
                result = self.process_lock.release()
                self.process_lock = None
                return result
            except Exception as e:
                logger.error(f"Error releasing process lock: {e}")
                logger.error(traceback.format_exc())
                return False
        return True
        
    def _cleanup_stale_processes(self) -> Dict[str, Any]:
        """
        Clean up stale processes.
        
        Returns:
            dict: Information about the cleanup
        """
        try:
            # Check if we have the specialized function
            result = cleanup_stale_processes([self.pid_file])
            logger.info(f"Cleaned up {result['stale_processes']} stale processes")
            return result
        except Exception as e:
            logger.error(f"Error cleaning up stale processes: {e}")
            logger.error(traceback.format_exc())
            return {"stale_processes": 0, "cleaned_pid_files": 0, "errors": [str(e)]}
            
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for cleanup on exit."""
        def signal_handler(sig, frame):
            """Handle signals by cleaning up and exiting."""
            logger.info(f"Received signal {signal.Signals(sig).name}, cleaning up")
            self.cleanup()
            # Use appropriate exit code
            sys.exit(0 if sig == signal.SIGTERM else 1)
            
        # Register handlers for SIGINT and SIGTERM
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        
        # Only register if they're not already custom handlers
        if original_sigint == signal.SIG_DFL or original_sigint == signal.SIG_IGN:
            signal.signal(signal.SIGINT, signal_handler)
            
        if original_sigterm == signal.SIG_DFL or original_sigterm == signal.SIG_IGN:
            signal.signal(signal.SIGTERM, signal_handler)
            
def find_available_port(start_port: int = 7800, end_port: int = 7900) -> int:
    """
    Find an available port in the specified range.
    
    Args:
        start_port: Start of port range to check
        end_port: End of port range to check
        
    Returns:
        int: Available port, or -1 if none found
    """
    for port in range(start_port, end_port + 1):
        if not is_port_in_use(port):
            return port
    return -1
    
def get_services_status() -> Dict[str, Dict[str, Any]]:
    """
    Get status of all services.
    
    Returns:
        dict: Status of each service
    """
    result = {}
    
    for service_type, pid_file in SERVICE_PID_FILES.items():
        service_status = {
            "running": False,
            "pid": None,
            "port": DEFAULT_PORTS.get(service_type),
            "pid_file_exists": False,
            "port_in_use": False
        }
        
        # Check PID file
        if pid_file.exists():
            service_status["pid_file_exists"] = True
            try:
                pidfile = PidFile(pid_file)
                pid_data = pidfile.read()
                
                if pid_data:
                    pid = pid_data.get("pid")
                    service_status["pid"] = pid
                    
                    # Check if process is running
                    if pid:
                        running = is_process_running(pid)
                        service_status["running"] = running
                        
                    # Get port from PID file data
                    port = pid_data.get("port")
                    if port:
                        service_status["port"] = port
            except Exception as e:
                service_status["error"] = str(e)
                
        # Check if port is in use
        try:
            port = service_status["port"]
            if port:
                service_status["port_in_use"] = is_port_in_use(port)
        except Exception as e:
            service_status["port_error"] = str(e)
            
        result[service_type] = service_status
        
    return result
    
def cleanup_all_services() -> Dict[str, Any]:
    """
    Clean up all services.
    
    Returns:
        dict: Results of the cleanup
    """
    result = {
        "cleaned_services": 0,
        "errors": []
    }
    
    for service_type, pid_file in SERVICE_PID_FILES.items():
        try:
            watchdog = ProcessWatchdog(service_type)
            watchdog.cleanup()
            result["cleaned_services"] += 1
        except Exception as e:
            result["errors"].append(f"Error cleaning up {service_type}: {e}")
            
    return result
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Watchdog Utility")
    parser.add_argument("--status", action="store_true", help="Show status of all services")
    parser.add_argument("--cleanup", action="store_true", help="Clean up all services")
    parser.add_argument("--check", choices=["fastapi", "data_collection", "inference_ui"], 
                       help="Check if a specific service can start")
    parser.add_argument("--port", type=int, help="Port for service check")
    
    args = parser.parse_args()
    
    if args.status:
        status = get_services_status()
        print("\nService Status:")
        print("==============")
        
        for service, info in status.items():
            print(f"\n{service}:")
            print(f"  Running: {info['running']}")
            print(f"  PID: {info['pid']}")
            print(f"  Port: {info['port']} ({'in use' if info['port_in_use'] else 'available'})")
            print(f"  PID file: {'exists' if info['pid_file_exists'] else 'missing'}")
            
            if info.get("error"):
                print(f"  Error: {info['error']}")
                
    elif args.cleanup:
        print("Cleaning up all services...")
        result = cleanup_all_services()
        print(f"Cleaned up {result['cleaned_services']} services")
        
        if result["errors"]:
            print("\nErrors:")
            for error in result["errors"]:
                print(f"  {error}")
                
    elif args.check:
        service_type = f"{args.check}_{'inference' if args.check == 'fastapi' else 'ui' if args.check in ('data_collection', 'inference_ui') else ''}"
        port = args.port or DEFAULT_PORTS.get(service_type)
        
        print(f"Checking if {service_type} can start on port {port}...")
        watchdog = ProcessWatchdog(service_type, port)
        
        if watchdog.can_start():
            print(f"Service {service_type} can start on port {port}")
        else:
            print(f"Service {service_type} cannot start: {watchdog.error_message}")
            sys.exit(1)
    else:
        parser.print_help()