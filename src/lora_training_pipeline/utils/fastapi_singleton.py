#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/fastapi_singleton.py

"""
FastAPI Singleton Manager

This module helps prevent multiple FastAPI servers from binding to the same port,
which is a common source of stability issues in the pipeline.

Features:
1. Socket-based port availability checking before binding 
2. Atomic file locking to prevent race conditions
3. Proper PID file management with structured data
4. Stale process detection and cleanup
5. Health endpoint for liveliness checks

Usage:
    from fastapi import FastAPI
    from src.lora_training_pipeline.utils.fastapi_singleton import SingletonFastAPI

    # Create a singleton-managed FastAPI app
    app = SingletonFastAPI(
        port=8001, 
        service_name="inference_server",
        pid_file="./inference_process.pid"
    )

    # Use it like a normal FastAPI app
    @app.get("/")
    def read_root():
        return {"Hello": "World"}
"""

import os
import sys
import json
import fcntl
import socket
import signal
import psutil
import atexit
import traceback
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Union, Any, Callable, TypeVar

# Enable verbose debugging
DEBUG_MODE = os.environ.get("DEBUG_PROCESS_MANAGEMENT", "true").lower() == "true"

def debug_print(*args, **kwargs):
    """Print debug information only when DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print("[FASTAPI-SINGLETON]", *args, **kwargs)

# Typing helpers
T = TypeVar('T')

class SingletonFastAPI(FastAPI):
    """
    A FastAPI extension that ensures only one instance can bind to a port.
    
    This class extends FastAPI with built-in singleton behavior to prevent
    port conflicts and race conditions when multiple processes try to start
    FastAPI servers on the same port.
    """
    
    def __init__(
        self, 
        *args,
        port: int = 8001,
        service_name: str = "fastapi_service",
        pid_file: Union[str, Path] = None,
        **kwargs
    ):
        """
        Initialize the SingletonFastAPI instance.
        
        Args:
            port: Port number this FastAPI server will use
            service_name: Name of the service (used in logs and PID file)
            pid_file: Path to the PID file (defaults to ./service_name.pid)
            *args, **kwargs: Arguments passed to FastAPI constructor
        """
        # Initialize FastAPI
        super().__init__(*args, **kwargs)
        
        # Store singleton configuration
        self.port = port
        self.service_name = service_name
        self.pid_file = Path(pid_file) if pid_file else Path(f"./{service_name}.pid")
        self.process_lock_file = Path(f"./.{service_name}_lock")
        self.startup_time = datetime.now()
        self.singleton_ready = False
        self.startup_complete = False
        
        # Register the health endpoint
        self.add_api_route("/health", self.health_endpoint, methods=["GET"])
        
        # Override startup event to add singleton check
        original_startup = getattr(self, "startup", None)
        
        @self.on_event("startup")
        async def singleton_startup_handler():
            # Check for existing process first
            if not self._ensure_singleton():
                # This will immediately exit if another instance is running
                sys.exit(1)
                
            self.singleton_ready = True
            
            # Call the original startup handler if it exists
            if original_startup:
                for handler in original_startup:
                    await handler()
                    
            # Register cleanup handlers
            self._register_cleanup_handlers()
            
            # Mark startup as complete
            self.startup_complete = True
        
        # Add middleware to check singleton status
        @self.middleware("http")
        async def singleton_check_middleware(request: Request, call_next):
            # Check if singleton is properly initialized
            if not self.singleton_ready:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"detail": "Server is shutting down or not properly initialized"}
                )
                
            # Process the request
            return await call_next(request)
    
    def _ensure_singleton(self) -> bool:
        """
        Ensure this is the only instance of the service running.
        
        This method implements robust process locking to prevent multiple
        instances from binding to the same port.
        
        Returns:
            bool: True if singleton check passed, False otherwise
        """
        debug_print(f"Checking if {self.service_name} is already running on port {self.port}")
        
        # First, check if port is already in use
        if self._is_port_in_use():
            print(f"Error: Port {self.port} is already in use by another process")
            return False
            
        # Check if PID file exists and points to a running process
        if self.pid_file.exists():
            try:
                with open(self.pid_file, 'r') as f:
                    # Use file locking to ensure consistent read
                    fcntl.flock(f, fcntl.LOCK_SH)
                    try:
                        pid_data = json.load(f)
                    except json.JSONDecodeError:
                        # Try to read as plain integer (legacy format)
                        f.seek(0)
                        content = f.read().strip()
                        try:
                            pid = int(content)
                            pid_data = {"pid": pid}
                        except ValueError:
                            # Invalid file format
                            print(f"Warning: Invalid PID file format, will overwrite")
                            pid_data = None
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
                        
                if pid_data and "pid" in pid_data:
                    pid = pid_data["pid"]
                    
                    # Check if process is still running
                    try:
                        process = psutil.Process(pid)
                        
                        # Check if zombie
                        if process.status() == psutil.STATUS_ZOMBIE:
                            print(f"Found zombie process with PID {pid}, cleaning up")
                            self.pid_file.unlink()
                        else:
                            # Process is running - check if it's actually our service
                            cmdline = " ".join(process.cmdline())
                            
                            # Look for indicators this is our service
                            if self.service_name in cmdline and "uvicorn" in cmdline:
                                print(f"Error: {self.service_name} is already running with PID {pid}")
                                return False
                            else:
                                print(f"Warning: Found process with PID {pid} but it doesn't appear to be {self.service_name}")
                                print(f"Process command line: {cmdline}")
                                
                                # Ask user what to do
                                if os.environ.get("FASTAPI_SINGLETON_FORCE_REPLACE", "").lower() == "true":
                                    print("FASTAPI_SINGLETON_FORCE_REPLACE is set, replacing PID file")
                                    self.pid_file.unlink()
                                elif os.environ.get("FASTAPI_SINGLETON_INTERACTIVE", "").lower() == "true":
                                    response = input(f"Replace PID file for {self.service_name}? [y/N] ")
                                    if response.lower() not in ['y', 'yes']:
                                        print("Aborted")
                                        return False
                                        
                                    self.pid_file.unlink()
                                else:
                                    # Default to not replacing
                                    print(f"Aborting startup to avoid conflicts")
                                    return False
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        print(f"PID file points to non-existent process {pid}, cleaning up")
                        self.pid_file.unlink()
            except Exception as e:
                print(f"Error checking PID file: {e}")
                
                # Safe approach: assume something might be running
                if os.environ.get("FASTAPI_SINGLETON_FORCE_REPLACE", "").lower() == "true":
                    print("FASTAPI_SINGLETON_FORCE_REPLACE is set, replacing PID file")
                    try:
                        self.pid_file.unlink()
                    except Exception as unlink_err:
                        print(f"[DEBUG] Error removing PID file during force replace: {type(unlink_err).__name__}: {unlink_err}")
                        pass
                else:
                    print(f"Error checking if {self.service_name} is already running")
                    return False
        
        # Check if we can create a process lock
        try:
            # Create lock file directory if it doesn't exist
            self.process_lock_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create lock file
            with open(self.process_lock_file, 'w') as f:
                # Try to get exclusive lock
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Lock acquired, write PID and metadata
                lock_data = {
                    "pid": os.getpid(),
                    "port": self.port,
                    "service_name": self.service_name,
                    "timestamp": datetime.now().isoformat(),
                    "hostname": socket.gethostname()
                }
                
                f.write(json.dumps(lock_data))
                f.flush()
                
                # Do NOT release the lock - keep it held for the process lifetime
                
                # Register cleanup handler for the lock file
                def cleanup_lock():
                    try:
                        if self.process_lock_file.exists():
                            self.process_lock_file.unlink()
                            debug_print(f"Removed process lock file {self.process_lock_file}")
                    except Exception as e:
                        print(f"Error removing process lock file: {e}")
                        
                atexit.register(cleanup_lock)
                
                debug_print(f"Acquired process lock for {self.service_name}")
        except IOError:
            print(f"Error: Could not acquire process lock, another instance of {self.service_name} may be starting")
            return False
        except Exception as e:
            print(f"Error creating process lock: {e}")
            return False
                
        # Write PID file
        try:
            with open(self.pid_file, 'w') as f:
                # Use exclusive lock while writing
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    pid_data = {
                        "pid": os.getpid(),
                        "port": self.port,
                        "service_name": self.service_name,
                        "timestamp": datetime.now().isoformat(),
                        "hostname": socket.gethostname(),
                        "python_executable": sys.executable,
                        "command": " ".join(sys.argv)
                    }
                    
                    json.dump(pid_data, f, indent=2)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            
            debug_print(f"Wrote PID file {self.pid_file}")
        except Exception as e:
            print(f"Error writing PID file: {e}")
            
            # Not critical, we can continue
            print(f"Warning: Failed to write PID file, but service will continue")
        
        # Final port check before returning success
        if self._is_port_in_use():
            # This should not happen, but check again to be sure
            print(f"Error: Port {self.port} was bound by another process during startup")
            return False
            
        print(f"{self.service_name} singleton check passed, continuing startup")
        return True
    
    def _is_port_in_use(self) -> bool:
        """
        Check if a port is already in use.
        
        Returns:
            bool: True if port is in use, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)  # Set a timeout to avoid hanging
                result = s.connect_ex(('localhost', self.port))
                port_in_use = result == 0
                debug_print(f"Port {self.port} check result: {'IN USE' if port_in_use else 'AVAILABLE'}")
                return port_in_use
        except socket.error as e:
            print(f"Warning: Socket error checking port {self.port}: {e}")
            # If we can't check reliably, assume it might be in use to be safe
            return True
        except Exception as e:
            print(f"Warning: Unexpected error checking port {self.port}: {e}")
            # If we can't check reliably, assume it might be in use to be safe
            return True
    
    def _register_cleanup_handlers(self):
        """Register handlers for clean shutdown."""
        # Register atexit handler to clean up PID file
        def cleanup_pid_file():
            try:
                if self.pid_file.exists():
                    # Read the current PID file
                    try:
                        with open(self.pid_file, 'r') as f:
                            # Use shared lock for reading
                            fcntl.flock(f, fcntl.LOCK_SH)
                            try:
                                pid_data = json.load(f)
                            except Exception:
                                pid_data = {"pid": os.getpid()}  # Fallback
                            finally:
                                fcntl.flock(f, fcntl.LOCK_UN)
                                
                        # Only remove if it's our PID
                        if pid_data.get("pid") == os.getpid():
                            self.pid_file.unlink()
                            debug_print(f"Removed PID file {self.pid_file}")
                    except Exception as e:
                        # If we can't verify, assume it's ours and remove it
                        self.pid_file.unlink()
                        debug_print(f"Removed PID file {self.pid_file} (error reading: {e})")
            except Exception as e:
                print(f"Error removing PID file during cleanup: {e}")
                
        atexit.register(cleanup_pid_file)
        
        # Register signal handlers
        def signal_handler(sig, frame):
            print(f"Received signal {sig}, shutting down {self.service_name}")
            
            # Clean up PID file
            cleanup_pid_file()
            
            # Exit
            sys.exit(0)
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def health_endpoint(self) -> Dict[str, Any]:
        """
        Health check endpoint for monitoring.
        
        Returns:
            dict: Health status information
        """
        uptime = (datetime.now() - self.startup_time).total_seconds()
        
        return {
            "status": "healthy" if self.startup_complete else "starting",
            "service": self.service_name,
            "pid": os.getpid(),
            "port": self.port,
            "uptime_seconds": uptime,
            "startup_time": self.startup_time.isoformat(),
            "singleton_ready": self.singleton_ready,
            "startup_complete": self.startup_complete
        }

def get_singleton_status() -> Dict[str, Any]:
    """
    Get information about FastAPI singleton processes.
    
    This function checks for running FastAPI singleton processes
    and returns information about them.
    
    Returns:
        dict: Dictionary with information about singleton processes
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "active_singletons": [],
        "errors": []
    }
    
    # Check if any process lock files exist
    try:
        lock_files = list(Path('.').glob(".*.lock"))
        debug_print(f"Found {len(lock_files)} process lock files")
        
        for lock_file in lock_files:
            try:
                with open(lock_file, 'r') as f:
                    # Use shared lock for reading
                    fcntl.flock(f, fcntl.LOCK_SH)
                    try:
                        try:
                            lock_data = json.load(f)
                        except json.JSONDecodeError:
                            # Not JSON
                            f.seek(0)
                            content = f.read().strip()
                            try:
                                pid = int(content)
                                lock_data = {"pid": pid}
                            except ValueError:
                                lock_data = {"raw_content": content[:100]}
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
                        
                # Check if process is running
                pid = lock_data.get("pid")
                if pid:
                    try:
                        process = psutil.Process(pid)
                        
                        # Check if zombie
                        if process.status() == psutil.STATUS_ZOMBIE:
                            debug_print(f"Found zombie process with PID {pid}, lock file: {lock_file}")
                            
                            # Add to result with zombie status
                            lock_data["status"] = "zombie"
                            lock_data["lock_file"] = str(lock_file)
                            result["active_singletons"].append(lock_data)
                        else:
                            # Process is running - get command line
                            try:
                                cmdline = " ".join(process.cmdline())
                                lock_data["cmdline"] = cmdline
                            except Exception:
                                lock_data["cmdline"] = "Could not get command line"
                                
                            # Add to result
                            lock_data["status"] = "running"
                            lock_data["lock_file"] = str(lock_file)
                            result["active_singletons"].append(lock_data)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        debug_print(f"Lock file {lock_file} points to non-existent process {pid}")
                        
                        # Add to result with not_running status
                        lock_data["status"] = "not_running"
                        lock_data["lock_file"] = str(lock_file)
                        result["active_singletons"].append(lock_data)
                else:
                    # No PID in lock file
                    lock_data["status"] = "invalid"
                    lock_data["lock_file"] = str(lock_file)
                    result["active_singletons"].append(lock_data)
            except Exception as e:
                error_msg = f"Error processing lock file {lock_file}: {e}"
                debug_print(error_msg)
                result["errors"].append(error_msg)
    except Exception as e:
        error_msg = f"Error scanning for process lock files: {e}"
        debug_print(error_msg)
        result["errors"].append(error_msg)
    
    return result

def cleanup_stale_singletons() -> Dict[str, Any]:
    """
    Clean up stale FastAPI singleton processes.
    
    This function removes lock files for processes that are no longer running.
    
    Returns:
        dict: Results of the cleanup operation
    """
    result = {
        "removed_lock_files": [],
        "errors": []
    }
    
    # Get singleton status
    status = get_singleton_status()
    
    # Clean up stale lock files
    for singleton in status["active_singletons"]:
        if singleton["status"] in ["not_running", "zombie", "invalid"]:
            lock_file = Path(singleton["lock_file"])
            
            try:
                if lock_file.exists():
                    lock_file.unlink()
                    result["removed_lock_files"].append(str(lock_file))
                    debug_print(f"Removed stale lock file {lock_file}")
            except Exception as e:
                error_msg = f"Error removing lock file {lock_file}: {e}"
                debug_print(error_msg)
                result["errors"].append(error_msg)
    
    return result

# Example usage in a FastAPI application
if __name__ == "__main__":
    import uvicorn
    
    # Create a singleton-managed FastAPI app
    app = SingletonFastAPI(
        port=8001,
        service_name="example_server",
        pid_file="./example_server.pid",
        title="Example Singleton FastAPI Server",
        description="A FastAPI server with singleton behavior"
    )
    
    @app.get("/")
    def read_root():
        return {"Hello": "World"}
        
    @app.get("/status")
    def read_status():
        return get_singleton_status()
        
    @app.get("/cleanup")
    def run_cleanup():
        return cleanup_stale_singletons()
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8001)