#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/process_lock.py

"""
Process Locking Module

This module provides robust service-level locking to prevent duplicate processes.
It handles common problems like:
1. Concurrent service startup
2. Port binding conflicts
3. Stale PID file detection
4. Process health verification

Usage:
    from process_lock import ServiceLock
    
    # For FastAPI Inference Server
    with ServiceLock("inference_server", port=8001) as lock:
        if lock.acquired:
            # Start the server securely knowing we're the only instance
            start_server(port=8001)
        else:
            # Another server is already running
            print(lock.error_message)
"""

import os
import sys
import json
import socket
import psutil
import signal
import time
import atexit
import traceback
import logging
import fcntl
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants
LOCKS_DIR = Path('./process_locks')
PID_FILES_DIR = Path('.')
LOCK_TIMEOUT = 3600  # 1 hour in seconds

# Ensure locks directory exists
LOCKS_DIR.mkdir(parents=True, exist_ok=True)

class ServiceLock:
    """
    Service-level locking mechanism to prevent duplicate processes.
    
    This class implements a robust file-based locking system with:
    - Port availability checking
    - Process status verification
    - Stale lock detection and cleanup
    - Automatic lock release on process exit
    """
    
    def __init__(
        self, 
        service_name: str, 
        port: Optional[int] = None,
        pid_file: Optional[Union[str, Path]] = None,
        timeout: int = LOCK_TIMEOUT,
        debug: bool = False
    ):
        """
        Initialize a service lock.
        
        Args:
            service_name: Name of the service (e.g., 'inference_server')
            port: Port the service will use (if applicable)
            pid_file: Custom PID file path (defaults to ./{service_name}.pid)
            timeout: Lock timeout in seconds
            debug: Enable debug logging
        """
        self.service_name = service_name
        self.port = port
        self.timeout = timeout
        self.debug = debug
        self.acquired = False
        self.error_message = None
        self.lock_file = None
        self.lock_fd = None
        self.pid = os.getpid()
        
        # Configure logger
        self.logger = logging.getLogger(f'service_lock.{service_name}')
        if debug:
            self.logger.setLevel(logging.DEBUG)
        
        # Determine lock file path
        if port:
            self.lock_file = LOCKS_DIR / f"{service_name}_{port}.lock"
        else:
            self.lock_file = LOCKS_DIR / f"{service_name}.lock"
        
        # Determine PID file path
        if pid_file:
            self.pid_file = Path(pid_file)
        else:
            self.pid_file = PID_FILES_DIR / f"{service_name}.pid"
        
        self.logger.debug(f"Initialized service lock for {service_name} (port: {port})")
        self.logger.debug(f"Lock file: {self.lock_file}")
        self.logger.debug(f"PID file: {self.pid_file}")
    
    def _check_port_available(self) -> bool:
        """
        Check if the port is available for binding.
        
        Returns:
            bool: True if port is available, False if it's in use
        """
        if not self.port:
            return True  # No port to check
        
        self.logger.debug(f"Checking if port {self.port} is available")
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                result = s.connect_ex(('127.0.0.1', self.port))
                port_available = result != 0
                
                if not port_available:
                    self.logger.debug(f"Port {self.port} is in use (return code: {result})")
                    self.error_message = f"Port {self.port} is already in use by another process"
                    
                    # Try to identify the process using this port
                    try:
                        for proc in psutil.process_iter(['pid', 'name', 'connections']):
                            for conn in proc.info.get('connections', []):
                                if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port') and conn.laddr.port == self.port:
                                    self.logger.debug(f"Port {self.port} is being used by process {proc.info['pid']} ({proc.info['name']})")
                                    self.error_message += f" (PID: {proc.info['pid']}, Name: {proc.info['name']})"
                                    break
                    except Exception as proc_err:
                        self.logger.debug(f"Error identifying process using port {self.port}: {proc_err}")
                else:
                    self.logger.debug(f"Port {self.port} is available")
                
                return port_available
        except Exception as e:
            self.logger.error(f"Error checking port {self.port}: {e}")
            self.error_message = f"Error checking port {self.port}: {e}"
            # If we can't check, assume it might be in use to be safe
            return False
    
    def _check_stale_lock(self) -> bool:
        """
        Check if existing lock file is stale.
        
        Returns:
            bool: True if lock is stale and was removed, False otherwise
        """
        if not self.lock_file.exists():
            return False  # No lock file, not stale
        
        self.logger.debug(f"Checking if lock file {self.lock_file} is stale")
        lock_stats = {
            "exists": True,
            "size": None,
            "mod_time": None,
            "content_preview": None,
            "has_exclusive_lock": False,
            "pid": None,
            "process_status": None,
            "lock_age": None,
            "errors": []
        }
        
        try:
            # Get basic file stats first for debugging
            try:
                stats = self.lock_file.stat()
                lock_stats["size"] = stats.st_size
                lock_stats["mod_time"] = datetime.fromtimestamp(stats.st_mtime).isoformat()
                self.logger.debug(f"Lock file stats: size={lock_stats['size']} bytes, modified={lock_stats['mod_time']}")
            except Exception as stat_err:
                self.logger.warning(f"Could not get lock file stats: {stat_err}")
                lock_stats["errors"].append(f"Stat error: {str(stat_err)}")
            
            # Check if file is valid
            if lock_stats.get("size", 0) == 0:
                self.logger.debug(f"Lock file {self.lock_file} is empty, removing it")
                try:
                    self.lock_file.unlink()
                except Exception as unlink_err:
                    self.logger.warning(f"Error removing empty lock file: {unlink_err}")
                    lock_stats["errors"].append(f"Empty file unlink error: {str(unlink_err)}")
                return True
            
            # Attempt to open and read the lock file
            with open(self.lock_file, 'r') as f:
                # Try to acquire a shared lock for reading
                try:
                    # Non-blocking shared lock
                    fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    
                    # If we got here, no exclusive lock is held
                    self.logger.debug(f"No exclusive lock held on {self.lock_file}")
                    
                    # Read the content
                    content = f.read().strip()
                    lock_stats["content_preview"] = content[:50] + ("..." if len(content) > 50 else "")
                    
                    # Try to parse content
                    data = None
                    pid = None
                    
                    # First try JSON format
                    try:
                        data = json.loads(content)
                        self.logger.debug(f"Lock file content is valid JSON: {data}")
                    except json.JSONDecodeError as json_err:
                        # Not JSON, try plain text
                        self.logger.debug(f"Lock file is not JSON: {json_err}")
                        lock_stats["errors"].append(f"JSON parse error: {str(json_err)}")
                        
                        try:
                            if content.isdigit() or (content.startswith('-') and content[1:].isdigit()):
                                pid = int(content)
                                data = {"pid": pid}
                                self.logger.debug(f"Lock file contains plain integer PID: {pid}")
                            else:
                                # Invalid content format
                                self.logger.debug(f"Lock file {self.lock_file} has invalid content format, removing it")
                                lock_stats["errors"].append(f"Invalid content format: {lock_stats['content_preview']}")
                                fcntl.flock(f, fcntl.LOCK_UN)  # Release shared lock
                                
                                try:
                                    self.lock_file.unlink()
                                except Exception as unlink_err:
                                    self.logger.warning(f"Error removing invalid lock file: {unlink_err}")
                                    lock_stats["errors"].append(f"Invalid content unlink error: {str(unlink_err)}")
                                return True
                        except (ValueError, TypeError) as val_err:
                            # Not an integer either
                            self.logger.debug(f"Lock file content is not an integer: {val_err}")
                            lock_stats["errors"].append(f"Integer parse error: {str(val_err)}")
                            fcntl.flock(f, fcntl.LOCK_UN)  # Release shared lock
                            
                            try:
                                self.lock_file.unlink()
                            except Exception as unlink_err:
                                self.logger.warning(f"Error removing invalid lock file: {unlink_err}")
                                lock_stats["errors"].append(f"Parse error unlink error: {str(unlink_err)}")
                            return True
                    
                    # Get PID from data
                    if data and "pid" in data:
                        pid = data["pid"]
                        lock_stats["pid"] = pid
                        
                        # Check timestamp if available
                        if "timestamp" in data:
                            try:
                                lock_time = datetime.fromisoformat(data["timestamp"])
                                lock_age = (datetime.now() - lock_time).total_seconds()
                                lock_stats["lock_age"] = lock_age
                                
                                # If lock is too old, consider it stale
                                if lock_age > self.timeout:
                                    self.logger.debug(f"Lock file is too old ({lock_age:.1f}s > {self.timeout}s), removing it")
                                    fcntl.flock(f, fcntl.LOCK_UN)  # Release shared lock
                                    
                                    try:
                                        self.lock_file.unlink()
                                    except Exception as unlink_err:
                                        self.logger.warning(f"Error removing stale lock file: {unlink_err}")
                                        lock_stats["errors"].append(f"Stale file unlink error: {str(unlink_err)}")
                                    return True
                            except (ValueError, TypeError) as time_err:
                                # Invalid timestamp, ignore it
                                self.logger.debug(f"Invalid timestamp in lock file: {data.get('timestamp')}")
                                lock_stats["errors"].append(f"Timestamp parse error: {str(time_err)}")
                        
                        # Check if process is running
                        try:
                            process = psutil.Process(pid)
                            
                            # Get detailed process info
                            try:
                                proc_name = process.name()
                                proc_cmdline = " ".join(process.cmdline())
                                proc_create_time = datetime.fromtimestamp(process.create_time()).isoformat()
                                
                                self.logger.debug(f"Process {pid} info: name={proc_name}, created={proc_create_time}")
                                self.logger.debug(f"Process {pid} command: {proc_cmdline[:100]}...")
                            except (psutil.AccessDenied, psutil.ZombieProcess) as proc_err:
                                self.logger.debug(f"Limited process info for PID {pid}: {proc_err}")
                            
                            # Check status
                            status = process.status()
                            lock_stats["process_status"] = status
                            
                            # Check if zombie
                            if status == psutil.STATUS_ZOMBIE:
                                self.logger.debug(f"Process {pid} is a zombie, lock is stale")
                                fcntl.flock(f, fcntl.LOCK_UN)  # Release shared lock
                                
                                try:
                                    self.lock_file.unlink()
                                except Exception as unlink_err:
                                    self.logger.warning(f"Error removing zombie process lock file: {unlink_err}")
                                    lock_stats["errors"].append(f"Zombie file unlink error: {str(unlink_err)}")
                                return True
                            
                            # Check if process responds to signals
                            try:
                                os.kill(pid, 0)  # Test signal, doesn't actually send anything
                                self.logger.debug(f"Process {pid} responds to signals")
                            except OSError as signal_err:
                                self.logger.debug(f"Process {pid} doesn't respond to signals: {signal_err}")
                                lock_stats["errors"].append(f"Signal test error: {str(signal_err)}")
                                # Continue anyway - process exists even if it doesn't respond to signals
                            
                            # Process is running - lock is not stale
                            self.logger.debug(f"Process {pid} is still running, lock is not stale")
                            fcntl.flock(f, fcntl.LOCK_UN)  # Release shared lock
                            
                            # Update error message
                            self.error_message = f"Service {self.service_name} is already running (PID: {pid})"
                            return False
                            
                        except psutil.NoSuchProcess:
                            # Process is not running, lock is stale
                            self.logger.debug(f"Process {pid} does not exist, lock is stale")
                            fcntl.flock(f, fcntl.LOCK_UN)  # Release shared lock
                            
                            try:
                                self.lock_file.unlink()
                            except Exception as unlink_err:
                                self.logger.warning(f"Error removing stale process lock file: {unlink_err}")
                                lock_stats["errors"].append(f"NoSuchProcess unlink error: {str(unlink_err)}")
                            return True
                            
                        except psutil.AccessDenied:
                            # Process exists but we can't access details
                            self.logger.warning(f"Access denied for process {pid}, assuming it's running")
                            fcntl.flock(f, fcntl.LOCK_UN)  # Release shared lock
                            
                            # Update error message
                            self.error_message = f"Service {self.service_name} appears to be running (PID: {pid}, access denied)"
                            return False
                            
                        except Exception as proc_err:
                            # Unknown error checking process
                            self.logger.error(f"Error checking process {pid}: {proc_err}")
                            self.logger.error(traceback.format_exc())
                            lock_stats["errors"].append(f"Process check error: {str(proc_err)}")
                            fcntl.flock(f, fcntl.LOCK_UN)  # Release shared lock
                            
                            # Assume it's not stale to be safe
                            self.error_message = f"Error checking lock for {self.service_name} ({proc_err})"
                            return False
                            
                    else:
                        # No PID in lock file, consider it stale
                        self.logger.debug(f"Lock file has no valid PID, removing it")
                        fcntl.flock(f, fcntl.LOCK_UN)  # Release shared lock
                        
                        try:
                            self.lock_file.unlink()
                        except Exception as unlink_err:
                            self.logger.warning(f"Error removing invalid PID lock file: {unlink_err}")
                            lock_stats["errors"].append(f"No PID unlink error: {str(unlink_err)}")
                        return True
                        
                except IOError as lock_err:
                    # Could not acquire shared lock - someone has an exclusive lock
                    # This means the lock is being held by another process
                    lock_stats["has_exclusive_lock"] = True
                    self.logger.debug(f"Lock file is actively held by another process: {lock_err}")
                    
                    # Try to read the file anyway to get some info about the lock holder
                    try:
                        # Rewind file pointer
                        f.seek(0)
                        peek_content = f.read().strip()
                        
                        try:
                            peek_data = json.loads(peek_content)
                            if "pid" in peek_data:
                                self.error_message = f"Service {self.service_name} is already running (PID: {peek_data['pid']})"
                            else:
                                self.error_message = f"Service {self.service_name} is already running (lock held)"
                        except Exception:
                            self.error_message = f"Service {self.service_name} is already running (lock held)"
                    except Exception as peek_err:
                        self.logger.debug(f"Could not peek at lock file contents: {peek_err}")
                        self.error_message = f"Service {self.service_name} is already running (lock held)"
                    
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error checking stale lock: {e}")
            self.logger.error(traceback.format_exc())
            lock_stats["errors"].append(f"General error: {str(e)}")
            
            # Log detailed diagnostic info
            self.logger.debug(f"Lock stats: {lock_stats}")
            
            # If we can't check, assume it's not stale to be safe
            self.error_message = f"Error checking lock for {self.service_name} ({e})"
            return False
            
        # This should never be reached
        self.logger.warning("Unexpected code path in _check_stale_lock")
        return False
    
    def _create_lock(self) -> bool:
        """
        Create a lock file and acquire an exclusive lock.
        
        Returns:
            bool: True if lock was acquired, False otherwise
        """
        # Ensure parent directory exists
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Open the lock file
            self.lock_fd = open(self.lock_file, 'w')
            
            # Try to acquire an exclusive lock
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except IOError:
                # Could not acquire lock - someone else has it
                self.logger.debug(f"Could not acquire lock, another process has it")
                self.error_message = f"Service {self.service_name} is already running (could not acquire lock)"
                self.lock_fd.close()
                self.lock_fd = None
                return False
            
            # Write lock information
            lock_data = {
                "pid": self.pid,
                "timestamp": datetime.now().isoformat(),
                "service_name": self.service_name,
                "hostname": socket.gethostname()
            }
            
            if self.port:
                lock_data["port"] = self.port
                
            json.dump(lock_data, self.lock_fd, indent=2)
            self.lock_fd.flush()
            
            # Register atexit handler to release lock
            atexit.register(self._release_lock)
            
            # Register signal handlers
            def signal_handler(sig, frame):
                self.logger.debug(f"Received signal {sig}, releasing lock")
                self._release_lock()
                sys.exit(0)
                
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            
            self.logger.debug(f"Successfully acquired lock for {self.service_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error creating lock: {e}")
            self.error_message = f"Error creating lock: {e}"
            
            if self.lock_fd:
                self.lock_fd.close()
                self.lock_fd = None
                
            return False
    
    def _write_pid_file(self) -> bool:
        """
        Write a PID file for this service.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create parent directory if needed
            self.pid_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine process command line for metadata
            cmdline = ' '.join(sys.argv)
            try:
                process = psutil.Process(self.pid)
                cmdline = ' '.join(process.cmdline())
            except Exception as proc_err:
                print(f"[DEBUG] Error getting process info for PID {self.pid}: {type(proc_err).__name__}: {proc_err}")
                pass
            
            # Create PID file data
            pid_data = {
                "pid": self.pid,
                "timestamp": datetime.now().isoformat(),
                "service_name": self.service_name,
                "command": cmdline,
                "hostname": socket.gethostname()
            }
            
            if self.port:
                pid_data["port"] = self.port
            
            # Write PID file
            with open(self.pid_file, 'w') as f:
                # Try to acquire an exclusive lock while writing
                try:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    json.dump(pid_data, f, indent=2)
                finally:
                    try:
                        fcntl.flock(f, fcntl.LOCK_UN)
                    except Exception as unlock_err:
                        print(f"[DEBUG] Error unlocking PID file: {type(unlock_err).__name__}: {unlock_err}")
                        pass
            
            # Register cleanup handler
            atexit.register(self._cleanup_pid_file)
            
            self.logger.debug(f"Successfully wrote PID file {self.pid_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error writing PID file: {e}")
            return False
    
    def _release_lock(self):
        """Release the lock if we're holding it."""
        if self.lock_fd:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                self.lock_fd.close()
                self.lock_fd = None
                
                # Remove the lock file
                if self.lock_file.exists():
                    self.lock_file.unlink()
                    
                self.logger.debug(f"Released lock for {self.service_name}")
                self.acquired = False
            except Exception as e:
                self.logger.error(f"Error releasing lock: {e}")
    
    def _cleanup_pid_file(self):
        """Clean up PID file on exit."""
        if self.pid_file.exists():
            try:
                # Read the PID file to make sure it's ours
                with open(self.pid_file, 'r') as f:
                    try:
                        data = json.load(f)
                        if data.get("pid") == self.pid:
                            # It's our PID file, remove it
                            self.pid_file.unlink()
                            self.logger.debug(f"Removed PID file {self.pid_file}")
                    except Exception:
                        # Can't read it, assume it's ours
                        self.pid_file.unlink()
                        self.logger.debug(f"Removed unreadable PID file {self.pid_file}")
            except Exception as e:
                self.logger.error(f"Error cleaning up PID file: {e}")
    
    def acquire(self) -> bool:
        """
        Acquire the service lock.
        
        Returns:
            bool: True if lock was acquired, False otherwise
        """
        self.logger.debug(f"Attempting to acquire lock for {self.service_name}")
        
        # Check if port is available (if applicable)
        if self.port and not self._check_port_available():
            self.logger.info(f"Could not acquire lock: {self.error_message}")
            return False
        
        # Check for stale lock file
        self._check_stale_lock()
        
        # Attempt to create lock
        if not self._create_lock():
            self.logger.info(f"Could not acquire lock: {self.error_message}")
            return False
        
        # Write PID file
        if not self._write_pid_file():
            self.logger.warning(f"Acquired lock but failed to write PID file")
            # Not a critical failure, we can continue
        
        # Lock acquired successfully
        self.acquired = True
        self.logger.info(f"Successfully acquired lock for {self.service_name}")
        return True
    
    def release(self):
        """Release the service lock."""
        if self.acquired:
            self._release_lock()
            # PID file will be cleaned up by atexit handler
            self.logger.info(f"Released lock for {self.service_name}")
    
    def __enter__(self):
        """Context manager enter."""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False  # Don't suppress exceptions

@contextmanager
def single_instance(service_name, port=None, pid_file=None):
    """
    Context manager ensuring only one instance of a service runs.
    
    Args:
        service_name: Name of the service
        port: Port number (if applicable)
        pid_file: Custom PID file path (if needed)
    
    Yields:
        ServiceLock: The service lock object
    
    Raises:
        RuntimeError: If another instance is already running
    """
    with ServiceLock(service_name, port, pid_file) as lock:
        if not lock.acquired:
            raise RuntimeError(lock.error_message)
        yield lock

def check_running_services():
    """
    Check for running services managed by this module.
    
    Returns:
        dict: Information about running services
    """
    result = {
        "running_services": [],
        "stale_locks": [],
        "errors": []
    }
    
    # Check all lock files
    if LOCKS_DIR.exists():
        for lock_file in LOCKS_DIR.glob("*.lock"):
            try:
                with open(lock_file, 'r') as f:
                    try:
                        fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
                        # If we got the lock, it's not held by another process
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            # Try to read as plain text
                            f.seek(0)
                            content = f.read().strip()
                            try:
                                pid = int(content)
                                data = {"pid": pid}
                            except ValueError:
                                # Invalid content
                                fcntl.flock(f, fcntl.LOCK_UN)
                                result["stale_locks"].append({
                                    "lock_file": str(lock_file),
                                    "reason": "invalid_content"
                                })
                                continue
                        
                        # Check if process is running
                        if "pid" in data:
                            pid = data["pid"]
                            try:
                                process = psutil.Process(pid)
                                if process.status() == psutil.STATUS_ZOMBIE:
                                    # Zombie process
                                    fcntl.flock(f, fcntl.LOCK_UN)
                                    result["stale_locks"].append({
                                        "lock_file": str(lock_file),
                                        "pid": pid,
                                        "reason": "zombie_process"
                                    })
                                else:
                                    # Process is running but lock is not held
                                    fcntl.flock(f, fcntl.LOCK_UN)
                                    result["stale_locks"].append({
                                        "lock_file": str(lock_file),
                                        "pid": pid,
                                        "reason": "process_running_lock_not_held"
                                    })
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                # Process not running
                                fcntl.flock(f, fcntl.LOCK_UN)
                                result["stale_locks"].append({
                                    "lock_file": str(lock_file),
                                    "pid": pid,
                                    "reason": "process_not_running"
                                })
                        else:
                            # No PID in lock file
                            fcntl.flock(f, fcntl.LOCK_UN)
                            result["stale_locks"].append({
                                "lock_file": str(lock_file),
                                "reason": "no_pid"
                            })
                    except IOError:
                        # Could not acquire shared lock - someone has an exclusive lock
                        service_name = lock_file.stem
                        # Try to extract port from filename (e.g., service_name_port.lock)
                        port = None
                        parts = service_name.split('_')
                        if len(parts) > 1 and parts[-1].isdigit():
                            port = int(parts[-1])
                            service_name = '_'.join(parts[:-1])
                        
                        # Try to get PID from lock file content
                        try:
                            # Just peek at the content without locking
                            f.seek(0)
                            content = f.read()
                            try:
                                data = json.loads(content)
                                pid = data.get("pid")
                            except Exception:
                                # Can't parse content
                                pid = None
                        except Exception:
                            pid = None
                        
                        result["running_services"].append({
                            "service_name": service_name,
                            "lock_file": str(lock_file),
                            "port": port,
                            "pid": pid
                        })
            except Exception as e:
                result["errors"].append({
                    "lock_file": str(lock_file),
                    "error": str(e)
                })
    
    return result

def cleanup_stale_locks():
    """
    Clean up stale locks.
    
    Returns:
        dict: Results of the cleanup operation
    """
    result = {
        "removed_locks": [],
        "errors": []
    }
    
    # Check for stale locks
    service_check = check_running_services()
    
    # Remove stale locks
    for lock in service_check["stale_locks"]:
        lock_file = Path(lock["lock_file"])
        try:
            lock_file.unlink()
            result["removed_locks"].append(lock["lock_file"])
        except Exception as e:
            result["errors"].append({
                "lock_file": lock["lock_file"],
                "error": str(e)
            })
    
    return result

if __name__ == "__main__":
    # Command-line interface for checking and cleaning up locks
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Lock Utility")
    parser.add_argument("--check", action="store_true", help="Check for running services and stale locks")
    parser.add_argument("--cleanup", action="store_true", help="Clean up stale locks")
    parser.add_argument("--service", help="Start a service with lock (for testing)")
    parser.add_argument("--port", type=int, help="Port to use with service")
    parser.add_argument("--sleep", type=int, default=60, help="Seconds to sleep when starting a service (for testing)")
    
    args = parser.parse_args()
    
    if args.check:
        print("Checking for running services and stale locks...")
        check_result = check_running_services()
        
        if check_result["running_services"]:
            print("\nRunning services:")
            for service in check_result["running_services"]:
                port_str = f" (port: {service['port']})" if service['port'] else ""
                pid_str = f" (PID: {service['pid']})" if service['pid'] else ""
                print(f"- {service['service_name']}{port_str}{pid_str}")
        else:
            print("No running services found.")
        
        if check_result["stale_locks"]:
            print("\nStale locks:")
            for lock in check_result["stale_locks"]:
                reason = lock["reason"].replace("_", " ")
                pid_str = f" (PID: {lock['pid']})" if "pid" in lock else ""
                print(f"- {lock['lock_file']} - {reason}{pid_str}")
        else:
            print("No stale locks found.")
        
        if check_result["errors"]:
            print("\nErrors:")
            for error in check_result["errors"]:
                print(f"- {error['lock_file']}: {error['error']}")
    
    elif args.cleanup:
        print("Cleaning up stale locks...")
        cleanup_result = cleanup_stale_locks()
        
        if cleanup_result["removed_locks"]:
            print("\nRemoved locks:")
            for lock in cleanup_result["removed_locks"]:
                print(f"- {lock}")
        else:
            print("No locks were removed.")
        
        if cleanup_result["errors"]:
            print("\nErrors:")
            for error in cleanup_result["errors"]:
                print(f"- {error['lock_file']}: {error['error']}")
    
    elif args.service:
        print(f"Starting service {args.service} with lock...")
        
        try:
            with single_instance(args.service, args.port) as lock:
                print(f"Acquired lock for {args.service}")
                print(f"PID: {os.getpid()}")
                
                if args.port:
                    print(f"Port: {args.port}")
                
                print(f"Sleeping for {args.sleep} seconds...")
                time.sleep(args.sleep)
                
                print("Releasing lock...")
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()