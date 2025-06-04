#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/port_conflict_resolver.py

"""
Port Conflict Resolution Utility

This module provides a comprehensive solution for the port conflicts observed
in the LoRA Training Pipeline, specifically:

1. Multiple FastAPI instances binding to the same port (8001)
2. Missing is_port_in_use function causing NameError
3. PID file format mismatch and validation errors
4. Properly validating and managing service processes
5. Preventing duplicate service instances

Usage:
    # To check for port conflicts
    python -m src.lora_training_pipeline.utils.port_conflict_resolver --check
    
    # To resolve port conflicts
    python -m src.lora_training_pipeline.utils.port_conflict_resolver --resolve
    
    # To resolve a specific service type
    python -m src.lora_training_pipeline.utils.port_conflict_resolver --resolve-service fastapi
"""

import os
import sys
import json
import time
import socket
import psutil
import traceback
import fcntl
import signal
import atexit
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('port_conflict.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('port_conflict_resolver')

# Constants
INFERENCE_PROCESS_PID_FILE = Path('./inference_process.pid')
DATA_COLLECTION_PID_FILE = Path('./data_collection_ui.pid')
INFERENCE_UI_PID_FILE = Path('./inference_ui.pid')
PROCESS_LOCK_DIR = Path('./process_locks')
MAX_PORT_CHECK_RETRIES = 3
PORT_CHECK_RETRY_DELAY = 0.5  # seconds
PROCESS_TERMINATION_TIMEOUT = 5.0  # seconds
PORT_CHECK_TIMEOUT = 2.0  # seconds

# Ensure lock directory exists
PROCESS_LOCK_DIR.mkdir(parents=True, exist_ok=True)

def is_port_in_use(
    port: int, 
    host: str = "localhost", 
    timeout: float = PORT_CHECK_TIMEOUT, 
    max_retries: int = MAX_PORT_CHECK_RETRIES
) -> bool:
    """
    Check if a port is already in use.
    
    Args:
        port: Port number to check
        host: Host to check (default: localhost)
        timeout: Timeout for socket connection
        max_retries: Number of connection attempts to make
        
    Returns:
        bool: True if port is in use, False otherwise
        
    Raises:
        ValueError: If port is not a valid port number
    """
    # Validate port parameter
    try:
        port = int(port)
        if port < 1 or port > 65535:
            logger.error(f"Invalid port number: {port} (must be 1-65535)")
            raise ValueError(f"Invalid port number: {port} (must be 1-65535)")
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid port parameter: {e}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Invalid port parameter: {e}")
    
    logger.debug(f"Checking if port {port} is in use on {host}... (retries={max_retries})")
    
    # Try multiple connection attempts for reliability
    for attempt in range(max_retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                
                # Try to connect - if connection succeeds, port is in use
                result = s.connect_ex((host, port))
                port_in_use = result == 0
                
                # Log detailed result code for debugging
                logger.debug(f"Socket connect_ex return code: {result} (attempt {attempt+1}/{max_retries})")
                
                if port_in_use:
                    logger.debug(f"Port {port} is IN USE on {host}")
                    
                    # Try to identify process using the port
                    try:
                        for proc in psutil.process_iter(['pid', 'name', 'connections']):
                            try:
                                # Check each connection for this process
                                for conn in proc.info.get('connections', []):
                                    if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port') and conn.laddr.port == port:
                                        logger.info(f"Port {port} is being used by: PID {proc.info['pid']} ({proc.info['name']})")
                                        break
                            except (psutil.AccessDenied, psutil.ZombieProcess):
                                # Skip processes we can't access
                                continue
                    except Exception as proc_err:
                        logger.debug(f"Could not identify process using port {port}: {proc_err}")
                        
                    return True
                else:
                    logger.debug(f"Port {port} is AVAILABLE on {host}")
                    
                    # For better reliability, double-check with a second attempt after a short delay
                    if attempt < max_retries - 1:
                        logger.debug(f"Waiting {PORT_CHECK_RETRY_DELAY}s before verification attempt...")
                        time.sleep(PORT_CHECK_RETRY_DELAY)
                        continue
                    else:
                        # If we reach the last attempt and it's still available, confirm it's free
                        return False
                        
        except socket.error as e:
            logger.warning(f"Socket error checking port {port} on {host} (attempt {attempt+1}/{max_retries}): {e}")
            
            # Only retry if we have attempts left
            if attempt < max_retries - 1:
                logger.debug(f"Retrying port check in {PORT_CHECK_RETRY_DELAY}s...")
                time.sleep(PORT_CHECK_RETRY_DELAY)
            else:
                logger.error(f"All port check attempts failed for port {port}")
                # If we can't check reliably, assume it might be in use to be safe
                return True
                
        except Exception as e:
            logger.error(f"Unexpected error checking port {port}: {e}")
            logger.error(traceback.format_exc())
            # If we can't check reliably, assume it might be in use to be safe
            return True
    
    # This should never be reached due to the return in the loop,
    # but just in case, assume port is in use (safer default)
    logger.warning(f"Port check code reached unexpected path for port {port}")
    return True

def check_process_running(pid: int) -> bool:
    """
    Check if a process with the given PID is running.
    
    Args:
        pid: Process ID to check
        
    Returns:
        bool: True if process is running and responsive, False otherwise
        
    Raises:
        ValueError: If pid is not a valid process ID
    """
    # Validate pid parameter
    try:
        pid = int(pid)
        if pid <= 0:
            logger.error(f"Invalid process ID: {pid} (must be positive)")
            raise ValueError(f"Invalid process ID: {pid} (must be positive)")
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid process ID parameter: {e}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Invalid process ID parameter: {e}")
    
    logger.debug(f"Checking if process {pid} is running...")
    
    try:
        # First check if process exists
        process = psutil.Process(pid)
        
        # Get more detailed process information for debugging
        try:
            process_name = process.name()
            cmd_line = " ".join(process.cmdline())
            create_time = datetime.fromtimestamp(process.create_time()).strftime('%Y-%m-%d %H:%M:%S')
            logger.debug(f"Process {pid} info: name={process_name}, created={create_time}")
            logger.debug(f"Process {pid} cmdline: {cmd_line[:100]}...")
        except (psutil.AccessDenied, psutil.ZombieProcess) as e:
            logger.debug(f"Could not get detailed info for process {pid}: {e}")
        
        # Check process status
        status = process.status()
        logger.debug(f"Process {pid} status: {status}")
        
        # Check if zombie
        if status == psutil.STATUS_ZOMBIE:
            logger.info(f"Process {pid} is a zombie - considered not running")
            return False
        
        # Check if process is responsive
        try:
            # Try to send signal 0 to check if process is responsive
            os.kill(pid, 0)
            logger.debug(f"Process {pid} responds to signals")
        except OSError as e:
            logger.warning(f"Process {pid} exists but does not respond to signals: {e}")
            # Consider it running anyway since the process exists
        
        # Process exists and is not a zombie
        logger.info(f"Process {pid} is running (status: {status})")
        return True
        
    except psutil.NoSuchProcess:
        logger.info(f"Process {pid} does not exist")
        return False
    except psutil.AccessDenied:
        # If access is denied, the process exists but we can't get info
        logger.warning(f"Access denied for process {pid}, assuming it's running")
        return True
    except psutil.ZombieProcess:
        logger.info(f"Process {pid} is a zombie process")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking process {pid}: {e}")
        logger.error(traceback.format_exc())
        # If we can't determine, assume it's not running (safer to allow starting a new process)
        return False

def read_pid_file(pid_file: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Read a PID file, supporting both JSON and integer formats.
    
    Args:
        pid_file: Path to the PID file
        
    Returns:
        dict: Dictionary with pid and metadata, or None if file doesn't exist/read fails
    """
    # Track file state for debugging
    file_stats = {
        "exists": False,
        "size_bytes": None,
        "mod_time": None,
        "permissions": None,
        "is_file": False,
        "attempted_encodings": [],
        "successful_encoding": None,
        "content_preview": None,
        "parse_method": None,
        "errors": []
    }

    # First ensure pid_file is a Path object
    try:
        pid_file = Path(pid_file)
        logger.debug(f"Reading PID file: {pid_file.absolute()}")
        
        # Check if file exists
        file_stats["exists"] = pid_file.exists()
        
        if not file_stats["exists"]:
            logger.debug(f"PID file {pid_file} does not exist")
            return None
        
        # Get file stats for debugging
        try:
            stats = pid_file.stat()
            file_stats["size_bytes"] = stats.st_size
            file_stats["mod_time"] = datetime.fromtimestamp(stats.st_mtime).isoformat()
            file_stats["is_file"] = pid_file.is_file()
            file_stats["permissions"] = oct(stats.st_mode)[-3:]
            
            logger.debug(f"PID file stats: size={file_stats['size_bytes']} bytes, "
                      f"modified={file_stats['mod_time']}, "
                      f"permissions={file_stats['permissions']}")
                      
            # Check if file is empty
            if file_stats["size_bytes"] == 0:
                logger.warning(f"PID file {pid_file} is empty")
                file_stats["errors"].append("Empty file")
                return None
                
            # Check if file is too large (likely not a PID file)
            if file_stats["size_bytes"] > 10000:
                logger.warning(f"PID file {pid_file} is suspiciously large ({file_stats['size_bytes']} bytes)")
                # Continue anyway but log the warning
        except Exception as stat_err:
            logger.warning(f"Could not get file stats for {pid_file}: {stat_err}")
            file_stats["errors"].append(f"Stat error: {str(stat_err)}")
            
        # Check if path is actually a file
        if not file_stats["is_file"]:
            logger.error(f"Path {pid_file} exists but is not a file")
            file_stats["errors"].append("Not a file")
            return None
        
        # Check if file is readable
        if not os.access(pid_file, os.R_OK):
            logger.error(f"PID file {pid_file} is not readable (permissions: {file_stats['permissions']})")
            file_stats["errors"].append("Not readable")
            return None
            
    except Exception as e:
        logger.error(f"Invalid PID file path: {e}")
        logger.error(traceback.format_exc())
        return None
    
    # Try multiple encodings if needed
    encodings = ['utf-8', 'latin-1', 'ascii']
    content = None
    
    for encoding in encodings:
        file_stats["attempted_encodings"].append(encoding)
        try:
            # Use a context manager with error handling
            with open(pid_file, 'r', encoding=encoding) as f:
                # Use file locking if possible
                try:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    content = f.read().strip()
                    file_stats["successful_encoding"] = encoding
                    
                    # Store content preview for debugging
                    if content:
                        file_stats["content_preview"] = content[:50] + ("..." if len(content) > 50 else "")
                        
                except IOError as lock_err:
                    logger.warning(f"Could not acquire shared lock on {pid_file}: {lock_err}")
                    # Try to read without locking
                    content = f.read().strip()
                    file_stats["successful_encoding"] = encoding
                    file_stats["errors"].append(f"Lock error: {str(lock_err)}")
                finally:
                    try:
                        fcntl.flock(f, fcntl.LOCK_UN)
                    except (IOError, OSError) as unlock_err:
                        logger.debug(f"[DEBUG] File unlock error (ignoring): {unlock_err}")
                        pass  # Ignore unlock errors
            
            logger.debug(f"Successfully read PID file with {encoding} encoding")
            break  # Break if successful
            
        except UnicodeDecodeError:
            logger.debug(f"Failed to read with {encoding} encoding, trying next...")
            continue
            
        except PermissionError as perm_err:
            logger.error(f"Permission error reading {pid_file}: {perm_err}")
            file_stats["errors"].append(f"Permission error: {str(perm_err)}")
            return None
            
        except Exception as e:
            logger.error(f"Error reading PID file {pid_file} with {encoding} encoding: {e}")
            logger.error(traceback.format_exc())
            file_stats["errors"].append(f"Read error ({encoding}): {str(e)}")
            return None
            
    if content is None:
        logger.error(f"Could not read PID file with any encoding: {pid_file}")
        logger.debug(f"File stats: {file_stats}")
        return None
            
    if not content:
        logger.warning(f"PID file {pid_file} is empty after stripping whitespace")
        return None
        
    # Try JSON format first
    try:
        data = json.loads(content)
        file_stats["parse_method"] = "json"
        
        # Log raw data at debug level
        logger.debug(f"Raw JSON data: {json.dumps(data)}")
        
        # Validate "pid" field
        if "pid" in data:
            try:
                pid = int(data["pid"])
                if pid <= 0:
                    logger.warning(f"Invalid PID value in JSON: {pid}")
                    file_stats["errors"].append(f"Invalid PID value: {pid}")
                    return None
                    
                # Log all data for debugging
                logger.debug(f"Valid JSON PID file: {data}")
                
                # Extract port if present
                port = None
                if "port" in data:
                    try:
                        port = int(data["port"])
                        if port < 1 or port > 65535:
                            logger.warning(f"Invalid port value in JSON: {port}")
                            # Don't store invalid port value
                            port = None
                            file_stats["errors"].append(f"Invalid port value: {data['port']}")
                    except (TypeError, ValueError) as port_err:
                        logger.warning(f"Non-integer port in JSON: {data['port']}")
                        port = None
                        file_stats["errors"].append(f"Port parse error: {str(port_err)}")
                
                # Check for suspicious PID values
                if pid > 4194304:  # Common upper limit in most systems
                    logger.warning(f"Suspiciously large PID value in file: {pid}")
                    # Continue anyway but log the warning
                        
                result = {
                    "pid": pid,
                    "port": port,
                    "format": "json",
                    "timestamp": data.get("timestamp"),
                    "updated_by": data.get("updated_by"),
                    "service_name": data.get("service_name"),
                    "raw_data": data,
                    "_file_stats": file_stats  # Include debugging info
                }
                
                logger.debug(f"Successfully parsed JSON PID file with PID {pid}"
                          f"{' and port ' + str(port) if port else ''}")
                          
                return result
                
            except (TypeError, ValueError) as pid_err:
                logger.warning(f"Non-integer PID in JSON: {data.get('pid', 'missing')}")
                file_stats["errors"].append(f"PID parse error: {str(pid_err)}")
                return None
        else:
            logger.warning(f"JSON missing 'pid' field: {data}")
            file_stats["errors"].append("Missing 'pid' field in JSON")
            return None
            
    except json.JSONDecodeError as json_err:
        logger.debug(f"Not a JSON file: {json_err}")
        file_stats["errors"].append(f"JSON decode error: {str(json_err)}")
        # Continue with integer format attempt
        
        # Try simple integer format
        try:
            # First check if content looks like an integer
            if not content.strip().isdigit() and not (content.strip().startswith('-') and content.strip()[1:].isdigit()):
                # Not likely to be an integer, log and return
                safe_content = content[:50] + ("..." if len(content) > 50 else "")
                logger.warning(f"PID file content doesn't look like an integer: {safe_content}")
                file_stats["errors"].append(f"Not integer-like content: {safe_content}")
                return None
                
            # Try converting to integer
            pid = int(content)
            file_stats["parse_method"] = "integer"
            
            if pid <= 0:
                logger.warning(f"Invalid PID value: {pid}")
                file_stats["errors"].append(f"Invalid integer PID: {pid}")
                return None
                
            # Check for suspicious PID values
            if pid > 4194304:  # Common upper limit in most systems
                logger.warning(f"Suspiciously large PID value in file: {pid}")
                # Continue but log the warning
                
            logger.info(f"Valid integer PID file with PID: {pid}")
            
            result = {
                "pid": pid,
                "format": "integer",
                "timestamp": datetime.now().isoformat(),
                "converted": True,
                "_file_stats": file_stats  # Include debugging info
            }
            
            # Run additional check to verify this PID exists
            try:
                if not psutil.pid_exists(pid):
                    logger.warning(f"PID {pid} from file does not exist")
                    # Return the data anyway, let caller decide what to do
            except Exception as proc_err:
                logger.debug(f"Error checking if PID {pid} exists: {proc_err}")
                
            return result
            
        except ValueError as val_err:
            # Log first part of content for debugging
            safe_content = content[:50] + ("..." if len(content) > 50 else "")
            logger.warning(f"Invalid PID file format (not JSON or integer): {safe_content}")
            logger.debug(f"ValueError details: {val_err}")
            file_stats["errors"].append(f"Integer parse error: {str(val_err)}")
            
            # Log full file stats for debugging
            logger.debug(f"File stats: {file_stats}")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error parsing PID file {pid_file}: {e}")
        logger.error(traceback.format_exc())
        file_stats["errors"].append(f"Unexpected error: {str(e)}")
        
        # Log full file stats for debugging
        logger.debug(f"File stats: {file_stats}")
        return None

def write_pid_file(pid_file: Union[str, Path], pid: int, port: Optional[int] = None, service_type: Optional[str] = None) -> bool:
    """
    Write a PID file in standardized JSON format.
    
    Args:
        pid_file: Path to the PID file
        pid: Process ID to write
        port: Port number (optional)
        service_type: Type of service (optional)
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Validate parameters
    try:
        # Validate pid_file
        if not isinstance(pid_file, Path):
            pid_file = Path(pid_file)
            
        # Validate pid
        pid = int(pid)
        if pid <= 0:
            logger.error(f"Invalid process ID: {pid} (must be positive)")
            return False
            
        # Validate port if provided
        if port is not None:
            port = int(port)
            if port < 1 or port > 65535:
                logger.warning(f"Invalid port number: {port} (must be 1-65535)")
                # Don't include invalid port
                port = None
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid parameter for write_pid_file: {e}")
        logger.error(traceback.format_exc())
        return False
        
    try:
        # Ensure parent directory exists
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create proper JSON data
        data = {
            "pid": pid,
            "timestamp": datetime.now().isoformat(),
            "updated_by": "port_conflict_resolver"
        }
        
        if port is not None:
            data["port"] = port
            
        if service_type:
            data["service_type"] = service_type
            
        # Create temporary file first to ensure atomic write
        temp_file = pid_file.with_suffix('.tmp')
        
        # Write to temporary file
        with open(temp_file, 'w', encoding='utf-8') as f:
            # Use file locking to ensure exclusive write
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
            
        # Rename temporary file to target file (atomic operation)
        temp_file.replace(pid_file)
            
        logger.info(f"Successfully wrote PID file {pid_file} with PID {pid}" + 
                  (f" on port {port}" if port is not None else ""))
        return True
        
    except (IOError, OSError) as e:
        logger.error(f"I/O error writing PID file {pid_file}: {e}")
        logger.error(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"Unexpected error writing PID file {pid_file}: {e}")
        logger.error(traceback.format_exc())
        return False

def get_process_info_by_port(port: int) -> Optional[Dict[str, Any]]:
    """
    Get information about the process using a specific port.
    
    Args:
        port: Port number to check
        
    Returns:
        dict: Process information or None if no process is using the port
    """
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        # Found a process using this port
                        cmdline = " ".join(proc.cmdline())
                        creation_time = datetime.fromtimestamp(proc.create_time())
                        
                        # Try to determine the service type from command line
                        service_type = "unknown"
                        if "fastapi_inference" in cmdline or "inference_server" in cmdline:
                            service_type = "fastapi"
                        elif "gradio" in cmdline and "inference" in cmdline:
                            service_type = "inference_ui"
                        elif "gradio" in cmdline and "data_collection" in cmdline:
                            service_type = "data_collection"
                            
                        return {
                            "pid": proc.pid,
                            "name": proc.name(),
                            "cmdline": cmdline,
                            "port": port,
                            "create_time": creation_time.isoformat(),
                            "create_time_seconds": proc.create_time(),
                            "service_type": service_type
                        }
            except (psutil.AccessDenied, psutil.ZombieProcess):
                # Skip processes we can't access
                continue
    except Exception as e:
        logger.error(f"Error getting process info for port {port}: {e}")
        logger.error(traceback.format_exc())
        
    return None

def get_service_processes(service_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get information about all service processes of a specific type.
    
    Args:
        service_type: Type of service to filter for (optional)
        
    Returns:
        list: List of dictionaries with process information
    """
    processes = []
    
    # Define patterns to identify service types from command line
    service_patterns = {
        "fastapi": ["fastapi_inference", "uvicorn", "port=8001", "port 8001"],
        "inference_ui": ["gradio_inference", "GRADIO_PORT=7861", "port=7861", "port 7861", "inference.py"],
        "data_collection": ["gradio_app", "GRADIO_PORT=7862", "port=7862", "port 7862", "data_collection"]
    }
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'status']):
            try:
                # Skip non-Python processes
                if not proc.name().lower().startswith("python"):
                    continue
                    
                cmdline = " ".join(proc.cmdline())
                detected_type = None
                
                # Check for each service type
                for svc_type, patterns in service_patterns.items():
                    # Check if command line matches any of the patterns for this service type
                    if any(pattern.lower() in cmdline.lower() for pattern in patterns):
                        detected_type = svc_type
                        break
                        
                # Skip if no service type detected or if filtering by type and not matching
                if detected_type is None or (service_type and detected_type != service_type):
                    continue
                    
                # Get process information
                ports = []
                try:
                    # Try to get port from connections
                    for conn in proc.connections(kind='inet'):
                        if conn.status == 'LISTEN' and conn.laddr.port not in ports:
                            ports.append(conn.laddr.port)
                except (psutil.AccessDenied, psutil.ZombieProcess) as proc_err:
                    print(f"[DEBUG] Process access issue when checking connections for PID {proc.pid}: {type(proc_err).__name__}")
                    pass
                    
                # Try to extract port from command line if not found in connections
                if not ports:
                    if "GRADIO_PORT=" in cmdline:
                        port_part = cmdline.split("GRADIO_PORT=")[1].split()[0]
                        try:
                            ports.append(int(port_part))
                        except ValueError:
                            pass
                    elif "port=" in cmdline.lower():
                        port_part = cmdline.lower().split("port=")[1].split()[0]
                        try:
                            ports.append(int(port_part))
                        except ValueError:
                            pass
                    elif "--port" in cmdline:
                        parts = cmdline.split("--port")
                        if len(parts) > 1:
                            port_part = parts[1].strip().split()[0]
                            try:
                                ports.append(int(port_part))
                            except ValueError:
                                pass
                                
                # Default ports if none found
                if not ports:
                    if detected_type == "fastapi":
                        ports.append(8001)
                    elif detected_type == "inference_ui":
                        ports.append(7861)
                    elif detected_type == "data_collection":
                        ports.append(7862)
                
                # Check if process is alive and not a zombie
                is_alive = proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
                
                # Check if each port is actually bound
                bound_ports = []
                for port in ports:
                    if is_port_in_use(port):
                        bound_ports.append(port)
                
                processes.append({
                    "pid": proc.pid,
                    "name": proc.name(),
                    "cmdline": cmdline,
                    "ports": ports,
                    "bound_ports": bound_ports,
                    "create_time": datetime.fromtimestamp(proc.create_time()).isoformat(),
                    "create_time_seconds": proc.create_time(),
                    "service_type": detected_type,
                    "status": proc.status(),
                    "is_alive": is_alive
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Skip processes we can't access
                continue
    except Exception as e:
        logger.error(f"Error enumerating service processes: {e}")
        logger.error(traceback.format_exc())
    
    return processes

def acquire_service_lock(service_type: str, port: int) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Acquire a lock for a service, ensuring only one instance can run.
    
    Args:
        service_type: Type of service
        port: Port number
        
    Returns:
        tuple: (success, pid, error_message)
    """
    lock_file = PROCESS_LOCK_DIR / f"{service_type}_{port}.lock"
    
    try:
        # Check if port is already in use
        if is_port_in_use(port):
            # Get the process using this port
            process_info = get_process_info_by_port(port)
            if process_info:
                return False, process_info["pid"], f"Port {port} is already in use by PID {process_info['pid']}"
            else:
                return False, None, f"Port {port} is already in use by an unknown process"
                
        # Try to create lock file directory
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Open the lock file
        lock_fd = open(lock_file, 'w')
        
        # Try to acquire an exclusive lock
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            lock_fd.close()
            return False, None, f"Another process holds the lock for {service_type} on port {port}"
            
        # Write lock information
        pid = os.getpid()
        lock_data = {
            "pid": pid,
            "port": port,
            "service_type": service_type,
            "timestamp": datetime.now().isoformat(),
            "hostname": socket.gethostname()
        }
        
        json.dump(lock_data, lock_fd, indent=2)
        lock_fd.flush()
        
        # Register cleanup handler
        def cleanup_lock():
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
                if lock_file.exists():
                    lock_file.unlink()
            except Exception as e:
                logger.error(f"Error cleaning up lock: {e}")
                
        atexit.register(cleanup_lock)
        
        # Register signal handlers
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, releasing lock")
            cleanup_lock()
            sys.exit(0)
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        return True, pid, None
    except Exception as e:
        logger.error(f"Error acquiring service lock: {e}")
        logger.error(traceback.format_exc())
        return False, None, f"Error acquiring lock: {e}"

def check_for_port_conflicts() -> Dict[str, Any]:
    """
    Check for port conflicts in the application.
    
    Returns:
        dict: Dictionary with conflict information
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "conflicts": [],
        "port_status": {},
        "pid_file_status": {},
        "recommendations": []
    }
    
    # Dictionary of service ports to check
    service_ports = {
        "fastapi": 8001,
        "inference_ui": 7861,
        "data_collection": 7862
    }
    
    # Dictionary of service PID files
    service_pid_files = {
        "fastapi": INFERENCE_PROCESS_PID_FILE,
        "inference_ui": INFERENCE_UI_PID_FILE,
        "data_collection": DATA_COLLECTION_PID_FILE
    }
    
    # Check each service port
    for service_type, port in service_ports.items():
        # Check if port is in use
        port_in_use = is_port_in_use(port)
        result["port_status"][service_type] = {
            "port": port,
            "in_use": port_in_use
        }
        
        # If port is in use, get process info
        if port_in_use:
            process_info = get_process_info_by_port(port)
            if process_info:
                result["port_status"][service_type]["process"] = process_info
                
                # Check if PID file exists
                pid_file = service_pid_files[service_type]
                pid_data = read_pid_file(pid_file)
                result["pid_file_status"][service_type] = {
                    "file": str(pid_file),
                    "exists": pid_file.exists(),
                    "valid": pid_data is not None,
                    "data": pid_data
                }
                
                # Check for conflict between PID file and actual process
                if pid_data and pid_data["pid"] != process_info["pid"]:
                    conflict = {
                        "service_type": service_type,
                        "port": port,
                        "pid_file_pid": pid_data["pid"],
                        "actual_pid": process_info["pid"],
                        "pid_file": str(pid_file),
                        "action": "update_pid_file"
                    }
                    
                    # Check if both processes are running
                    pid_file_process_running = check_process_running(pid_data["pid"])
                    
                    if pid_file_process_running:
                        # Both processes are running - we have a real conflict
                        conflict["both_processes_running"] = True
                        conflict["action"] = "resolve_conflict"
                        
                        # Add recommendation
                        result["recommendations"].append(
                            f"Port conflict: {service_type} on port {port}. "
                            f"PID file has {pid_data['pid']} but port is used by {process_info['pid']}. "
                            f"Run with --resolve to fix."
                        )
                    else:
                        # PID file refers to a non-existent process
                        conflict["both_processes_running"] = False
                        conflict["action"] = "update_pid_file"
                        
                        # Add recommendation
                        result["recommendations"].append(
                            f"Stale PID file for {service_type}. "
                            f"PID file has {pid_data['pid']} (not running) but port is used by {process_info['pid']}. "
                            f"Run with --update-pid-files to fix."
                        )
                    
                    result["conflicts"].append(conflict)
                    
        # Check for multiple instances of the same service
        processes = get_service_processes(service_type)
        
        if len(processes) > 1:
            # Found multiple instances of the same service
            active_processes = [p for p in processes if p["is_alive"]]
            
            if len(active_processes) > 1:
                # Multiple active instances
                conflict = {
                    "service_type": service_type,
                    "processes": active_processes,
                    "action": "terminate_duplicates"
                }
                
                # Sort by creation time, newest first
                sorted_procs = sorted(active_processes, key=lambda p: p["create_time_seconds"], reverse=True)
                
                # Keep the newest process with a bound port, if any
                keep_process = None
                for proc in sorted_procs:
                    if proc["bound_ports"]:
                        keep_process = proc
                        break
                        
                # If no process with bound port, keep the newest process
                if not keep_process and sorted_procs:
                    keep_process = sorted_procs[0]
                    
                if keep_process:
                    conflict["keep_pid"] = keep_process["pid"]
                    
                    # Add recommendation
                    result["recommendations"].append(
                        f"Multiple instances of {service_type} running. "
                        f"Keeping PID {keep_process['pid']} and terminating others. "
                        f"Run with --resolve to fix."
                    )
                    
                result["conflicts"].append(conflict)
    
    # Check for PID file format issues
    for service_type, pid_file in service_pid_files.items():
        if pid_file.exists():
            pid_data = read_pid_file(pid_file)
            
            if pid_data:
                # Check if format is integer (legacy)
                if pid_data.get("format") == "integer":
                    result["pid_file_status"][service_type] = {
                        "file": str(pid_file),
                        "exists": True,
                        "valid": True,
                        "legacy_format": True,
                        "data": pid_data
                    }
                    
                    # Add recommendation
                    result["recommendations"].append(
                        f"Legacy PID file format for {service_type}. "
                        f"Convert to JSON format with --update-pid-files."
                    )
    
    # Add general recommendations if conflicts found
    if result["conflicts"]:
        result["recommendations"].insert(0, 
            "Port conflicts detected. Run with --resolve to automatically fix these issues, "
            "or --update-pid-files to only update PID files without terminating processes."
        )
    else:
        result["recommendations"].append(
            "No port conflicts detected. All services appear to be running correctly."
        )
    
    return result

def resolve_port_conflicts(update_pid_files_only: bool = False) -> Dict[str, Any]:
    """
    Resolve port conflicts automatically.
    
    Args:
        update_pid_files_only: Only update PID files, don't terminate processes
        
    Returns:
        dict: Results of the resolution
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "conflicts_resolved": [],
        "pid_files_updated": [],
        "processes_terminated": [],
        "errors": []
    }
    
    # First, check for conflicts
    conflicts = check_for_port_conflicts()
    
    # No conflicts to resolve
    if not conflicts["conflicts"]:
        result["message"] = "No port conflicts detected"
        return result
        
    # Resolve each conflict
    for conflict in conflicts["conflicts"]:
        service_type = conflict["service_type"]
        
        if conflict["action"] == "update_pid_file":
            # Only need to update PID file to match actual process
            if "port" in conflict and "actual_pid" in conflict:
                port = conflict["port"]
                pid = conflict["actual_pid"]
                pid_file = service_pid_files[service_type]
                
                logger.info(f"Updating PID file {pid_file} to match process {pid} on port {port}")
                
                if write_pid_file(pid_file, pid, port, service_type):
                    result["pid_files_updated"].append({
                        "service_type": service_type,
                        "pid": pid,
                        "port": port,
                        "file": str(pid_file)
                    })
                else:
                    result["errors"].append(f"Failed to update PID file {pid_file}")
                    
                result["conflicts_resolved"].append({
                    "service_type": service_type,
                    "action": "updated_pid_file",
                    "pid": pid,
                    "port": port
                })
                
        elif conflict["action"] == "resolve_conflict" and not update_pid_files_only:
            # Need to resolve a genuine conflict
            port = conflict["port"]
            pid_file_pid = conflict["pid_file_pid"] 
            actual_pid = conflict["actual_pid"]
            pid_file = service_pid_files[service_type]
            
            logger.info(f"Resolving conflict for {service_type} on port {port}")
            logger.info(f"PID file has {pid_file_pid}, but port is used by {actual_pid}")
            
            # Check if PID file process is actually running
            pid_file_process_running = check_process_running(pid_file_pid)
            
            if pid_file_process_running:
                # Terminate the process from the PID file
                logger.info(f"Terminating process {pid_file_pid} from PID file")
                
                try:
                    process = psutil.Process(pid_file_pid)
                    process.terminate()
                    
                    # Wait for termination
                    try:
                        process.wait(timeout=PROCESS_TERMINATION_TIMEOUT)
                    except psutil.TimeoutExpired:
                        logger.warning(f"Process {pid_file_pid} did not terminate, using SIGKILL")
                        process.kill()
                        
                    result["processes_terminated"].append({
                        "pid": pid_file_pid,
                        "service_type": service_type,
                        "reason": "conflict_resolution"
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.warning(f"Could not terminate process {pid_file_pid}")
                    result["errors"].append(f"Could not terminate process {pid_file_pid}")
            
            # Update PID file to match the process actually using the port
            logger.info(f"Updating PID file {pid_file} to match process {actual_pid}")
            
            if write_pid_file(pid_file, actual_pid, port, service_type):
                result["pid_files_updated"].append({
                    "service_type": service_type,
                    "pid": actual_pid,
                    "port": port,
                    "file": str(pid_file)
                })
            else:
                result["errors"].append(f"Failed to update PID file {pid_file}")
                
            result["conflicts_resolved"].append({
                "service_type": service_type,
                "action": "terminated_and_updated",
                "terminated_pid": pid_file_pid,
                "kept_pid": actual_pid,
                "port": port
            })
                
        elif conflict["action"] == "terminate_duplicates" and not update_pid_files_only:
            # Need to terminate duplicate service instances
            processes = conflict["processes"]
            keep_pid = conflict.get("keep_pid")
            
            if keep_pid and len(processes) > 1:
                # Terminate all processes except the one to keep
                for proc in processes:
                    if proc["pid"] != keep_pid:
                        logger.info(f"Terminating duplicate {service_type} process {proc['pid']}")
                        
                        try:
                            process = psutil.Process(proc["pid"])
                            process.terminate()
                            
                            # Wait for termination
                            try:
                                process.wait(timeout=PROCESS_TERMINATION_TIMEOUT)
                            except psutil.TimeoutExpired:
                                logger.warning(f"Process {proc['pid']} did not terminate, using SIGKILL")
                                process.kill()
                                
                            result["processes_terminated"].append({
                                "pid": proc["pid"],
                                "service_type": service_type,
                                "reason": "duplicate_instance"
                            })
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            logger.warning(f"Could not terminate process {proc['pid']}")
                            result["errors"].append(f"Could not terminate process {proc['pid']}")
                
                # Update PID file with the kept process
                pid_file = service_pid_files[service_type]
                
                # Find default port for this service
                port = service_ports[service_type]
                
                # If the kept process has bound ports, use the first one
                kept_process = next((p for p in processes if p["pid"] == keep_pid), None)
                if kept_process and kept_process["bound_ports"]:
                    port = kept_process["bound_ports"][0]
                    
                logger.info(f"Updating PID file {pid_file} with kept process {keep_pid} on port {port}")
                
                if write_pid_file(pid_file, keep_pid, port, service_type):
                    result["pid_files_updated"].append({
                        "service_type": service_type,
                        "pid": keep_pid,
                        "port": port,
                        "file": str(pid_file)
                    })
                else:
                    result["errors"].append(f"Failed to update PID file {pid_file}")
                    
                result["conflicts_resolved"].append({
                    "service_type": service_type,
                    "action": "terminated_duplicates",
                    "kept_pid": keep_pid,
                    "terminated_count": len(processes) - 1
                })
    
    # Fix any PID file format issues
    for service_type, status in conflicts.get("pid_file_status", {}).items():
        if status.get("legacy_format"):
            pid_file = service_pid_files[service_type]
            pid_data = status["data"]
            
            # Check if process is still running
            pid = pid_data["pid"]
            if check_process_running(pid):
                # Process is running, update PID file to JSON format
                port = service_ports[service_type]  # Default port
                
                # Try to get actual port from process connections
                try:
                    process = psutil.Process(pid)
                    for conn in process.connections(kind='inet'):
                        if conn.status == 'LISTEN':
                            port = conn.laddr.port
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
                    
                logger.info(f"Converting legacy PID file {pid_file} to JSON format")
                
                if write_pid_file(pid_file, pid, port, service_type):
                    result["pid_files_updated"].append({
                        "service_type": service_type,
                        "pid": pid,
                        "port": port,
                        "file": str(pid_file),
                        "action": "format_conversion"
                    })
                else:
                    result["errors"].append(f"Failed to convert PID file {pid_file}")
            else:
                # Process is not running, remove PID file
                logger.info(f"Removing stale legacy PID file {pid_file}")
                
                try:
                    pid_file.unlink()
                    result["pid_files_updated"].append({
                        "service_type": service_type,
                        "action": "removed_stale"
                    })
                except Exception as e:
                    logger.error(f"Failed to remove stale PID file {pid_file}: {e}")
                    result["errors"].append(f"Failed to remove stale PID file {pid_file}")
    
    # Return final result with summary
    if result["conflicts_resolved"]:
        result["message"] = f"Resolved {len(result['conflicts_resolved'])} conflicts"
    elif result["pid_files_updated"]:
        result["message"] = f"Updated {len(result['pid_files_updated'])} PID files"
    else:
        result["message"] = "No changes made"
        
    return result

def init_service(service_type: str) -> Dict[str, Any]:
    """
    Initialize a service with proper port locking and PID file management.
    
    Args:
        service_type: Type of service to initialize
        
    Returns:
        dict: Results of the initialization
    """
    result = {
        "service_type": service_type,
        "success": False,
        "pid": None,
        "port": None,
        "error": None
    }
    
    # Get default port for this service
    if service_type not in service_ports:
        result["error"] = f"Unknown service type: {service_type}"
        return result
        
    port = service_ports[service_type]
    pid_file = service_pid_files[service_type]
    
    logger.info(f"Initializing {service_type} service on port {port}")
    
    # First, check if the service is already running
    processes = get_service_processes(service_type)
    active_processes = [p for p in processes if p["is_alive"]]
    
    if active_processes:
        # Service is already running
        sorted_procs = sorted(active_processes, key=lambda p: p["create_time_seconds"], reverse=True)
        
        # Check if any process has the port bound
        bound_processes = [p for p in sorted_procs if p["bound_ports"] and port in p["bound_ports"]]
        
        if bound_processes:
            # Port is already bound by a process of the correct type
            process = bound_processes[0]  # Use the newest process with bound port
            
            result["success"] = True
            result["pid"] = process["pid"]
            result["port"] = port
            result["message"] = f"{service_type} already running on port {port} (PID: {process['pid']})"
            result["action"] = "already_running"
            
            # Update PID file to ensure it matches
            write_pid_file(pid_file, process["pid"], port, service_type)
            
            return result
    
    # Try to acquire service lock
    success, lock_pid, error = acquire_service_lock(service_type, port)
    
    if not success:
        result["error"] = error or f"Failed to acquire lock for {service_type} on port {port}"
        return result
        
    # Lock acquired successfully
    result["success"] = True
    result["pid"] = lock_pid
    result["port"] = port
    result["message"] = f"Successfully initialized {service_type} on port {port} (PID: {lock_pid})"
    result["action"] = "initialized"
    
    # Create PID file
    write_pid_file(pid_file, lock_pid, port, service_type)
    
    return result

# Define dictionaries of service ports and PID files
service_ports = {
    "fastapi": 8001,
    "inference_ui": 7861,
    "data_collection": 7862
}

service_pid_files = {
    "fastapi": INFERENCE_PROCESS_PID_FILE,
    "inference_ui": INFERENCE_UI_PID_FILE,
    "data_collection": DATA_COLLECTION_PID_FILE
}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Port Conflict Resolution Utility")
    parser.add_argument("--check", action="store_true", help="Check for port conflicts")
    parser.add_argument("--resolve", action="store_true", help="Automatically resolve port conflicts")
    parser.add_argument("--update-pid-files", action="store_true", help="Update PID files only, don't terminate processes")
    parser.add_argument("--init-service", choices=["fastapi", "inference_ui", "data_collection"], help="Initialize a service with proper locking")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    if args.check:
        print("Checking for port conflicts...")
        conflicts = check_for_port_conflicts()
        
        print("\nPort Status:")
        for service_type, status in conflicts["port_status"].items():
            print(f"- {service_type}: Port {status['port']} is {'IN USE' if status['in_use'] else 'AVAILABLE'}")
            if status.get("process"):
                print(f"  Used by PID {status['process']['pid']} ({status['process'].get('name', 'unknown')})")
                
        print("\nPID File Status:")
        for service_type, status in conflicts.get("pid_file_status", {}).items():
            if status:
                print(f"- {service_type}: {status['file']} {'EXISTS' if status['exists'] else 'MISSING'} {'(valid)' if status.get('valid') else '(invalid)'}")
                if status.get("data"):
                    print(f"  PID: {status['data']['pid']}")
                    if status.get("legacy_format"):
                        print("  Warning: Legacy format (needs conversion)")
        
        if conflicts["conflicts"]:
            print(f"\nDetected {len(conflicts['conflicts'])} conflicts:")
            for i, conflict in enumerate(conflicts["conflicts"], 1):
                action = conflict["action"].replace("_", " ").title()
                service = conflict["service_type"]
                
                if "port" in conflict:
                    port = conflict["port"]
                    if "pid_file_pid" in conflict and "actual_pid" in conflict:
                        print(f"{i}. {service} on port {port}: PID file has {conflict['pid_file_pid']} but port used by {conflict['actual_pid']}")
                        if conflict.get("both_processes_running"):
                            print(f"   Action needed: {action} (both processes are running)")
                        else:
                            print(f"   Action needed: {action}")
                elif "processes" in conflict:
                    print(f"{i}. Multiple instances of {service} detected ({len(conflict['processes'])} processes)")
                    if "keep_pid" in conflict:
                        print(f"   Action needed: Keep PID {conflict['keep_pid']} and terminate others")
        else:
            print("\nNo conflicts detected.")
            
        if conflicts["recommendations"]:
            print("\nRecommendations:")
            for i, recommendation in enumerate(conflicts["recommendations"], 1):
                print(f"{i}. {recommendation}")
    
    elif args.resolve or args.update_pid_files:
        print("Resolving port conflicts...")
        result = resolve_port_conflicts(args.update_pid_files)
        
        print(f"\nResolution complete: {result['message']}")
        
        if result["conflicts_resolved"]:
            print(f"\nResolved {len(result['conflicts_resolved'])} conflicts:")
            for conflict in result["conflicts_resolved"]:
                action = conflict["action"].replace("_", " ").title()
                service = conflict["service_type"]
                
                if "port" in conflict:
                    port = conflict["port"]
                    if "terminated_pid" in conflict and "kept_pid" in conflict:
                        print(f"- {service} on port {port}: Terminated PID {conflict['terminated_pid']} and kept {conflict['kept_pid']}")
                    elif "pid" in conflict:
                        print(f"- {service} on port {port}: Updated PID file to {conflict['pid']}")
                elif "kept_pid" in conflict and "terminated_count" in conflict:
                    print(f"- {service}: Kept PID {conflict['kept_pid']} and terminated {conflict['terminated_count']} duplicates")
        
        if result["pid_files_updated"]:
            print(f"\nUpdated {len(result['pid_files_updated'])} PID files:")
            for update in result["pid_files_updated"]:
                service = update["service_type"]
                if "pid" in update:
                    print(f"- {service}: Updated PID file to {update['pid']} on port {update.get('port')}")
                elif update.get("action") == "removed_stale":
                    print(f"- {service}: Removed stale PID file")
                else:
                    print(f"- {service}: Updated PID file")
        
        if result["errors"]:
            print(f"\nEncountered {len(result['errors'])} errors:")
            for error in result["errors"]:
                print(f"- {error}")
    
    elif args.init_service:
        print(f"Initializing {args.init_service} service...")
        result = init_service(args.init_service)
        
        if result["success"]:
            print(f"Success: {result['message']}")
        else:
            print(f"Error: {result['error']}")
    
    else:
        parser.print_help()