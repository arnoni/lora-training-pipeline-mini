#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/pid_file_handler.py

"""
PID File Handler

Standardized module for all PID file operations across the application,
ensuring consistent format and preventing duplicate processes.

This module resolves the critical issue of inconsistent PID file formatting
that was causing ValueError exceptions and leading to duplicate processes.
"""

import os
import sys
import json
import fcntl
import psutil
import socket
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger('pid_file_handler')

# PID file constants
DEFAULT_PID_FILE_DIR = Path('.')
FASTAPI_PID_FILE = Path('./inference_process.pid')
DATA_COLLECTION_PID_FILE = Path('./data_collection_ui.pid')
INFERENCE_UI_PID_FILE = Path('./inference_ui.pid')

# Process types
PROCESS_TYPES = {
    'fastapi': {'default_port': 8001, 'pid_file': FASTAPI_PID_FILE},
    'data_collection': {'default_port': 7862, 'pid_file': DATA_COLLECTION_PID_FILE},
    'inference_ui': {'default_port': 7861, 'pid_file': INFERENCE_UI_PID_FILE},
}

class PIDFileError(Exception):
    """Base exception for PID file related errors."""
    pass

class PIDParseError(PIDFileError):
    """Exception raised when a PID file cannot be parsed."""
    pass

class PIDFileAccessError(PIDFileError):
    """Exception raised when a PID file cannot be accessed."""
    pass

class ProcessRunningError(PIDFileError):
    """Exception raised when a process is already running."""
    pass

def read_pid_file(pid_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a PID file with standardized error handling.
    
    Args:
        pid_file: Path to the PID file
        
    Returns:
        dict: Dictionary with pid and metadata
        
    Raises:
        PIDParseError: If the PID file exists but cannot be parsed
        PIDFileAccessError: If the PID file cannot be accessed
    """
    # Standardize path
    try:
        pid_file = Path(pid_file).resolve()
    except Exception as e:
        raise PIDFileAccessError(f"Invalid PID file path: {e}")
    
    # Check existence
    if not pid_file.exists():
        logger.debug(f"PID file {pid_file} does not exist")
        return {"exists": False}
    
    # Check accessibility
    if not os.access(pid_file, os.R_OK):
        raise PIDFileAccessError(f"PID file {pid_file} is not readable")
    
    # Try to read with locking
    try:
        with open(pid_file, 'r') as f:
            # Acquire shared lock for reading
            try:
                fcntl.flock(f, fcntl.LOCK_SH)
                content = f.read().strip()
            finally:
                try:
                    fcntl.flock(f, fcntl.LOCK_UN)
                except (IOError, OSError) as unlock_err:
                    logger.debug(f"[DEBUG] File unlock error (ignoring): {unlock_err}")
                    pass  # Ignore unlock errors
    except Exception as e:
        raise PIDFileAccessError(f"Error reading PID file {pid_file}: {e}")
    
    # Handle empty file
    if not content:
        logger.warning(f"PID file {pid_file} is empty")
        return {"exists": True, "valid": False, "error": "Empty file"}
    
    # Try to parse as JSON first
    try:
        data = json.loads(content)
        
        # Validate PID field
        if "pid" not in data:
            raise PIDParseError(f"Missing 'pid' field in PID file {pid_file}")
        
        try:
            pid = int(data["pid"])
            if pid <= 0:
                raise PIDParseError(f"Invalid PID value {pid} in file {pid_file}")
            
            # Update data
            data["pid"] = pid
            data["format"] = "json"
            data["exists"] = True
            data["valid"] = True
            
            # Add port if missing but known
            if "port" not in data and "process_type" in data:
                if data["process_type"] in PROCESS_TYPES:
                    data["port"] = PROCESS_TYPES[data["process_type"]]["default_port"]
            
            return data
        except (ValueError, TypeError) as e:
            raise PIDParseError(f"Invalid PID value in file {pid_file}: {e}")
            
    except json.JSONDecodeError:
        # Try to parse as plain integer
        try:
            pid = int(content)
            if pid <= 0:
                raise PIDParseError(f"Invalid PID value {pid} in file {pid_file}")
            
            # Create a standardized data dictionary
            data = {
                "pid": pid,
                "format": "integer",
                "exists": True,
                "valid": True,
                "converted": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Try to determine process type from file name
            process_type = None
            for p_type, info in PROCESS_TYPES.items():
                if str(info["pid_file"]) == str(pid_file):
                    process_type = p_type
                    data["process_type"] = p_type
                    data["port"] = info["default_port"]
                    break
            
            return data
        except ValueError as e:
            raise PIDParseError(f"PID file {pid_file} contains invalid data: {e}")

def write_pid_file(
    pid_file: Union[str, Path], 
    pid: int = None, 
    process_type: str = None, 
    port: int = None, 
    additional_data: Dict[str, Any] = None
) -> bool:
    """
    Write a PID file in standardized JSON format with atomic operations.
    
    Args:
        pid_file: Path to the PID file
        pid: Process ID (default: current process)
        process_type: Type of process ('fastapi', 'data_collection', 'inference_ui')
        port: Port number
        additional_data: Additional data to include
        
    Returns:
        bool: True if successful
        
    Raises:
        PIDFileAccessError: If the PID file cannot be written
        ValueError: If the parameters are invalid
    """
    # Validate parameters
    pid = os.getpid() if pid is None else pid
    
    try:
        pid = int(pid)
        if pid <= 0:
            raise ValueError(f"Invalid PID value: {pid}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid PID parameter: {e}")
    
    if port is not None:
        try:
            port = int(port)
            if port < 1 or port > 65535:
                raise ValueError(f"Invalid port number: {port}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid port parameter: {e}")
    
    # Standardize path
    try:
        pid_file = Path(pid_file).resolve()
    except Exception as e:
        raise PIDFileAccessError(f"Invalid PID file path: {e}")
    
    # Create parent directory if needed
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    data = {
        "pid": pid,
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname()
    }
    
    # Add process type if provided
    if process_type:
        data["process_type"] = process_type
        
        # Add default port if not specified
        if port is None and process_type in PROCESS_TYPES:
            port = PROCESS_TYPES[process_type]["default_port"]
    
    # Add port if provided
    if port is not None:
        data["port"] = port
    
    # Add additional data
    if additional_data:
        # Don't overwrite core fields
        for key, value in additional_data.items():
            if key not in ["pid", "timestamp", "hostname"]:
                data[key] = value
    
    # Add command line for diagnostic purposes
    try:
        data["command"] = " ".join(sys.argv)
    except Exception as cmd_err:
        logger.debug(f"[DEBUG] Error getting command line for PID file: {cmd_err}")
        pass
    
    # Use atomic write with temporary file
    temp_file = pid_file.with_suffix('.tmp')
    
    try:
        # Write to temporary file with locking
        with open(temp_file, 'w') as f:
            # Acquire exclusive lock
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk
            finally:
                try:
                    fcntl.flock(f, fcntl.LOCK_UN)
                except (IOError, OSError) as unlock_err:
                    logger.debug(f"[DEBUG] File unlock error (ignoring): {unlock_err}")
                    pass  # Ignore unlock errors
        
        # Atomic rename
        temp_file.replace(pid_file)
        
        logger.debug(f"Successfully wrote PID file {pid_file} with PID {pid}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing PID file {pid_file}: {e}")
        logger.error(traceback.format_exc())
        
        # Clean up temporary file if it exists
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception as cleanup_err:
            logger.debug(f"[DEBUG] Error cleaning up temporary file: {cleanup_err}")
            pass
            
        raise PIDFileAccessError(f"Error writing PID file {pid_file}: {e}")

def check_process_running(
    pid_file: Union[str, Path],
    verify_process_type: bool = True,
    verify_port: bool = True
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Check if a process described in a PID file is running.
    
    Args:
        pid_file: Path to the PID file
        verify_process_type: Whether to verify the process matches the expected type
        verify_port: Whether to verify the process is using the expected port
        
    Returns:
        Tuple[bool, dict]: (is_running, pid_data)
        
    Raises:
        PIDFileAccessError: If the PID file cannot be accessed
        PIDParseError: If the PID file cannot be parsed
    """
    # Read PID file
    try:
        data = read_pid_file(pid_file)
    except (PIDFileAccessError, PIDParseError) as e:
        logger.warning(str(e))
        return False, None
    
    # Check if PID file exists and is valid
    if not data.get("exists", False) or not data.get("valid", False):
        return False, data
    
    # Get PID
    pid = data.get("pid")
    if not pid:
        return False, data
    
    # Check if process exists
    try:
        process = psutil.Process(pid)
        
        # Check if zombie
        if process.status() == psutil.STATUS_ZOMBIE:
            logger.debug(f"Process {pid} is a zombie")
            return False, data
        
        # Verify process type if requested
        if verify_process_type and "process_type" in data:
            process_type = data["process_type"]
            cmdline = " ".join(process.cmdline())
            
            # Check if command line matches expected process type
            if process_type == "fastapi" and not any(x in cmdline.lower() for x in ["fastapi", "uvicorn", "inference_server"]):
                logger.warning(f"Process {pid} does not appear to be a FastAPI server: {cmdline[:100]}")
                return False, data
            
            elif process_type == "data_collection" and not any(x in cmdline.lower() for x in ["data_collection", "gradio"]):
                logger.warning(f"Process {pid} does not appear to be a Data Collection UI: {cmdline[:100]}")
                return False, data
                
            elif process_type == "inference_ui" and not any(x in cmdline.lower() for x in ["inference", "gradio"]):
                logger.warning(f"Process {pid} does not appear to be an Inference UI: {cmdline[:100]}")
                return False, data
        
        # Verify port if requested
        if verify_port and "port" in data:
            port = data["port"]
            
            # Check if process is using the expected port
            for conn in process.connections():
                if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port') and conn.laddr.port == port:
                    logger.debug(f"Process {pid} is using port {port} as expected")
                    # Process is running and using the expected port
                    return True, data
            
            # Process is running but not using the expected port
            logger.warning(f"Process {pid} is running but not using port {data['port']}")
            
            # Still consider it running but note the port mismatch
            data["port_mismatch"] = True
            return True, data
        
        # Process is running
        logger.debug(f"Process {pid} is running")
        return True, data
        
    except psutil.NoSuchProcess:
        logger.debug(f"Process {pid} does not exist")
        return False, data
        
    except psutil.AccessDenied:
        logger.warning(f"Access denied for process {pid}, assuming it's running")
        return True, data
        
    except Exception as e:
        logger.error(f"Error checking process {pid}: {e}")
        logger.error(traceback.format_exc())
        return False, data

def ensure_single_instance(
    process_type: str,
    port: int = None,
    pid_file: Union[str, Path] = None,
    check_health: bool = True,
    additional_data: Dict[str, Any] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Ensure only one instance of a process is running.
    
    Args:
        process_type: Type of process ('fastapi', 'data_collection', 'inference_ui')
        port: Port number (optional, default from PROCESS_TYPES)
        pid_file: Path to the PID file (optional, default from PROCESS_TYPES)
        check_health: Whether to check process health
        additional_data: Additional data to include in PID file
        
    Returns:
        Tuple[bool, str, dict]: (can_proceed, message, pid_data)
        
    Raises:
        ValueError: If the process type is invalid
    """
    # Validate process type
    if process_type not in PROCESS_TYPES:
        raise ValueError(f"Invalid process type: {process_type}")
    
    # Use default port if not specified
    if port is None:
        port = PROCESS_TYPES[process_type]["default_port"]
    
    # Use default PID file if not specified
    if pid_file is None:
        pid_file = PROCESS_TYPES[process_type]["pid_file"]
    
    # Check if PID file exists and points to a running process
    try:
        is_running, pid_data = check_process_running(pid_file, verify_process_type=True, verify_port=True)
        
        if is_running:
            # Process is already running
            pid = pid_data.get("pid")
            message = f"{process_type} is already running (PID: {pid})"
            
            # If port mismatch, note it but still consider it running
            if pid_data.get("port_mismatch"):
                message += f" (port mismatch: expected {port}, actual unknown)"
            
            return False, message, pid_data
    except (PIDFileAccessError, PIDParseError) as e:
        # PID file issues - continue and potentially overwrite it
        logger.warning(str(e))
    
    # Check if port is in use by something else
    if check_health:
        try:
            # Import here to avoid circular imports
            from src.lora_training_pipeline.utils.fix_port_issues import is_port_in_use
            
            if is_port_in_use(port):
                logger.warning(f"Port {port} is already in use by another process")
                message = f"Port {port} is already in use by another process"
                return False, message, None
        except ImportError:
            # fall back to simple socket check
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    result = s.connect_ex(('localhost', port))
                    if result == 0:
                        logger.warning(f"Port {port} is already in use by another process")
                        message = f"Port {port} is already in use by another process"
                        return False, message, None
            except Exception as e:
                logger.warning(f"Error checking port {port}: {e}")
                # Continue - assume port is available
    
    # All checks passed, write PID file for this process
    try:
        write_pid_file(pid_file, process_type=process_type, port=port, additional_data=additional_data)
        message = f"{process_type} can start on port {port}"
        
        # Create minimal pid_data
        pid_data = {
            "pid": os.getpid(),
            "port": port,
            "process_type": process_type
        }
        
        return True, message, pid_data
    except Exception as e:
        logger.error(f"Error writing PID file {pid_file}: {e}")
        message = f"Error writing PID file: {e}"
        return False, message, None

def cleanup_stale_pid_file(pid_file: Union[str, Path]) -> bool:
    """
    Clean up a stale PID file.
    
    Args:
        pid_file: Path to the PID file
        
    Returns:
        bool: True if file was removed, False otherwise
    """
    try:
        pid_file = Path(pid_file)
        
        # Check if file exists
        if not pid_file.exists():
            return False
        
        # Check if process is running
        is_running, _ = check_process_running(pid_file)
        
        if not is_running:
            # Process is not running, remove PID file
            pid_file.unlink()
            logger.info(f"Removed stale PID file {pid_file}")
            return True
        
        return False
    except Exception as e:
        logger.error(f"Error cleaning up stale PID file {pid_file}: {e}")
        return False

def cleanup_stale_pid_files() -> Dict[str, int]:
    """
    Clean up all stale PID files.
    
    Returns:
        dict: Count of removed files by type
    """
    result = {
        "fastapi": 0,
        "data_collection": 0,
        "inference_ui": 0,
        "unknown": 0
    }
    
    # Check known PID files
    for process_type, info in PROCESS_TYPES.items():
        pid_file = info["pid_file"]
        if cleanup_stale_pid_file(pid_file):
            result[process_type] += 1
    
    # Look for other .pid files
    for pid_file in Path('.').glob('*.pid'):
        # Skip known files to avoid double-counting
        if any(str(pid_file) == str(info["pid_file"]) for info in PROCESS_TYPES.values()):
            continue
            
        if cleanup_stale_pid_file(pid_file):
            result["unknown"] += 1
    
    return result

# Example usage
if __name__ == "__main__":
    import argparse
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="PID File Handler")
    parser.add_argument("--check", choices=['fastapi', 'data_collection', 'inference_ui', 'all'],
                      help="Check if a process is running")
    parser.add_argument("--write", choices=['fastapi', 'data_collection', 'inference_ui'],
                      help="Write a PID file for the current process")
    parser.add_argument("--port", type=int, help="Port number for the process")
    parser.add_argument("--cleanup", action="store_true", help="Clean up stale PID files")
    parser.add_argument("--ensure", choices=['fastapi', 'data_collection', 'inference_ui'],
                      help="Ensure single instance of a process")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    if args.check:
        process_types = list(PROCESS_TYPES.keys()) if args.check == 'all' else [args.check]
        
        for process_type in process_types:
            pid_file = PROCESS_TYPES[process_type]["pid_file"]
            
            print(f"Checking {process_type} (PID file: {pid_file})...")
            
            try:
                is_running, data = check_process_running(pid_file)
                
                if is_running:
                    print(f"✅ {process_type} is running (PID: {data.get('pid')})")
                    
                    if data.get("port"):
                        print(f"   Port: {data.get('port')}")
                    
                    if data.get("port_mismatch"):
                        print(f"   ⚠️ Port mismatch detected")
                else:
                    print(f"❌ {process_type} is not running")
                    
                    if data and data.get("exists"):
                        print(f"   PID file exists but process is not running")
                        
                        if data.get("pid"):
                            print(f"   PID in file: {data.get('pid')}")
            except Exception as e:
                print(f"❌ Error checking {process_type}: {e}")
    
    elif args.write:
        process_type = args.write
        pid_file = PROCESS_TYPES[process_type]["pid_file"]
        port = args.port or PROCESS_TYPES[process_type]["default_port"]
        
        print(f"Writing PID file for {process_type} (PID: {os.getpid()}, port: {port})...")
        
        try:
            write_pid_file(pid_file, process_type=process_type, port=port)
            print(f"✅ Successfully wrote PID file {pid_file}")
        except Exception as e:
            print(f"❌ Error writing PID file: {e}")
    
    elif args.cleanup:
        print("Cleaning up stale PID files...")
        
        result = cleanup_stale_pid_files()
        
        total = sum(result.values())
        if total > 0:
            print(f"✅ Removed {total} stale PID files:")
            for process_type, count in result.items():
                if count > 0:
                    print(f"   - {process_type}: {count}")
        else:
            print("✅ No stale PID files found")
    
    elif args.ensure:
        process_type = args.ensure
        port = args.port or PROCESS_TYPES[process_type]["default_port"]
        
        print(f"Ensuring single instance of {process_type} on port {port}...")
        
        try:
            can_proceed, message, data = ensure_single_instance(process_type, port)
            
            if can_proceed:
                print(f"✅ {message}")
                print(f"   PID: {os.getpid()}")
                print(f"   Port: {port}")
            else:
                print(f"❌ {message}")
                
                if data and "pid" in data:
                    print(f"   PID: {data['pid']}")
                if data and "port" in data:
                    print(f"   Port: {data['port']}")
        except Exception as e:
            print(f"❌ Error ensuring single instance: {e}")
    
    else:
        parser.print_help()