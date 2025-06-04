#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/fix_port_issues.py

"""
Port Conflict Resolution Utility

This module fixes port and process issues by:
1. Standardizing PID file formats
2. Adding proper port checking before service startup
3. Adding explicit service locks
4. Validating process liveness before termination decisions
"""

import os
import sys
import json
import socket
import psutil
import time
import traceback
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('port_fix.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('port_fix')

# Constants
INFERENCE_PROCESS_PID_FILE = Path('./inference_process.pid')
DATA_COLLECTION_PID_FILE = Path('./data_collection_ui.pid')
INFERENCE_UI_PID_FILE = Path('./inference_ui.pid')
PROCESS_LOCKS_DIR = Path('./process_locks')

def is_port_in_use(port):
    """
    Check if a port is already in use.
    
    Args:
        port: Port number to check
        
    Returns:
        bool: True if port is in use, False otherwise
        
    Raises:
        ValueError: If port is not a valid port number
    """
    # Validate port parameter
    try:
        port = int(port)
        if port < 1 or port > 65535:
            raise ValueError(f"Invalid port number: {port} (must be 1-65535)")
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid port parameter: {e}")
        # Log the stack trace for debugging
        logger.error(traceback.format_exc())
        raise ValueError(f"Invalid port parameter: {e}")
    
    logger.info(f"Checking if port {port} is in use...")
    
    # Try multiple connection attempts for reliability
    attempts = 2
    for attempt in range(attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)  # 2 second timeout
                
                # For debugging connection attempts
                logger.debug(f"Port {port} check attempt {attempt+1}/{attempts}")
                
                # Try to connect - if connection succeeds, port is in use
                result = s.connect_ex(('localhost', port))
                port_in_use = result == 0
                
                # Log detailed result code for debugging
                logger.debug(f"Socket connect_ex return code: {result}")
                
                if port_in_use:
                    logger.info(f"Port {port} is IN USE (code: {result})")
                    
                    # Try to get more information about what's using the port
                    try:
                        for proc in psutil.process_iter(['pid', 'name', 'connections']):
                            for conn in proc.info.get('connections', []):
                                if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port') and conn.laddr.port == port:
                                    logger.info(f"Port {port} is being used by: PID {proc.info['pid']} ({proc.info['name']})")
                                    break
                    except Exception as proc_err:
                        logger.warning(f"Could not identify process using port {port}: {proc_err}")
                else:
                    logger.info(f"Port {port} is AVAILABLE (code: {result})")
                
                return port_in_use
                
        except socket.error as e:
            logger.error(f"Socket error checking port {port} (attempt {attempt+1}/{attempts}): {e}")
            logger.debug(traceback.format_exc())
            
            # Only retry if we have attempts left
            if attempt < attempts - 1:
                logger.info(f"Retrying port check for port {port}")
                time.sleep(0.5)  # Wait a bit before retry
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
    logger.warning(f"Port check code reached unexpected point for port {port}")
    return True

def check_process_running(pid):
    """
    Check if a process with the given PID is running.
    
    Args:
        pid: Process ID to check
        
    Returns:
        bool: True if process is running, False otherwise
        
    Raises:
        ValueError: If pid is not a valid process ID
    """
    # Validate pid parameter
    try:
        pid = int(pid)
        if pid <= 0:
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
            logger.debug(f"Process {pid} info: name={process_name}, cmdline={cmd_line[:100]}...")
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

def read_pid_file(pid_file):
    """
    Read a PID file, supporting both JSON and integer formats.
    
    Args:
        pid_file: Path to the PID file
        
    Returns:
        dict: Dictionary with pid and metadata, or None if file doesn't exist/read fails
    """
    # First ensure pid_file is a Path object
    try:
        pid_file = Path(pid_file)
    except Exception as e:
        logger.error(f"Invalid PID file path: {e}")
        logger.error(traceback.format_exc())
        return None
    
    # Check file existence
    if not pid_file.exists():
        logger.info(f"PID file {pid_file} does not exist")
        return None
        
    try:
        # Check file permissions and readability
        if not os.access(pid_file, os.R_OK):
            logger.error(f"PID file {pid_file} exists but is not readable")
            return None
        
        # Get file stats for debugging
        try:
            stats = pid_file.stat()
            file_size = stats.st_size
            mod_time = datetime.fromtimestamp(stats.st_mtime)
            logger.debug(f"PID file {pid_file} stats: size={file_size} bytes, modified={mod_time}")
        except Exception as stat_err:
            logger.warning(f"Could not get file stats for {pid_file}: {stat_err}")
        
        # Read file content with proper error handling
        try:
            with open(pid_file, 'r') as f:
                content = f.read().strip()
        except (IOError, PermissionError) as e:
            logger.error(f"I/O error reading PID file {pid_file}: {e}")
            return None
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error reading PID file {pid_file}: {e}")
            # Try with different encoding
            try:
                with open(pid_file, 'r', encoding='latin-1') as f:
                    content = f.read().strip()
                logger.info(f"Successfully read PID file with latin-1 encoding")
            except Exception as alt_e:
                logger.error(f"Failed to read with alternative encoding: {alt_e}")
                return None
                
        # If empty, return None
        if not content:
            logger.info(f"PID file {pid_file} is empty")
            return None
            
        # Debug the content
        logger.debug(f"PID file content (first 100 chars): {content[:100]}")
        
        # Try to parse as JSON first
        try:
            data = json.loads(content)
            logger.debug(f"Successfully parsed JSON from PID file: keys={list(data.keys())}")
            
            # Validate PID field exists and is an integer
            if "pid" in data:
                try:
                    pid_value = data["pid"]
                    data["pid"] = int(pid_value)
                    logger.info(f"Read JSON PID file {pid_file} with PID {data['pid']}")
                    
                    # Check timestamp if available
                    if "timestamp" in data:
                        logger.debug(f"PID file timestamp: {data['timestamp']}")
                        
                    # Return the validated data
                    return data
                except (ValueError, TypeError) as pid_err:
                    logger.warning(f"Invalid PID value in JSON: {pid_value!r}: {pid_err}")
                    return None
            else:
                logger.warning(f"Missing 'pid' field in JSON PID file")
                return None
                
        except json.JSONDecodeError as json_err:
            logger.info(f"PID file is not in JSON format: {json_err}")
            
            # Not JSON, try as plain integer with extra validation
            try:
                # Remove any whitespace and check for numeric content
                content = content.strip()
                if not content.isdigit() and not (content.startswith('-') and content[1:].isdigit()):
                    logger.warning(f"PID file content is not a valid integer: {content!r}")
                    return None
                    
                pid = int(content)
                
                # Validate PID value is reasonable
                if pid <= 0:
                    logger.warning(f"Invalid negative or zero PID in file: {pid}")
                    return None
                elif pid > 4194304:  # Common upper limit of PIDs in most systems
                    logger.warning(f"Suspiciously large PID in file: {pid}")
                    # Continue anyway, but log the warning
                
                logger.info(f"Read legacy integer PID file {pid_file} with PID {pid}")
                return {
                    "pid": pid, 
                    "legacy_format": True,
                    "detected_at": datetime.now().isoformat()
                }
            except ValueError as val_err:
                logger.warning(f"PID file {pid_file} contains invalid data (not JSON or integer): {val_err}")
                return None
                
    except Exception as e:
        logger.error(f"Unexpected error reading PID file {pid_file}: {e}")
        logger.error(traceback.format_exc())
        return None

def fix_pid_file(pid_file):
    """
    Fix a PID file by standardizing its format.
    
    Args:
        pid_file: Path to the PID file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not pid_file.exists():
        logger.info(f"PID file {pid_file} does not exist, nothing to fix")
        return False
    
    logger.info(f"Fixing PID file {pid_file}...")
    
    # Read current content
    data = read_pid_file(pid_file)
    if not data:
        logger.warning(f"Could not read PID file {pid_file}, removing it")
        try:
            pid_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to remove invalid PID file {pid_file}: {e}")
            return False
    
    # Get the PID
    pid = data.get("pid")
    if not pid:
        logger.warning(f"No valid PID in {pid_file}, removing it")
        try:
            pid_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to remove invalid PID file {pid_file}: {e}")
            return False
    
    # Check if process is running
    if not check_process_running(pid):
        logger.info(f"Process {pid} is not running, removing PID file {pid_file}")
        try:
            pid_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to remove stale PID file {pid_file}: {e}")
            return False
    
    # If the PID file is in legacy format, upgrade it to JSON
    if data.get("legacy_format"):
        logger.info(f"Converting legacy PID file {pid_file} to JSON format")
        try:
            # Extract process info for metadata
            process = psutil.Process(pid)
            cmdline = " ".join(process.cmdline())
            
            # Create standardized JSON with metadata
            new_data = {
                "pid": pid,
                "timestamp": datetime.now().isoformat(),
                "command": cmdline,
                "converted_from_legacy": True,
                "fixed_at": datetime.now().isoformat()
            }
            
            # Write back as JSON
            with open(pid_file, 'w') as f:
                json.dump(new_data, f, indent=2)
                
            logger.info(f"Successfully converted {pid_file} to JSON format")
            return True
        except Exception as e:
            logger.error(f"Failed to convert legacy PID file {pid_file}: {e}")
            return False
    
    # If we get here, the PID file is already in good shape
    logger.info(f"PID file {pid_file} is already in correct format")
    return True

def fix_all_pid_files():
    """
    Fix all PID files to ensure they have consistent format.
    
    Returns:
        dict: Summary of actions taken
    """
    result = {
        "fixed": [],
        "removed": [],
        "errors": []
    }
    
    # List of standard PID files to check
    pid_files = [
        INFERENCE_PROCESS_PID_FILE,
        DATA_COLLECTION_PID_FILE,
        INFERENCE_UI_PID_FILE
    ]
    
    # Look for additional PID files
    for pid_file in Path('.').glob('*.pid'):
        if pid_file not in pid_files:
            pid_files.append(pid_file)
    
    # Fix each PID file
    for pid_file in pid_files:
        if not pid_file.exists():
            continue
            
        try:
            # Read file to determine action
            data = read_pid_file(pid_file)
            if not data:
                # Invalid format - remove the file
                logger.info(f"Removing invalid PID file {pid_file}")
                try:
                    pid_file.unlink()
                    result["removed"].append(str(pid_file))
                except Exception as e:
                    error = f"Failed to remove invalid PID file {pid_file}: {e}"
                    logger.error(error)
                    result["errors"].append(error)
                continue
            
            # Check if process is running
            pid = data.get("pid")
            if not pid or not check_process_running(pid):
                # Stale PID - remove the file
                logger.info(f"Removing stale PID file {pid_file} (PID {pid} not running)")
                try:
                    pid_file.unlink()
                    result["removed"].append(str(pid_file))
                except Exception as e:
                    error = f"Failed to remove stale PID file {pid_file}: {e}"
                    logger.error(error)
                    result["errors"].append(error)
                continue
            
            # Process is running - check format
            if data.get("legacy_format"):
                # Legacy format - upgrade to JSON
                logger.info(f"Fixing legacy PID file {pid_file}")
                if fix_pid_file(pid_file):
                    result["fixed"].append(str(pid_file))
                else:
                    result["errors"].append(f"Failed to fix legacy PID file {pid_file}")
            else:
                logger.info(f"PID file {pid_file} is already in correct format")
        except Exception as e:
            error = f"Error processing PID file {pid_file}: {e}"
            logger.error(f"{error}\n{traceback.format_exc()}")
            result["errors"].append(error)
    
    return result

def scan_ports():
    """
    Scan common ports used in the project for active services.
    
    Returns:
        dict: Dictionary with information about active ports
    """
    result = {
        "active_ports": {},
        "port_conflicts": []
    }
    
    # Common ports used in the project
    ports_to_check = [
        (8001, "FastAPI Inference Server"),
        (7861, "Inference UI"),
        (7862, "Data Collection UI"),
        (7863, "Dashboard UI"),
        (7864, "Alternate Data Collection UI"),
        (7865, "Alternate Inference UI")
    ]
    
    # Check each port
    for port, service_name in ports_to_check:
        in_use = is_port_in_use(port)
        
        # Find which process is using this port
        process_info = None
        if in_use:
            try:
                # Try to find the process using this port
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    for conn in proc.info.get('connections', []):
                        if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port') and conn.laddr.port == port:
                            process_info = {
                                "pid": proc.info['pid'],
                                "name": proc.info['name'],
                                "cmdline": " ".join(proc.info.get('cmdline', []))
                            }
                            break
                    if process_info:
                        break
            except Exception as e:
                logger.error(f"Error finding process for port {port}: {e}")
        
        result["active_ports"][port] = {
            "port": port,
            "service_name": service_name,
            "in_use": in_use,
            "process": process_info
        }
    
    # Check for port conflicts (multiple PIDs claiming the same port)
    pid_files = [
        (INFERENCE_PROCESS_PID_FILE, 8001, "FastAPI Inference Server"),
        (INFERENCE_UI_PID_FILE, 7861, "Inference UI"),
        (DATA_COLLECTION_PID_FILE, 7862, "Data Collection UI")
    ]
    
    for pid_file, expected_port, service_name in pid_files:
        if not pid_file.exists():
            continue
            
        data = read_pid_file(pid_file)
        if not data or "pid" not in data:
            continue
            
        pid = data["pid"]
        port = data.get("port", expected_port)
        
        # Get information about the process
        process_info = None
        try:
            if check_process_running(pid):
                process = psutil.Process(pid)
                process_info = {
                    "pid": pid,
                    "name": process.name(),
                    "cmdline": " ".join(process.cmdline()),
                    "status": process.status()
                }
        except Exception as e:
            logger.error(f"Error getting process info for PID {pid}: {e}")
        
        # Compare with port scan results
        port_status = result["active_ports"].get(port, {})
        if port_status.get("in_use") and port_status.get("process") and port_status["process"]["pid"] != pid:
            # Conflict detected - two different processes claiming the same port
            result["port_conflicts"].append({
                "port": port,
                "service_name": service_name,
                "pid_file": str(pid_file),
                "pid_from_file": pid,
                "pid_using_port": port_status["process"]["pid"],
                "process_from_pid_file": process_info,
                "process_using_port": port_status["process"]
            })
    
    return result

def fix_port_conflicts():
    """
    Resolve port conflicts by killing competing processes.
    
    Returns:
        dict: Summary of actions taken
    """
    result = {
        "conflicts_resolved": [],
        "errors": []
    }
    
    # Get current port state
    scan_result = scan_ports()
    
    # Fix conflicts one by one
    for conflict in scan_result["port_conflicts"]:
        port = conflict["port"]
        pid_file = Path(conflict["pid_file"])
        pid_from_file = conflict["pid_from_file"]
        pid_using_port = conflict["pid_using_port"]
        
        logger.info(f"Resolving conflict for port {port}:")
        logger.info(f"- PID file {pid_file} contains PID {pid_from_file}")
        logger.info(f"- Port {port} is being used by PID {pid_using_port}")
        
        # Determine which process to keep
        # Strategy: Keep the process that's actually using the port
        if pid_from_file != pid_using_port:
            # The PID in the file is not the one using the port
            logger.info(f"PID file contains incorrect process - fixing...")
            
            # Check if either process is dead
            pid_file_alive = check_process_running(pid_from_file)
            port_user_alive = check_process_running(pid_using_port)
            
            if not pid_file_alive and not port_user_alive:
                # Both processes are dead - remove the PID file
                logger.info(f"Both processes are dead - removing PID file")
                try:
                    pid_file.unlink()
                    result["conflicts_resolved"].append({
                        "port": port,
                        "action": "removed_pid_file",
                        "reason": "both_processes_dead"
                    })
                except Exception as e:
                    error = f"Failed to remove PID file {pid_file}: {e}"
                    logger.error(error)
                    result["errors"].append(error)
            elif not pid_file_alive:
                # Process in PID file is dead, but port user is alive
                logger.info(f"Process in PID file is dead - updating PID file to current port user")
                try:
                    # Update PID file with the process actually using the port
                    process = psutil.Process(pid_using_port)
                    new_data = {
                        "pid": pid_using_port,
                        "port": port,
                        "timestamp": datetime.now().isoformat(),
                        "command": " ".join(process.cmdline()),
                        "fixed_by": "port_conflict_resolution"
                    }
                    with open(pid_file, 'w') as f:
                        json.dump(new_data, f, indent=2)
                        
                    result["conflicts_resolved"].append({
                        "port": port,
                        "action": "updated_pid_file",
                        "old_pid": pid_from_file,
                        "new_pid": pid_using_port
                    })
                except Exception as e:
                    error = f"Failed to update PID file {pid_file}: {e}"
                    logger.error(error)
                    result["errors"].append(error)
            elif not port_user_alive:
                # Process using port is dead, but PID file process is alive
                logger.info(f"Process using port is dead but PID file process is alive - unusual situation")
                # This shouldn't happen since we detected a conflict, 
                # but we'll handle it just in case
                result["conflicts_resolved"].append({
                    "port": port,
                    "action": "no_action",
                    "reason": "port_user_died_during_check"
                })
            else:
                # Both processes are alive - conflict!
                logger.info(f"Both processes are alive - terminating PID file process and updating PID file")
                try:
                    # Terminate the process from the PID file
                    process = psutil.Process(pid_from_file)
                    process.terminate()
                    
                    # Wait for it to terminate
                    try:
                        process.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        logger.warning(f"Process {pid_from_file} did not terminate gracefully, using kill")
                        process.kill()
                    
                    # Update PID file with the process actually using the port
                    port_user_process = psutil.Process(pid_using_port)
                    new_data = {
                        "pid": pid_using_port,
                        "port": port,
                        "timestamp": datetime.now().isoformat(),
                        "command": " ".join(port_user_process.cmdline()),
                        "fixed_by": "port_conflict_resolution"
                    }
                    with open(pid_file, 'w') as f:
                        json.dump(new_data, f, indent=2)
                        
                    result["conflicts_resolved"].append({
                        "port": port,
                        "action": "terminated_pid_file_process_and_updated",
                        "terminated_pid": pid_from_file,
                        "kept_pid": pid_using_port
                    })
                except Exception as e:
                    error = f"Failed to resolve conflict for port {port}: {e}"
                    logger.error(error)
                    result["errors"].append(error)
    
    return result

if __name__ == "__main__":
    logger.info("Starting port and PID file fixer utility")
    
    # Fix PID files
    logger.info("Fixing PID files...")
    pid_file_result = fix_all_pid_files()
    logger.info(f"Fixed {len(pid_file_result['fixed'])} PID files")
    logger.info(f"Removed {len(pid_file_result['removed'])} PID files")
    if pid_file_result['errors']:
        logger.warning(f"Encountered {len(pid_file_result['errors'])} errors while fixing PID files")
    
    # Scan ports
    logger.info("Scanning ports...")
    port_scan = scan_ports()
    active_ports = sum(1 for port_info in port_scan["active_ports"].values() if port_info["in_use"])
    logger.info(f"Found {active_ports} active ports")
    logger.info(f"Detected {len(port_scan['port_conflicts'])} port conflicts")
    
    # Fix port conflicts
    if port_scan['port_conflicts']:
        logger.info("Resolving port conflicts...")
        fix_result = fix_port_conflicts()
        logger.info(f"Resolved {len(fix_result['conflicts_resolved'])} conflicts")
        if fix_result['errors']:
            logger.warning(f"Encountered {len(fix_result['errors'])} errors while fixing port conflicts")
    else:
        logger.info("No port conflicts to resolve")
    
    # Print summary
    logger.info("\nSummary:")
    logger.info(f"- PID files fixed: {len(pid_file_result['fixed'])}")
    logger.info(f"- PID files removed: {len(pid_file_result['removed'])}")
    logger.info(f"- Port conflicts resolved: {len(port_scan['port_conflicts'])}")
    
    total_errors = len(pid_file_result['errors'])
    if port_scan['port_conflicts']:
        total_errors += len(fix_result.get('errors', []))
        
    if total_errors:
        logger.warning(f"- Total errors: {total_errors}")
        sys.exit(1)
    else:
        logger.info("- No errors encountered")
        sys.exit(0)