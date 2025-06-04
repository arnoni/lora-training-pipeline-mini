#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/fix_gradio_issue.py

"""
Gradio Process Management Fix Utility

This script resolves issues with multiple Gradio UIs starting
and the auto-cleanup logic terminating active processes.

Key fixes:
1. Proper PID file management with process verification
2. Smarter process selection based on health checks
3. Improved cleanup that preserves responsive UIs
"""

import os
import sys
import json
import time
import psutil
import socket
import requests
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gradio_fix.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('gradio_fix')

# Constants
DATA_COLLECTION_PID_FILE = Path('./data_collection_ui.pid')
INFERENCE_UI_PID_FILE = Path('./inference_ui.pid')
DEFAULT_DATA_COLLECTION_PORT = 7862
DEFAULT_INFERENCE_UI_PORT = 7861
PROCESS_TIMEOUT_SECONDS = 10

def is_port_in_use(port, max_attempts=2, delay_between_attempts=0.5):
    """
    Check if a port is already in use.
    
    Args:
        port: The port number to check
        max_attempts: Maximum number of connection attempts
        delay_between_attempts: Delay in seconds between attempts
        
    Returns:
        bool: True if port is in use, False otherwise
    """
    # Validate port parameter
    try:
        port = int(port)
        if port < 1 or port > 65535:
            logger.error(f"Invalid port number: {port} (must be 1-65535)")
            # Default to assuming it's in use for safety
            return True
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid port parameter: {e}")
        logger.error(traceback.format_exc())
        # Default to assuming it's in use for safety
        return True
        
    logger.info(f"Checking if port {port} is in use... (attempts={max_attempts})")
    
    for attempt in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                result = s.connect_ex(('localhost', port))
                port_in_use = result == 0
                
                # Log detailed result
                if port_in_use:
                    logger.info(f"Port {port} is IN USE (code: {result}, attempt {attempt+1}/{max_attempts})")
                    return True
                else:
                    logger.info(f"Port {port} appears AVAILABLE (code: {result}, attempt {attempt+1}/{max_attempts})")
                    
                    # For better reliability, double-check with a second attempt after a short delay
                    if attempt < max_attempts - 1:
                        logger.debug(f"Waiting {delay_between_attempts}s before verification attempt...")
                        time.sleep(delay_between_attempts)
                        continue
                    else:
                        # If we reach the last attempt and it's still available, confirm it's free
                        return False
                        
        except socket.error as e:
            logger.error(f"Socket error checking port {port} (attempt {attempt+1}/{max_attempts}): {e}")
            logger.error(traceback.format_exc())
            if attempt < max_attempts - 1:
                time.sleep(delay_between_attempts)
            else:
                # If we can't check reliably after all attempts, assume it might be in use to be safe
                return True
        except Exception as e:
            logger.error(f"Unexpected error checking port {port}: {e}")
            logger.error(traceback.format_exc())
            # If we can't check reliably, assume it might be in use to be safe
            return True
            
    # Default assumption (should not reach here, but just in case)
    logger.warning("Port check logic reached unexpected path - assuming port is in use")
    return True

def is_process_alive(pid):
    """
    Check if a process with given PID is alive.
    
    Args:
        pid: Process ID to check
        
    Returns:
        bool: True if process is alive, False otherwise
    """
    # Validate pid parameter
    try:
        pid = int(pid)
        if pid <= 0:
            logger.error(f"Invalid process ID: {pid} (must be positive)")
            return False
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid process ID parameter: {e}")
        logger.error(traceback.format_exc())
        return False
        
    try:
        process = psutil.Process(pid)
        
        # Check running status
        is_running = process.is_running()
        
        # Check for zombie status
        status = "UNKNOWN"
        try:
            status = process.status()
            is_zombie = status == psutil.STATUS_ZOMBIE
        except Exception as e:
            logger.warning(f"Could not check zombie status for PID {pid}: {e}")
            is_zombie = False
            
        # Log detailed status
        if is_running and not is_zombie:
            logger.debug(f"Process {pid} is alive with status: {status}")
            return True
        elif is_running and is_zombie:
            logger.info(f"Process {pid} is a zombie (status: {status})")
            return False
        else:
            logger.info(f"Process {pid} is not running")
            return False
            
    except psutil.NoSuchProcess:
        logger.info(f"Process {pid} does not exist")
        return False
    except psutil.AccessDenied:
        logger.warning(f"Access denied checking process {pid}")
        # Conservatively assume it's still running if we can't check
        return True
    except Exception as e:
        logger.error(f"Unexpected error checking process {pid}: {e}")
        logger.error(traceback.format_exc())
        # Default to assuming not running in case of unknown errors
        return False

def check_ui_responsiveness(port, timeout=3, max_attempts=2, delay_between_attempts=1.0, gradio_markers=None):
    """
    Check if a Gradio UI is responsive on the given port.
    
    Args:
        port: The port number to check
        timeout: Timeout in seconds for each request
        max_attempts: Maximum number of connection attempts
        delay_between_attempts: Delay in seconds between attempts
        gradio_markers: List of strings to identify a Gradio UI (defaults to standard markers)
        
    Returns:
        bool: True if UI is responsive, False otherwise
    """
    # Set default Gradio markers if none provided
    if gradio_markers is None:
        gradio_markers = ['gradio', '<title>gradio</title>', 'gr-app', 'webui', 'window.gradio']
    
    # Validate port parameter
    try:
        port = int(port)
        if port < 1 or port > 65535:
            logger.error(f"Invalid port number: {port} (must be 1-65535)")
            logger.debug(f"Port validation failed for {port}")
            return False
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid port parameter: {e}")
        logger.error(traceback.format_exc())
        return False
        
    # Validate timeout parameter
    try:
        timeout = float(timeout)
        if timeout <= 0:
            logger.error(f"Invalid timeout value: {timeout} (must be positive)")
            timeout = 3.0  # Fall back to default
            logger.debug(f"Using default timeout of {timeout}s after validation failure")
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid timeout parameter: {e}")
        logger.error(traceback.format_exc())
        timeout = 3.0  # Fall back to default
        
    # Validate max_attempts parameter
    try:
        max_attempts = int(max_attempts)
        if max_attempts <= 0:
            logger.error(f"Invalid max_attempts value: {max_attempts} (must be positive)")
            max_attempts = 2  # Fall back to default
            logger.debug(f"Using default max_attempts of {max_attempts} after validation failure")
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid max_attempts parameter: {e}")
        logger.error(traceback.format_exc())
        max_attempts = 2  # Fall back to default
        
    url = f"http://localhost:{port}"
    logger.info(f"Checking UI responsiveness at {url} (timeout={timeout}s, attempts={max_attempts})")
    
    # Track overall success across attempts
    last_error = None
    last_content = None
    last_status_code = None
    
    # Implement exponential backoff for retries
    backoff_factor = 1.5
    
    for attempt in range(max_attempts):
        current_delay = delay_between_attempts * (backoff_factor ** attempt)
        logger.debug(f"Attempt {attempt+1}/{max_attempts} with timeout={timeout}s")
        
        try:
            start_time = time.time()
            
            # Create a session with additional headers to mimic a browser
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gradio Health Check',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            })
            
            # Send request with session
            response = session.get(url, timeout=timeout)
            response_time = time.time() - start_time
            
            # Log response details for debugging
            response_size = len(response.content)
            response_type = response.headers.get('Content-Type', 'unknown')
            logger.debug(f"Response details: status={response.status_code}, " 
                        f"type={response_type}, size={response_size} bytes, "
                        f"time={response_time:.2f}s")
            
            # Store for diagnostics
            last_status_code = response.status_code
            
            if response.status_code == 200:
                # Verify this is actually a Gradio UI by checking for common Gradio markers in response
                response_text = response.text.lower()
                last_content = response_text[:300] + "..." if len(response_text) > 300 else response_text
                
                # Check for multiple markers to confirm it's a Gradio UI
                found_markers = []
                for marker in gradio_markers:
                    if marker.lower() in response_text:
                        found_markers.append(marker)
                
                if found_markers:
                    logger.info(f"UI is confirmed responsive on port {port} "
                              f"(response time: {response_time:.2f}s, found markers: {', '.join(found_markers)})")
                    # Additional verification of key Gradio elements
                    if 'streamlit' in response_text and 'gradio' not in response_text:
                        logger.warning(f"Port {port} may be a Streamlit app, not Gradio")
                    return True
                else:
                    logger.warning(f"Port {port} responded but content doesn't appear to be a Gradio UI")
                    logger.debug(f"Response start: {response_text[:200]}...")
                    # Continue to next attempt as this might be a different service
            else:
                logger.warning(f"UI on port {port} returned status code {response.status_code} (attempt {attempt+1}/{max_attempts})")
                # Additional debug info for non-200 responses
                if response.status_code >= 300 and response.status_code < 400:
                    logger.debug(f"Redirect detected to: {response.headers.get('Location', 'unknown')}")
                elif response.status_code >= 400:
                    logger.debug(f"Error response: {response.text[:200]}...")
                
                # Try again for non-200 responses
            
            # If we're not on the last attempt, wait before retrying
            if attempt < max_attempts - 1:
                logger.debug(f"Waiting {current_delay:.1f}s before next attempt...")
                time.sleep(current_delay)
                
        except requests.ConnectionError as e:
            logger.warning(f"Connection error checking UI on port {port} (attempt {attempt+1}/{max_attempts}): {e}")
            last_error = f"Connection error: {str(e)}"
            
            # Try quick socket check to verify port is even open
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    result = s.connect_ex(('localhost', port))
                    if result != 0:
                        logger.debug(f"Port {port} is not accepting connections (socket error code: {result})")
                    else:
                        logger.debug(f"Port {port} is accepting socket connections but HTTP failed")
            except Exception as socket_err:
                logger.debug(f"Socket check failed: {socket_err}")
                
            if attempt < max_attempts - 1:
                logger.debug(f"Waiting {current_delay:.1f}s before next attempt...")
                time.sleep(current_delay)
                
        except requests.Timeout as e:
            logger.warning(f"Timeout checking UI on port {port} (attempt {attempt+1}/{max_attempts}): {e}")
            last_error = f"Request timeout after {timeout}s"
            
            if attempt < max_attempts - 1:
                # Increase timeout for next attempt
                timeout = min(timeout * 1.5, 10.0)  # Max 10 seconds
                logger.debug(f"Increasing timeout to {timeout:.1f}s for next attempt")
                logger.debug(f"Waiting {current_delay:.1f}s before next attempt...")
                time.sleep(current_delay)
                
        except requests.RequestException as e:
            logger.warning(f"Request error checking UI on port {port} (attempt {attempt+1}/{max_attempts}): {e}")
            last_error = f"Request error: {str(e)}"
            
            if attempt < max_attempts - 1:
                logger.debug(f"Waiting {current_delay:.1f}s before next attempt...")
                time.sleep(current_delay)
                
        except Exception as e:
            # Catch all other exceptions to prevent silent failures
            logger.error(f"Unexpected error checking UI on port {port}: {e}")
            logger.error(traceback.format_exc())
            last_error = f"Unexpected error: {str(e)}"
            # Don't retry on unexpected errors
            break
    
    # If we get here, all attempts failed - log comprehensive diagnostics
    logger.warning(f"UI on port {port} is not responsive after {max_attempts} attempts")
    logger.debug(f"Last error: {last_error}")
    logger.debug(f"Last status code: {last_status_code}")
    if last_content:
        logger.debug(f"Last content preview: {last_content[:150]}...")
        
    # Run a final OS-level check to see if anything is listening
    try:
        if sys.platform == 'win32':
            netstat_output = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()
            logger.debug(f"Port {port} netstat check: {netstat_output.strip()}")
        elif sys.platform in ['linux', 'linux2', 'darwin']:
            netstat_output = subprocess.check_output(f'netstat -tuln | grep :{port}', shell=True).decode()
            logger.debug(f"Port {port} netstat check: {netstat_output.strip()}")
    except subprocess.CalledProcessError as netstat_proc_err:
        print(f"[DEBUG] netstat subprocess error type: {type(netstat_proc_err).__name__}")
        print(f"[DEBUG] netstat subprocess error details: {netstat_proc_err}")
        logger.debug(f"No process found listening on port {port} via netstat")
    except Exception as netstat_err:
        print(f"[DEBUG] netstat error type: {type(netstat_err).__name__}")
        print(f"[DEBUG] netstat error details: {netstat_err}")
        logger.debug(f"Error running netstat check: {netstat_err}")
    
    return False

def find_gradio_processes(service_type=None):
    """
    Find all Gradio processes of a specific type.
    
    Args:
        service_type: 'data_collection' or 'inference_ui' or None for all
        
    Returns:
        list: List of process information dictionaries
    """
    gradio_processes = []
    
    # Go through all Python processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            # Filter for Python processes
            if "python" not in proc.info['name'].lower():
                continue
                
            cmdline = " ".join(proc.info.get('cmdline', []))
            
            # Check if this is a Gradio process of the right type
            if "gradio" not in cmdline:
                continue
                
            is_data_collection = "data_collection" in cmdline
            is_inference_ui = "inference" in cmdline and not is_data_collection
            
            # Filter by service type
            if service_type == "data_collection" and not is_data_collection:
                continue
            elif service_type == "inference_ui" and not is_inference_ui:
                continue
                
            # Get process type
            process_type = "data_collection" if is_data_collection else "inference_ui" if is_inference_ui else "unknown"
            
            # Try to extract the port
            port = None
            if "GRADIO_PORT=" in cmdline:
                parts = cmdline.split("GRADIO_PORT=")[1].split()
                if parts:
                    try:
                        port = int(parts[0])
                    except ValueError as port_err:
                        print(f"[DEBUG] Error parsing port from command line '{parts[0]}': {type(port_err).__name__}: {port_err}")
                        pass
                        
            if port is None:
                # Use default ports if not specified
                if process_type == "data_collection":
                    port = DEFAULT_DATA_COLLECTION_PORT
                elif process_type == "inference_ui":
                    port = DEFAULT_INFERENCE_UI_PORT
            
            # Check if the process is responsive
            is_responsive = False
            if is_process_alive(proc.info['pid']):
                is_responsive = check_ui_responsiveness(port)
            
            # Create process info
            process_info = {
                "pid": proc.info['pid'],
                "type": process_type,
                "cmdline": cmdline,
                "create_time": proc.info['create_time'],
                "alive": is_process_alive(proc.info['pid']),
                "port": port,
                "responsive": is_responsive
            }
            
            gradio_processes.append(process_info)
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception as e:
            logger.error(f"Error processing a process: {e}")
            
    return gradio_processes

def get_pid_file_info(pid_file):
    """
    Get information from a PID file.
    
    Args:
        pid_file: Path object or string path to PID file
        
    Returns:
        dict: PID file information or None if invalid
    """
    # Validate and convert pid_file to Path object
    try:
        if not isinstance(pid_file, Path):
            pid_file = Path(pid_file)
        
        # Log the file we're checking
        logger.debug(f"Checking PID file: {pid_file.absolute()}")
            
        if not pid_file.exists():
            logger.info(f"PID file does not exist: {pid_file}")
            return None
            
        if not pid_file.is_file():
            logger.warning(f"PID file path exists but is not a file: {pid_file}")
            return None
    except Exception as e:
        logger.error(f"Invalid PID file path: {e}")
        logger.error(traceback.format_exc())
        return None
        
    # Try multiple encodings if needed
    encodings = ['utf-8', 'latin-1', 'ascii']
    content = None
    
    for encoding in encodings:
        try:
            with open(pid_file, 'r', encoding=encoding) as f:
                content = f.read().strip()
            break  # Break if successful
        except UnicodeDecodeError:
            logger.debug(f"Failed to read with {encoding} encoding, trying next...")
            continue
        except Exception as e:
            logger.error(f"Error reading PID file {pid_file} with {encoding} encoding: {e}")
            logger.error(traceback.format_exc())
            return None
            
    if content is None:
        logger.error(f"Could not read PID file with any encoding: {pid_file}")
        return None
            
    if not content:
        logger.warning(f"PID file is empty: {pid_file}")
        return None
        
    # Try JSON format first
    try:
        data = json.loads(content)
        
        # Validate "pid" field
        if "pid" in data:
            try:
                pid = int(data["pid"])
                if pid <= 0:
                    logger.warning(f"Invalid PID value in JSON: {pid}")
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
                            port = None
                    except (TypeError, ValueError):
                        logger.warning(f"Non-integer port in JSON: {data['port']}")
                        port = None
                        
                return {
                    "pid": pid,
                    "port": port,
                    "format": "json",
                    "timestamp": data.get("timestamp"),
                    "updated_by": data.get("updated_by")
                }
            except (TypeError, ValueError):
                logger.warning(f"Non-integer PID in JSON: {data['pid']}")
                return None
        else:
            logger.warning(f"JSON missing 'pid' field: {data}")
            return None
            
    except json.JSONDecodeError:
        # Try simple integer format
        try:
            pid = int(content)
            if pid <= 0:
                logger.warning(f"Invalid PID value: {pid}")
                return None
                
            logger.info(f"Valid integer PID file: {pid}")
            return {
                "pid": pid,
                "format": "integer"
            }
        except ValueError:
            # Log first part of content for debugging
            safe_content = content[:50] + ("..." if len(content) > 50 else "")
            logger.warning(f"Invalid PID file format (not JSON or integer): {safe_content}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error parsing PID file {pid_file}: {e}")
        logger.error(traceback.format_exc())
        return None

def write_pid_file(pid_file, pid, port=None):
    """
    Write PID file in standardized JSON format.
    
    Args:
        pid_file: Path object or string path to PID file
        pid: Process ID to write
        port: Optional port number associated with the process
        
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
            "updated_by": "fix_gradio_issue.py"
        }
        
        if port is not None:
            data["port"] = port
            
        # Create temporary file first to ensure atomic write
        temp_file = pid_file.with_suffix('.tmp')
        
        # Write to temporary file
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        # Rename temporary file to target file (atomic operation)
        temp_file.replace(pid_file)
            
        logger.info(f"Updated PID file {pid_file} with PID {pid}" + 
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

def fix_gradio_processes(service_type=None, keep_responsive=True, force_termination=False):
    """
    Fix issues with Gradio processes.
    
    Args:
        service_type: 'data_collection', 'inference_ui', or None for both
        keep_responsive: Whether to keep responsive processes
        force_termination: Force termination of problematic processes
        
    Returns:
        dict: Results of the operation
    """
    result = {
        "found_processes": 0,
        "kept_processes": [],
        "terminated_processes": [],
        "errors": []
    }
    
    # Find Gradio processes
    processes = find_gradio_processes(service_type)
    result["found_processes"] = len(processes)
    
    if not processes:
        logger.info(f"No Gradio {service_type or 'UI'} processes found")
        return result
        
    # Group by type
    processes_by_type = {}
    for proc in processes:
        proc_type = proc["type"]
        if proc_type not in processes_by_type:
            processes_by_type[proc_type] = []
            
        processes_by_type[proc_type].append(proc)
        
    # Process each type
    for proc_type, type_processes in processes_by_type.items():
        logger.info(f"Found {len(type_processes)} {proc_type} processes")
        
        # Skip if there's only one process of this type
        if len(type_processes) <= 1:
            only_proc = type_processes[0]
            result["kept_processes"].append({
                "pid": only_proc["pid"],
                "type": proc_type,
                "port": only_proc["port"],
                "reason": "only_process"
            })
            continue
            
        # Get relevant PID file
        pid_file = DATA_COLLECTION_PID_FILE if proc_type == "data_collection" else INFERENCE_UI_PID_FILE
        pid_file_info = get_pid_file_info(pid_file)
        
        # Find responsive processes
        responsive_processes = [p for p in type_processes if p["responsive"]]
        
        # If we have responsive processes and should keep them
        if responsive_processes and keep_responsive:
            # Keep the most recently created responsive process
            responsive_processes.sort(key=lambda p: p["create_time"], reverse=True)
            keep_process = responsive_processes[0]
            
            # Update PID file
            write_pid_file(pid_file, keep_process["pid"], keep_process["port"])
            
            # Add to kept processes
            result["kept_processes"].append({
                "pid": keep_process["pid"],
                "type": proc_type,
                "port": keep_process["port"],
                "reason": "responsive_process"
            })
            
            # Terminate other processes
            for proc in type_processes:
                if proc["pid"] != keep_process["pid"]:
                    try:
                        if proc["alive"]:
                            logger.info(f"Terminating {proc_type} process {proc['pid']} (port {proc['port']})")
                            process = psutil.Process(proc["pid"])
                            process.terminate()
                            
                            # Wait for termination
                            try:
                                process.wait(timeout=PROCESS_TIMEOUT_SECONDS)
                            except psutil.TimeoutExpired:
                                if force_termination:
                                    logger.warning(f"Process {proc['pid']} did not terminate in time, killing")
                                    process.kill()
                                else:
                                    logger.warning(f"Process {proc['pid']} did not terminate in time, leaving running")
                                    result["errors"].append(f"Process {proc['pid']} did not terminate in time")
                                    continue
                                    
                            result["terminated_processes"].append({
                                "pid": proc["pid"],
                                "type": proc_type,
                                "port": proc["port"]
                            })
                        else:
                            logger.info(f"Process {proc['pid']} is already not running")
                    except Exception as e:
                        error_msg = f"Error terminating process {proc['pid']}: {e}"
                        logger.error(error_msg)
                        result["errors"].append(error_msg)
        
        # If no responsive processes or not keeping responsive ones
        elif type_processes:
            # Keep the most recently created process
            type_processes.sort(key=lambda p: p["create_time"], reverse=True)
            keep_process = type_processes[0]
            
            # If it's not responsive, try to restart it
            if not keep_process["responsive"] and keep_process["alive"]:
                logger.warning(f"Process {keep_process['pid']} is not responsive, considering restart")
                
                # For now, we'll keep it anyway but note the issue
                result["errors"].append(f"Kept non-responsive process {keep_process['pid']} (port {keep_process['port']})")
            
            # Update PID file
            write_pid_file(pid_file, keep_process["pid"], keep_process["port"])
            
            # Add to kept processes
            result["kept_processes"].append({
                "pid": keep_process["pid"],
                "type": proc_type,
                "port": keep_process["port"],
                "reason": "newest_process"
            })
            
            # Terminate other processes
            for proc in type_processes:
                if proc["pid"] != keep_process["pid"]:
                    try:
                        if proc["alive"]:
                            logger.info(f"Terminating {proc_type} process {proc['pid']} (port {proc['port']})")
                            process = psutil.Process(proc["pid"])
                            process.terminate()
                            
                            # Wait for termination
                            try:
                                process.wait(timeout=PROCESS_TIMEOUT_SECONDS)
                            except psutil.TimeoutExpired:
                                if force_termination:
                                    logger.warning(f"Process {proc['pid']} did not terminate in time, killing")
                                    process.kill()
                                else:
                                    logger.warning(f"Process {proc['pid']} did not terminate in time, leaving running")
                                    result["errors"].append(f"Process {proc['pid']} did not terminate in time")
                                    continue
                                    
                            result["terminated_processes"].append({
                                "pid": proc["pid"],
                                "type": proc_type,
                                "port": proc["port"]
                            })
                        else:
                            logger.info(f"Process {proc['pid']} is already not running")
                    except Exception as e:
                        error_msg = f"Error terminating process {proc['pid']}: {e}"
                        logger.error(error_msg)
                        result["errors"].append(error_msg)
    
    return result

def ensure_gradio_ui_running(service_type, script_path=None, port=None):
    """
    Ensure a Gradio UI is running, starting it if needed.
    
    Args:
        service_type: 'data_collection' or 'inference_ui'
        script_path: Path to the script to run (if None, use default)
        port: Port to use (if None, use default)
        
    Returns:
        dict: Result of the operation
    """
    result = {
        "status": "unknown",
        "pid": None,
        "port": None,
        "error": None
    }
    
    # Set defaults based on service type
    if service_type == "data_collection":
        pid_file = DATA_COLLECTION_PID_FILE
        default_port = DEFAULT_DATA_COLLECTION_PORT
        default_script = "src.lora_training_pipeline.data_collection.gradio_app"
    elif service_type == "inference_ui":
        pid_file = INFERENCE_UI_PID_FILE
        default_port = DEFAULT_INFERENCE_UI_PORT
        default_script = "src.lora_training_pipeline.inference.gradio_inference"
    else:
        result["status"] = "error"
        result["error"] = f"Invalid service type: {service_type}"
        return result
        
    port = port or default_port
    script_path = script_path or default_script
    
    # Check if service is already running
    pid_file_info = get_pid_file_info(pid_file)
    if pid_file_info and pid_file_info["pid"]:
        # Check if the process is alive
        if is_process_alive(pid_file_info["pid"]):
            # Check if it's responsive
            proc_port = pid_file_info.get("port", port)
            if check_ui_responsiveness(proc_port):
                # Process is running and responsive
                logger.info(f"{service_type} is already running (PID: {pid_file_info['pid']}, port: {proc_port})")
                result["status"] = "already_running"
                result["pid"] = pid_file_info["pid"]
                result["port"] = proc_port
                return result
            else:
                # Process is running but not responsive
                logger.warning(f"{service_type} is running (PID: {pid_file_info['pid']}) but not responsive on port {proc_port}")
                # Continue to start a new one after terminating this one
                try:
                    process = psutil.Process(pid_file_info["pid"])
                    process.terminate()
                    try:
                        process.wait(timeout=PROCESS_TIMEOUT_SECONDS)
                    except psutil.TimeoutExpired:
                        logger.warning(f"Process {pid_file_info['pid']} did not terminate in time, killing")
                        process.kill()
                except Exception as e:
                    logger.error(f"Error terminating unresponsive process {pid_file_info['pid']}: {e}")
        else:
            # Process is not running, remove PID file
            logger.info(f"PID file points to non-existent process, removing it")
            try:
                pid_file.unlink()
            except Exception as e:
                logger.error(f"Error removing stale PID file: {e}")
    
    # Check if port is in use
    if is_port_in_use(port):
        result["status"] = "port_in_use"
        result["error"] = f"Port {port} is already in use by another process"
        logger.error(result["error"])
        return result
        
    # Start the UI
    logger.info(f"Starting {service_type} on port {port}")
    
    # Set environment variables
    env = os.environ.copy()
    env["GRADIO_PORT"] = str(port)
    
    try:
        # Start process
        process = subprocess.Popen(
            [sys.executable, "-m", script_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for it to start
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is not None:
            # Process exited
            try:
                stdout, stderr = process.communicate()
                stderr_decoded = stderr.decode()[:500] if stderr else "No error output"
                stdout_decoded = stdout.decode()[:500] if stdout else "No output"
            except Exception as comm_err:
                print(f"[DEBUG] Process communication error type: {type(comm_err).__name__}")
                print(f"[DEBUG] Process communication error details: {comm_err}")
                stderr_decoded = "Failed to decode error output"
                stdout_decoded = "Failed to decode output"
            
            result["status"] = "start_failed"
            result["error"] = f"Process exited with code {process.returncode}: {stderr_decoded}"
            logger.error(result["error"])
            logger.debug(f"Process stdout: {stdout_decoded}")
            return result
            
        # Update PID file
        write_pid_file(pid_file, process.pid, port)
        
        # Wait a bit longer for UI to initialize
        time.sleep(8)
        
        # Check if it's responsive
        if check_ui_responsiveness(port):
            result["status"] = "started"
            result["pid"] = process.pid
            result["port"] = port
            logger.info(f"{service_type} started successfully (PID: {process.pid}, port: {port})")
        else:
            result["status"] = "started_not_responsive"
            result["pid"] = process.pid
            result["port"] = port
            result["error"] = f"UI started but not responsive on port {port}"
            logger.warning(result["error"])
            
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Error starting {service_type}: {e}"
        logger.error(f"{result['error']}\n{traceback.format_exc()}")
        return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix Gradio UI issues")
    parser.add_argument("--fix", action="store_true", help="Fix Gradio process issues")
    parser.add_argument("--info", action="store_true", help="Show information about running Gradio processes")
    parser.add_argument("--force", action="store_true", help="Force termination of problematic processes")
    parser.add_argument("--type", choices=["data_collection", "inference_ui"], help="Service type to manage")
    parser.add_argument("--ensure", action="store_true", help="Ensure service is running, starting if needed")
    parser.add_argument("--port", type=int, help="Port to use (with --ensure)")
    
    args = parser.parse_args()
    
    if args.info:
        print("Gathering information about Gradio processes...")
        processes = find_gradio_processes(args.type)
        
        if not processes:
            print(f"No Gradio {args.type or 'UI'} processes found")
        else:
            print(f"Found {len(processes)} Gradio {args.type or 'UI'} processes:")
            
            for proc in processes:
                status = "ALIVE" if proc["alive"] else "DEAD"
                if proc["alive"]:
                    status += " & RESPONSIVE" if proc["responsive"] else " but NOT RESPONSIVE"
                    
                print(f"- PID {proc['pid']} ({status}): {proc['type']} on port {proc['port']}")
                print(f"  Created: {datetime.fromtimestamp(proc['create_time'])}")
                print(f"  Command: {proc['cmdline'][:100]}...")
                print()
                
            # Also check PID files
            if args.type in [None, "data_collection"]:
                pid_info = get_pid_file_info(DATA_COLLECTION_PID_FILE)
                if pid_info:
                    pid = pid_info["pid"]
                    alive = is_process_alive(pid)
                    print(f"Data Collection PID file: {pid} ({'ALIVE' if alive else 'DEAD'})")
                else:
                    print("No Data Collection PID file found")
                    
            if args.type in [None, "inference_ui"]:
                pid_info = get_pid_file_info(INFERENCE_UI_PID_FILE)
                if pid_info:
                    pid = pid_info["pid"]
                    alive = is_process_alive(pid)
                    print(f"Inference UI PID file: {pid} ({'ALIVE' if alive else 'DEAD'})")
                else:
                    print("No Inference UI PID file found")
    
    elif args.fix:
        print(f"Fixing Gradio {args.type or 'UI'} processes...")
        result = fix_gradio_processes(args.type, force_termination=args.force)
        
        if result["kept_processes"]:
            print("\nKept processes:")
            for proc in result["kept_processes"]:
                print(f"- {proc['type']} (PID: {proc['pid']}, port: {proc['port']}): {proc['reason']}")
                
        if result["terminated_processes"]:
            print("\nTerminated processes:")
            for proc in result["terminated_processes"]:
                print(f"- {proc['type']} (PID: {proc['pid']}, port: {proc['port']})")
                
        if result["errors"]:
            print("\nErrors:")
            for error in result["errors"]:
                print(f"- {error}")
    
    elif args.ensure:
        if not args.type:
            print("Error: --type is required with --ensure")
            sys.exit(1)
            
        print(f"Ensuring {args.type} is running...")
        result = ensure_gradio_ui_running(args.type, port=args.port)
        
        if result["status"] == "already_running":
            print(f"{args.type} is already running (PID: {result['pid']}, port: {result['port']})")
        elif result["status"] == "started":
            print(f"{args.type} started successfully (PID: {result['pid']}, port: {result['port']})")
        elif result["status"] == "started_not_responsive":
            print(f"Warning: {args.type} started (PID: {result['pid']}, port: {result['port']}) but not responsive")
        else:
            print(f"Error: {result['error']}")
            sys.exit(1)
    
    else:
        parser.print_help()