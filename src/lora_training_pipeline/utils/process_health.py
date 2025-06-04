#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/process_health.py

import os
import json
import time
import socket
import psutil
import requests
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
import fcntl
import jsonschema
from typing import Dict, List, Optional, Union, Any, Tuple

# Enable verbose debugging
DEBUG_MODE = os.environ.get("DEBUG_PROCESS_MANAGEMENT", "true").lower() == "true"

def debug_print(*args, **kwargs):
    """Print debug information only when DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print("[PROCESS-HEALTH-DEBUG]", *args, **kwargs)

# Path constants
INFERENCE_PROCESS_PID_FILE = Path('./inference_process.pid')
DATA_COLLECTION_PID_FILE = Path('./data_collection_ui.pid')
INFERENCE_UI_PID_FILE = Path('./inference_ui.pid')
HEALTH_CHECK_LOG = Path('./process_health.log')

# Schema for validating PID file structure
PID_FILE_SCHEMA = {
    "type": "object",
    "required": ["pid"],
    "properties": {
        "pid": {"type": "integer"},
        "timestamp": {"type": "string"},
        "port": {"type": ["integer", "string"]},
        "process_type": {"type": "string"},
        "hostname": {"type": "string"}
    }
}

def log_health_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log process health events to a dedicated log file."""
    try:
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "type": event_type,
            "pid": os.getpid(),
            "details": details
        }
        
        with open(HEALTH_CHECK_LOG, 'a') as f:
            # Use fcntl to lock the file during write to prevent race conditions
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(log_entry) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                
        debug_print(f"Logged health event: {event_type}")
    except Exception as e:
        error_msg = f"Error logging health event: {e}"
        print(f"WARNING: {error_msg}")
        debug_print(f"{error_msg}\n{traceback.format_exc()}")

def read_pid_file_safely(pid_file: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Read a PID file with robust error handling and schema validation.
    
    Args:
        pid_file: Path to the PID file
        
    Returns:
        Tuple[Optional[Dict], Optional[str]]: A tuple containing:
            - The parsed PID data if successful, None otherwise
            - An error message if reading or parsing failed, None if successful
    """
    if not pid_file.exists():
        return None, f"PID file {pid_file} does not exist"
        
    # Use file locking to ensure we get a consistent read
    try:
        with open(pid_file, 'r') as f:
            # Lock the file for reading (shared lock)
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                content = f.read().strip()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                
        # Early return for empty files
        if not content:
            return None, f"PID file {pid_file} is empty"
            
        # Try parsing as JSON
        try:
            data = json.loads(content)
            
            # Validate against schema
            try:
                jsonschema.validate(instance=data, schema=PID_FILE_SCHEMA)
                return data, None
            except jsonschema.exceptions.ValidationError as ve:
                error_msg = f"Invalid PID file schema: {ve}"
                debug_print(error_msg)
                
                # Check if it at least has a PID
                if isinstance(data, dict) and "pid" in data and isinstance(data["pid"], int):
                    debug_print(f"PID file has invalid schema but contains valid PID: {data['pid']}")
                    # Still return the data but log the schema issue
                    log_health_event("PID_FILE_SCHEMA_WARNING", {
                        "pid_file": str(pid_file),
                        "validation_error": str(ve),
                        "data": data
                    })
                    return data, None
                else:
                    return None, error_msg
                    
        except json.JSONDecodeError:
            # Not JSON, try to parse as integer (legacy format)
            try:
                pid = int(content)
                debug_print(f"Converted legacy PID file {pid_file} with raw value: {pid}")
                
                # Create a compatible dictionary
                converted_data = {
                    "pid": pid,
                    "legacy_format": True,
                    "converted_at": datetime.now().isoformat()
                }
                
                # Log this legacy format detection
                log_health_event("LEGACY_PID_FILE_DETECTED", {
                    "pid_file": str(pid_file),
                    "pid": pid,
                    "converted_data": converted_data
                })
                
                return converted_data, None
            except ValueError:
                error_msg = f"PID file {pid_file} contains invalid data (not JSON or integer)"
                debug_print(error_msg)
                return None, error_msg
                
    except Exception as e:
        error_msg = f"Error reading PID file {pid_file}: {e}"
        debug_print(f"{error_msg}\n{traceback.format_exc()}")
        return None, error_msg

def is_process_alive_and_healthy(pid: int) -> bool:
    """
    Check if a process is running and responds to signals.
    
    Args:
        pid: Process ID to check
        
    Returns:
        bool: True if process is running and responsive, False otherwise
    """
    try:
        # Check if process exists
        process = psutil.Process(pid)
        
        # Check process status
        status = process.status()
        
        # Consider zombie processes as not alive
        if status == psutil.STATUS_ZOMBIE:
            debug_print(f"Process {pid} is a zombie")
            return False
            
        # Check if process responds to signals
        try:
            # Send a null signal (0) to test if process is responding
            os.kill(pid, 0)
            debug_print(f"Process {pid} is alive and responding to signals")
            return True
        except OSError:
            debug_print(f"Process {pid} does not respond to signals")
            return False
            
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        debug_print(f"Process {pid} does not exist or access denied")
        return False

def check_port_binding(port: int, timeout: float = 2.0) -> bool:
    """
    Check if a port is bound and accepting connections.
    
    Args:
        port: Port number to check
        timeout: Timeout in seconds
        
    Returns:
        bool: True if port is bound and accepting connections, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex(('localhost', port))
            bound = result == 0
            debug_print(f"Port {port} is {'bound' if bound else 'not bound'} (code: {result})")
            return bound
    except Exception as e:
        error_msg = f"Error checking port {port}: {e}"
        debug_print(error_msg)
        return False

def check_fastapi_health(port: int) -> Dict[str, Any]:
    """
    Check the health of a FastAPI server.
    
    Args:
        port: Port number the FastAPI server is running on
        
    Returns:
        dict: Health check results
    """
    result = {
        "status": "unknown",
        "port_bound": False,
        "api_reachable": False,
        "health_endpoint_working": False,
        "response_time_ms": None,
        "errors": []
    }
    
    # Check if port is bound
    result["port_bound"] = check_port_binding(port)
    if not result["port_bound"]:
        result["status"] = "port_not_bound"
        result["errors"].append(f"Port {port} is not bound")
        return result
        
    # Try to reach the API
    url = f"http://localhost:{port}"
    try:
        start_time = time.time()
        response = requests.get(url, timeout=5)
        end_time = time.time()
        
        result["response_time_ms"] = round((end_time - start_time) * 1000, 2)
        result["api_reachable"] = True
        
        # Check if we got a 200 OK or other response
        result["status_code"] = response.status_code
        if response.status_code == 200:
            debug_print(f"API at {url} returned 200 OK in {result['response_time_ms']}ms")
        else:
            debug_print(f"API at {url} returned status code {response.status_code}")
            result["errors"].append(f"API returned status code {response.status_code}")
    except requests.RequestException as e:
        result["status"] = "api_unreachable"
        result["errors"].append(f"Could not reach API at {url}: {e}")
        debug_print(f"Error reaching API at {url}: {e}")
        
    # Try the health endpoint if API is reachable
    if result["api_reachable"]:
        health_url = f"{url}/health"
        try:
            start_time = time.time()
            response = requests.get(health_url, timeout=5)
            end_time = time.time()
            
            result["health_response_time_ms"] = round((end_time - start_time) * 1000, 2)
            
            if response.status_code == 200:
                result["health_endpoint_working"] = True
                try:
                    result["health_data"] = response.json()
                except ValueError:
                    result["health_data"] = {"content": response.text[:100]}
                
                debug_print(f"Health endpoint returned OK in {result['health_response_time_ms']}ms")
                result["status"] = "healthy"
            else:
                debug_print(f"Health endpoint returned status code {response.status_code}")
                result["errors"].append(f"Health endpoint returned {response.status_code}")
                result["status"] = "health_check_failed"
        except requests.RequestException as e:
            result["status"] = "health_check_unreachable"
            result["errors"].append(f"Could not reach health endpoint: {e}")
            debug_print(f"Error reaching health endpoint: {e}")
    
    return result

def check_gradio_health(port: int) -> Dict[str, Any]:
    """
    Check the health of a Gradio UI.
    
    Args:
        port: Port number the Gradio UI is running on
        
    Returns:
        dict: Health check results
    """
    result = {
        "status": "unknown",
        "port_bound": False,
        "ui_reachable": False,
        "response_time_ms": None,
        "errors": []
    }
    
    # Check if port is bound
    result["port_bound"] = check_port_binding(port)
    if not result["port_bound"]:
        result["status"] = "port_not_bound"
        result["errors"].append(f"Port {port} is not bound")
        return result
        
    # Try to reach the UI
    url = f"http://localhost:{port}"
    try:
        start_time = time.time()
        response = requests.get(url, timeout=5)
        end_time = time.time()
        
        result["response_time_ms"] = round((end_time - start_time) * 1000, 2)
        result["ui_reachable"] = True
        
        # Check if we got a 200 OK or other response
        result["status_code"] = response.status_code
        if response.status_code == 200:
            debug_print(f"UI at {url} returned 200 OK in {result['response_time_ms']}ms")
            result["status"] = "healthy"
            
            # Check for common Gradio indicators in the response
            content = response.text.lower()
            if "gradio" in content or "<title>gradio</title>" in content or "webui" in content:
                result["detected_ui"] = "gradio"
            else:
                debug_print("Response doesn't contain Gradio indicators")
                result["detected_ui"] = "unknown"
                result["errors"].append("Response doesn't appear to be from Gradio")
        else:
            debug_print(f"UI at {url} returned status code {response.status_code}")
            result["errors"].append(f"UI returned status code {response.status_code}")
            result["status"] = "unhealthy"
    except requests.RequestException as e:
        result["status"] = "ui_unreachable"
        result["errors"].append(f"Could not reach UI at {url}: {e}")
        debug_print(f"Error reaching UI at {url}: {e}")
    
    return result

def scan_active_ports(start_port: int = 7860, end_port: int = 8010) -> List[Dict[str, Any]]:
    """
    Scan a range of ports for active services.
    
    Args:
        start_port: Starting port number
        end_port: Ending port number
        
    Returns:
        list: List of dictionaries with information about active ports
    """
    active_ports = []
    
    for port in range(start_port, end_port + 1):
        try:
            # Skip common system ports
            if port < 1024:
                continue
                
            # Check if port is bound
            if check_port_binding(port):
                port_info = {
                    "port": port,
                    "active": True
                }
                
                # Try to detect the service
                try:
                    url = f"http://localhost:{port}"
                    response = requests.get(url, timeout=1)
                    
                    # Check content to identify service
                    content = response.text.lower()
                    
                    if "gradio" in content:
                        port_info["detected_service"] = "gradio"
                    elif "fastapi" in content or "swagger" in content or "redoc" in content:
                        port_info["detected_service"] = "fastapi"
                    elif "jupyter" in content:
                        port_info["detected_service"] = "jupyter"
                    else:
                        port_info["detected_service"] = "unknown_web"
                        
                    port_info["status_code"] = response.status_code
                except requests.RequestException:
                    port_info["detected_service"] = "non_http"
                
                # Find the process using this port
                try:
                    for proc in psutil.process_iter(['pid', 'name', 'connections']):
                        for conn in proc.info['connections']:
                            if conn.laddr.port == port:
                                port_info["pid"] = proc.info['pid']
                                port_info["process_name"] = proc.info['name']
                                break
                        if "pid" in port_info:
                            break
                except Exception as e:
                    debug_print(f"Error finding process for port {port}: {e}")
                
                active_ports.append(port_info)
        except Exception as e:
            debug_print(f"Error scanning port {port}: {e}")
    
    return active_ports

def check_all_process_health() -> Dict[str, Any]:
    """
    Check the health of all managed processes.
    
    Returns:
        dict: Dictionary with health information for all processes
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "fastapi_server": None,
        "data_collection_ui": None,
        "inference_ui": None,
        "active_ports": None,
        "stale_processes": []
    }
    
    # Scan active ports for context
    try:
        result["active_ports"] = scan_active_ports()
    except Exception as e:
        debug_print(f"Error scanning active ports: {e}")
    
    # Check FastAPI server
    if INFERENCE_PROCESS_PID_FILE.exists():
        pid_data, error = read_pid_file_safely(INFERENCE_PROCESS_PID_FILE)
        
        if error:
            result["stale_processes"].append({
                "type": "fastapi",
                "pid_file": str(INFERENCE_PROCESS_PID_FILE),
                "error": error
            })
        elif pid_data:
            try:
                pid = pid_data["pid"]
                port = pid_data.get("port", 8001)
                
                # Ensure port is an integer
                if isinstance(port, str) and port.isdigit():
                    port = int(port)
                elif not isinstance(port, int):
                    port = 8001  # Use default port if invalid
                
                # Check both process and port health
                is_alive = is_process_alive_and_healthy(pid)
                api_health = check_fastapi_health(port) if is_alive else {"status": "process_not_running"}
                
                # Combine information
                result["fastapi_server"] = {
                    "pid": pid,
                    "port": port,
                    "process_alive": is_alive,
                    "api_health": api_health,
                    "metadata": pid_data
                }
                
                # Check for inconsistencies
                if is_alive and not api_health["port_bound"]:
                    # Process is running but port is not bound
                    debug_print(f"Process {pid} is running but port {port} is not bound")
                    result["stale_processes"].append({
                        "type": "fastapi",
                        "pid": pid,
                        "port": port,
                        "reason": "Process running but port not bound",
                        "metadata": pid_data
                    })
                elif not is_alive and api_health["port_bound"]:
                    # Port is bound but process is not running
                    debug_print(f"Port {port} is bound but process {pid} is not running")
                    result["stale_processes"].append({
                        "type": "fastapi",
                        "pid": pid,
                        "port": port,
                        "reason": "Port bound but process not running",
                        "metadata": pid_data
                    })
                elif not is_alive:
                    # Process is not running
                    debug_print(f"FastAPI process {pid} is not running")
                    result["stale_processes"].append({
                        "type": "fastapi",
                        "pid": pid,
                        "reason": "Process not running",
                        "metadata": pid_data
                    })
            except Exception as e:
                error_msg = f"Error checking FastAPI health: {e}"
                debug_print(f"{error_msg}\n{traceback.format_exc()}")
                result["stale_processes"].append({
                    "type": "fastapi",
                    "error": error_msg,
                    "metadata": pid_data
                })
    
    # Check Data Collection UI
    if DATA_COLLECTION_PID_FILE.exists():
        pid_data, error = read_pid_file_safely(DATA_COLLECTION_PID_FILE)
        
        if error:
            result["stale_processes"].append({
                "type": "data_collection",
                "pid_file": str(DATA_COLLECTION_PID_FILE),
                "error": error
            })
        elif pid_data:
            try:
                pid = pid_data["pid"]
                port = pid_data.get("port", 7862)
                
                # Ensure port is an integer
                if isinstance(port, str) and port.isdigit():
                    port = int(port)
                elif not isinstance(port, int):
                    port = 7862  # Use default port if invalid
                
                # Check both process and port health
                is_alive = is_process_alive_and_healthy(pid)
                ui_health = check_gradio_health(port) if is_alive else {"status": "process_not_running"}
                
                # Combine information
                result["data_collection_ui"] = {
                    "pid": pid,
                    "port": port,
                    "process_alive": is_alive,
                    "ui_health": ui_health,
                    "metadata": pid_data
                }
                
                # Check for inconsistencies
                if is_alive and not ui_health["port_bound"]:
                    # Process is running but port is not bound
                    debug_print(f"Process {pid} is running but port {port} is not bound")
                    result["stale_processes"].append({
                        "type": "data_collection",
                        "pid": pid,
                        "port": port,
                        "reason": "Process running but port not bound",
                        "metadata": pid_data
                    })
                elif not is_alive and ui_health["port_bound"]:
                    # Port is bound but process is not running
                    debug_print(f"Port {port} is bound but process {pid} is not running")
                    result["stale_processes"].append({
                        "type": "data_collection",
                        "pid": pid,
                        "port": port,
                        "reason": "Port bound but process not running",
                        "metadata": pid_data
                    })
                elif not is_alive:
                    # Process is not running
                    debug_print(f"Data Collection UI process {pid} is not running")
                    result["stale_processes"].append({
                        "type": "data_collection",
                        "pid": pid,
                        "reason": "Process not running",
                        "metadata": pid_data
                    })
            except Exception as e:
                error_msg = f"Error checking Data Collection UI health: {e}"
                debug_print(f"{error_msg}\n{traceback.format_exc()}")
                result["stale_processes"].append({
                    "type": "data_collection",
                    "error": error_msg,
                    "metadata": pid_data
                })
    
    # Check Inference UI
    if INFERENCE_UI_PID_FILE.exists():
        pid_data, error = read_pid_file_safely(INFERENCE_UI_PID_FILE)
        
        if error:
            result["stale_processes"].append({
                "type": "inference_ui",
                "pid_file": str(INFERENCE_UI_PID_FILE),
                "error": error
            })
        elif pid_data:
            try:
                pid = pid_data["pid"]
                port = pid_data.get("port", 7861)
                
                # Ensure port is an integer
                if isinstance(port, str) and port.isdigit():
                    port = int(port)
                elif not isinstance(port, int):
                    port = 7861  # Use default port if invalid
                
                # Check both process and port health
                is_alive = is_process_alive_and_healthy(pid)
                ui_health = check_gradio_health(port) if is_alive else {"status": "process_not_running"}
                
                # Combine information
                result["inference_ui"] = {
                    "pid": pid,
                    "port": port,
                    "process_alive": is_alive,
                    "ui_health": ui_health,
                    "metadata": pid_data
                }
                
                # Check for inconsistencies
                if is_alive and not ui_health["port_bound"]:
                    # Process is running but port is not bound
                    debug_print(f"Process {pid} is running but port {port} is not bound")
                    result["stale_processes"].append({
                        "type": "inference_ui",
                        "pid": pid,
                        "port": port,
                        "reason": "Process running but port not bound",
                        "metadata": pid_data
                    })
                elif not is_alive and ui_health["port_bound"]:
                    # Port is bound but process is not running
                    debug_print(f"Port {port} is bound but process {pid} is not running")
                    result["stale_processes"].append({
                        "type": "inference_ui",
                        "pid": pid,
                        "port": port,
                        "reason": "Port bound but process not running",
                        "metadata": pid_data
                    })
                elif not is_alive:
                    # Process is not running
                    debug_print(f"Inference UI process {pid} is not running")
                    result["stale_processes"].append({
                        "type": "inference_ui",
                        "pid": pid,
                        "reason": "Process not running",
                        "metadata": pid_data
                    })
            except Exception as e:
                error_msg = f"Error checking Inference UI health: {e}"
                debug_print(f"{error_msg}\n{traceback.format_exc()}")
                result["stale_processes"].append({
                    "type": "inference_ui",
                    "error": error_msg,
                    "metadata": pid_data
                })
    
    # Log health check result
    log_health_event("PROCESS_HEALTH_CHECK", {
        "fastapi_status": result["fastapi_server"]["api_health"]["status"] if result["fastapi_server"] else "not_running",
        "data_collection_status": result["data_collection_ui"]["ui_health"]["status"] if result["data_collection_ui"] else "not_running",
        "inference_ui_status": result["inference_ui"]["ui_health"]["status"] if result["inference_ui"] else "not_running",
        "stale_process_count": len(result["stale_processes"]),
        "active_port_count": len(result["active_ports"]) if result["active_ports"] else 0
    })
    
    return result

def cleanup_process_conflict(process_type: str, port: int, reason: str = None) -> Dict[str, Any]:
    """
    Clean up a process conflict by safely terminating the conflicting process.
    
    Args:
        process_type: Type of process (fastapi, data_collection, inference_ui)
        port: Port number
        reason: Reason for cleanup (optional)
        
    Returns:
        dict: Results of the cleanup operation
    """
    result = {
        "process_type": process_type,
        "port": port,
        "reason": reason or "manual_cleanup",
        "success": False,
        "actions_taken": [],
        "errors": []
    }
    
    # Get the appropriate PID file
    if process_type == "fastapi":
        pid_file = INFERENCE_PROCESS_PID_FILE
    elif process_type == "data_collection":
        pid_file = DATA_COLLECTION_PID_FILE
    elif process_type == "inference_ui":
        pid_file = INFERENCE_UI_PID_FILE
    else:
        result["errors"].append(f"Unknown process type: {process_type}")
        return result
    
    debug_print(f"Cleaning up {process_type} process conflict on port {port}")
    
    # First, check if the PID file exists and contains a valid PID
    if pid_file.exists():
        pid_data, error = read_pid_file_safely(pid_file)
        
        if error:
            result["errors"].append(f"Error reading PID file: {error}")
        elif pid_data:
            try:
                pid = pid_data["pid"]
                
                # Verify this is the right process and it's running
                if is_process_alive_and_healthy(pid):
                    debug_print(f"Process {pid} is alive, attempting to terminate")
                    
                    # Get process name for verification before terminating
                    try:
                        process = psutil.Process(pid)
                        process_name = process.name()
                        
                        # Make sure it's the right type of process
                        if (process_type == "fastapi" and ("python" in process_name.lower() or "uvicorn" in process_name.lower())) or \
                           ((process_type == "data_collection" or process_type == "inference_ui") and "python" in process_name.lower()):
                            # Try to terminate gracefully
                            debug_print(f"Sending SIGTERM to process {pid}")
                            process.terminate()
                            
                            # Wait for termination
                            try:
                                process.wait(timeout=5)
                                debug_print(f"Process {pid} terminated gracefully")
                                result["actions_taken"].append(f"Terminated process {pid} gracefully")
                            except psutil.TimeoutExpired:
                                # Force kill if graceful termination fails
                                debug_print(f"Process {pid} did not terminate gracefully, sending SIGKILL")
                                process.kill()
                                result["actions_taken"].append(f"Force killed process {pid}")
                                
                            result["success"] = True
                        else:
                            # Process doesn't match expected type
                            error_msg = f"Process {pid} ({process_name}) doesn't match expected type {process_type}"
                            debug_print(error_msg)
                            result["errors"].append(error_msg)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        error_msg = f"Process {pid} not found or access denied during verification"
                        debug_print(error_msg)
                        result["errors"].append(error_msg)
                else:
                    debug_print(f"Process {pid} is not running")
                    result["actions_taken"].append(f"Process {pid} was not running")
            except Exception as e:
                error_msg = f"Error terminating process: {e}"
                debug_print(f"{error_msg}\n{traceback.format_exc()}")
                result["errors"].append(error_msg)
    
    # Now check if port is still bound
    if check_port_binding(port):
        debug_print(f"Port {port} is still bound, finding process")
        
        # Find process using the port
        try:
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                for conn in proc.info['connections']:
                    if conn.laddr.port == port:
                        port_pid = proc.info['pid']
                        port_name = proc.info['name']
                        
                        debug_print(f"Found process {port_pid} ({port_name}) using port {port}")
                        
                        # Try to terminate
                        try:
                            port_process = psutil.Process(port_pid)
                            
                            # Try to terminate gracefully
                            debug_print(f"Sending SIGTERM to process {port_pid}")
                            port_process.terminate()
                            
                            # Wait for termination
                            try:
                                port_process.wait(timeout=5)
                                debug_print(f"Process {port_pid} terminated gracefully")
                                result["actions_taken"].append(f"Terminated process {port_pid} bound to port {port}")
                            except psutil.TimeoutExpired:
                                # Force kill if graceful termination fails
                                debug_print(f"Process {port_pid} did not terminate gracefully, sending SIGKILL")
                                port_process.kill()
                                result["actions_taken"].append(f"Force killed process {port_pid} bound to port {port}")
                                
                            result["success"] = True
                            break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            error_msg = f"Process {port_pid} not found or access denied"
                            debug_print(error_msg)
                            result["errors"].append(error_msg)
                            
                if result["success"]:
                    break
        except Exception as e:
            error_msg = f"Error finding process using port {port}: {e}"
            debug_print(f"{error_msg}\n{traceback.format_exc()}")
            result["errors"].append(error_msg)
    
    # Finally, remove the PID file if it still exists
    if pid_file.exists():
        try:
            pid_file.unlink()
            debug_print(f"Removed PID file {pid_file}")
            result["actions_taken"].append(f"Removed PID file {pid_file}")
            result["success"] = True
        except Exception as e:
            error_msg = f"Error removing PID file {pid_file}: {e}"
            debug_print(error_msg)
            result["errors"].append(error_msg)
    
    # Log the cleanup operation
    log_health_event("PROCESS_CONFLICT_CLEANUP", {
        "process_type": process_type,
        "port": port,
        "reason": result["reason"],
        "success": result["success"],
        "actions_taken": result["actions_taken"],
        "errors": result["errors"]
    })
    
    return result

def cleanup_stale_gradio_processes(keep_newest: bool = False) -> Dict[str, Any]:
    """
    Safely cleanup stale Gradio processes with proper health checking.
    
    Args:
        keep_newest: If True, keep the newest Gradio process for each type
        
    Returns:
        dict: Results of the cleanup operation
    """
    result = {
        "data_collection": {
            "total": 0,
            "active": 0,
            "terminated": 0,
            "kept": []
        },
        "inference_ui": {
            "total": 0,
            "active": 0,
            "terminated": 0,
            "kept": []
        },
        "errors": []
    }
    
    # First, get all Gradio processes and verify their health
    try:
        data_collection_processes = []
        inference_ui_processes = []
        
        # Find all Python processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            # Filter for Python processes
            if "python" in proc.info['name'].lower():
                try:
                    cmdline = " ".join([str(c) for c in proc.info.get('cmdline', [])])
                    
                    # Check for Gradio data collection
                    if "data_collection" in cmdline and "gradio" in cmdline:
                        debug_print(f"Found Gradio data collection process: {proc.info['pid']}")
                        
                        # Check process health
                        health_info = {
                            "pid": proc.info['pid'],
                            "cmdline": cmdline,
                            "create_time": proc.info['create_time'],
                            "is_alive": is_process_alive_and_healthy(proc.info['pid']),
                            "health_checked": False,
                            "port": None
                        }
                        
                        # Try to extract port from command line
                        try:
                            if "GRADIO_PORT" in cmdline:
                                port_part = cmdline.split("GRADIO_PORT=")[1].split()[0]
                                health_info["port"] = int(port_part)
                            else:
                                # Default data collection port
                                health_info["port"] = 7862
                        except Exception:
                            health_info["port"] = 7862  # Use default port
                            
                        # Check UI responsiveness if port is known
                        if health_info["port"] and health_info["is_alive"]:
                            ui_health = check_gradio_health(health_info["port"])
                            health_info["ui_health"] = ui_health
                            health_info["health_checked"] = True
                            health_info["is_responsive"] = ui_health["ui_reachable"]
                        
                        data_collection_processes.append(health_info)
                        result["data_collection"]["total"] += 1
                        
                        if health_info["is_alive"]:
                            result["data_collection"]["active"] += 1
                    
                    # Check for Gradio inference UI
                    elif "inference" in cmdline and "gradio" in cmdline:
                        debug_print(f"Found Gradio inference UI process: {proc.info['pid']}")
                        
                        # Check process health
                        health_info = {
                            "pid": proc.info['pid'],
                            "cmdline": cmdline,
                            "create_time": proc.info['create_time'],
                            "is_alive": is_process_alive_and_healthy(proc.info['pid']),
                            "health_checked": False,
                            "port": None
                        }
                        
                        # Try to extract port from command line
                        try:
                            if "GRADIO_PORT" in cmdline:
                                port_part = cmdline.split("GRADIO_PORT=")[1].split()[0]
                                health_info["port"] = int(port_part)
                            else:
                                # Default inference UI port
                                health_info["port"] = 7861
                        except Exception:
                            health_info["port"] = 7861  # Use default port
                            
                        # Check UI responsiveness if port is known
                        if health_info["port"] and health_info["is_alive"]:
                            ui_health = check_gradio_health(health_info["port"])
                            health_info["ui_health"] = ui_health
                            health_info["health_checked"] = True
                            health_info["is_responsive"] = ui_health["ui_reachable"]
                        
                        inference_ui_processes.append(health_info)
                        result["inference_ui"]["total"] += 1
                        
                        if health_info["is_alive"]:
                            result["inference_ui"]["active"] += 1
                except Exception as e:
                    error_msg = f"Error checking process {proc.info['pid']}: {e}"
                    debug_print(error_msg)
                    result["errors"].append(error_msg)
        
        # Sort processes by create time (newest first)
        data_collection_processes.sort(key=lambda p: p["create_time"], reverse=True)
        inference_ui_processes.sort(key=lambda p: p["create_time"], reverse=True)
        
        # Determine which processes to keep
        if keep_newest and data_collection_processes:
            # Find the newest responsive process
            kept_data_collection = None
            for proc in data_collection_processes:
                if proc["is_alive"] and proc.get("is_responsive", False):
                    kept_data_collection = proc
                    break
                    
            # If no responsive process found, keep the newest alive one
            if not kept_data_collection and any(p["is_alive"] for p in data_collection_processes):
                kept_data_collection = next(p for p in data_collection_processes if p["is_alive"])
                
            # Mark the process to keep
            if kept_data_collection:
                kept_data_collection["keep"] = True
                result["data_collection"]["kept"].append(kept_data_collection["pid"])
        
        if keep_newest and inference_ui_processes:
            # Find the newest responsive process
            kept_inference_ui = None
            for proc in inference_ui_processes:
                if proc["is_alive"] and proc.get("is_responsive", False):
                    kept_inference_ui = proc
                    break
                    
            # If no responsive process found, keep the newest alive one
            if not kept_inference_ui and any(p["is_alive"] for p in inference_ui_processes):
                kept_inference_ui = next(p for p in inference_ui_processes if p["is_alive"])
                
            # Mark the process to keep
            if kept_inference_ui:
                kept_inference_ui["keep"] = True
                result["inference_ui"]["kept"].append(kept_inference_ui["pid"])
        
        # Terminate processes that should be terminated
        for process_list, process_type in [(data_collection_processes, "data_collection"),
                                           (inference_ui_processes, "inference_ui")]:
            for proc in process_list:
                if proc.get("keep", False):
                    debug_print(f"Keeping {process_type} process {proc['pid']}")
                    continue
                    
                if proc["is_alive"]:
                    debug_print(f"Terminating {process_type} process {proc['pid']}")
                    
                    try:
                        # Try to terminate gracefully
                        process = psutil.Process(proc["pid"])
                        process.terminate()
                        
                        # Wait for termination
                        try:
                            process.wait(timeout=5)
                            debug_print(f"Process {proc['pid']} terminated gracefully")
                        except psutil.TimeoutExpired:
                            # Force kill if graceful termination fails
                            debug_print(f"Process {proc['pid']} did not terminate gracefully, sending SIGKILL")
                            process.kill()
                            
                        result[process_type]["terminated"] += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        error_msg = f"Error terminating process {proc['pid']}: {e}"
                        debug_print(error_msg)
                        result["errors"].append(error_msg)
        
        # Update PID files to match the kept processes
        if keep_newest:
            # Update data collection PID file
            if result["data_collection"]["kept"]:
                kept_pid = result["data_collection"]["kept"][0]
                kept_proc = next(p for p in data_collection_processes if p["pid"] == kept_pid)
                
                # Only update PID file if process is still alive
                if is_process_alive_and_healthy(kept_pid):
                    try:
                        # Create proper PID file data
                        pid_data = {
                            "pid": kept_pid,
                            "port": kept_proc.get("port", 7862),
                            "timestamp": datetime.now().isoformat(),
                            "process_type": "GradioDataCollection",
                            "updated_by": "process_health_checker"
                        }
                        
                        # Write to PID file with file locking
                        with open(DATA_COLLECTION_PID_FILE, 'w') as f:
                            fcntl.flock(f, fcntl.LOCK_EX)
                            try:
                                json.dump(pid_data, f, indent=2)
                            finally:
                                fcntl.flock(f, fcntl.LOCK_UN)
                                
                        debug_print(f"Updated data collection PID file with PID {kept_pid}")
                    except Exception as e:
                        error_msg = f"Error updating data collection PID file: {e}"
                        debug_print(error_msg)
                        result["errors"].append(error_msg)
            
            # Update inference UI PID file
            if result["inference_ui"]["kept"]:
                kept_pid = result["inference_ui"]["kept"][0]
                kept_proc = next(p for p in inference_ui_processes if p["pid"] == kept_pid)
                
                # Only update PID file if process is still alive
                if is_process_alive_and_healthy(kept_pid):
                    try:
                        # Create proper PID file data
                        pid_data = {
                            "pid": kept_pid,
                            "port": kept_proc.get("port", 7861),
                            "timestamp": datetime.now().isoformat(),
                            "process_type": "GradioInferenceUI",
                            "updated_by": "process_health_checker"
                        }
                        
                        # Write to PID file with file locking
                        with open(INFERENCE_UI_PID_FILE, 'w') as f:
                            fcntl.flock(f, fcntl.LOCK_EX)
                            try:
                                json.dump(pid_data, f, indent=2)
                            finally:
                                fcntl.flock(f, fcntl.LOCK_UN)
                                
                        debug_print(f"Updated inference UI PID file with PID {kept_pid}")
                    except Exception as e:
                        error_msg = f"Error updating inference UI PID file: {e}"
                        debug_print(error_msg)
                        result["errors"].append(error_msg)
                        
    except Exception as e:
        error_msg = f"Error cleaning up Gradio processes: {e}"
        debug_print(f"{error_msg}\n{traceback.format_exc()}")
        result["errors"].append(error_msg)
    
    # Log the cleanup operation
    log_health_event("GRADIO_PROCESS_CLEANUP", {
        "data_collection_total": result["data_collection"]["total"],
        "data_collection_active": result["data_collection"]["active"],
        "data_collection_terminated": result["data_collection"]["terminated"],
        "data_collection_kept": result["data_collection"]["kept"],
        "inference_ui_total": result["inference_ui"]["total"],
        "inference_ui_active": result["inference_ui"]["active"],
        "inference_ui_terminated": result["inference_ui"]["terminated"],
        "inference_ui_kept": result["inference_ui"]["kept"],
        "error_count": len(result["errors"])
    })
    
    return result

if __name__ == "__main__":
    # Run a health check if executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Process health checker and conflict resolver")
    parser.add_argument("--check", action="store_true", help="Run a health check")
    parser.add_argument("--cleanup", action="store_true", help="Clean up stale processes")
    parser.add_argument("--cleanup-gradio", action="store_true", help="Clean up duplicate Gradio processes")
    parser.add_argument("--resolve", choices=["fastapi", "data_collection", "inference_ui"], 
                        help="Resolve a specific process conflict")
    parser.add_argument("--port", type=int, help="Port for conflict resolution")
    parser.add_argument("--keep-newest", action="store_true", help="Keep newest Gradio process when cleaning up")
    
    args = parser.parse_args()
    
    if args.check:
        print("Running process health check...")
        health_result = check_all_process_health()
        print(f"Health check completed. Found {len(health_result['stale_processes'])} stale processes.")
        
        # Print summary
        print("\nProcess Status Summary:")
        print("=======================")
        
        if health_result["fastapi_server"]:
            status = health_result["fastapi_server"]["api_health"]["status"]
            print(f"FastAPI Server (PID {health_result['fastapi_server']['pid']}): {status}")
        else:
            print("FastAPI Server: Not running")
            
        if health_result["data_collection_ui"]:
            status = health_result["data_collection_ui"]["ui_health"]["status"]
            print(f"Data Collection UI (PID {health_result['data_collection_ui']['pid']}): {status}")
        else:
            print("Data Collection UI: Not running")
            
        if health_result["inference_ui"]:
            status = health_result["inference_ui"]["ui_health"]["status"]
            print(f"Inference UI (PID {health_result['inference_ui']['pid']}): {status}")
        else:
            print("Inference UI: Not running")
        
        # Print stale processes
        if health_result["stale_processes"]:
            print("\nStale Processes:")
            print("===============")
            
            for proc in health_result["stale_processes"]:
                proc_type = proc.get("type", "Unknown")
                pid = proc.get("pid", "N/A")
                reason = proc.get("reason", "Unknown reason")
                print(f"- {proc_type} (PID {pid}): {reason}")
        
        # Print active ports
        if health_result["active_ports"]:
            print("\nActive Ports:")
            print("============")
            
            for port_info in health_result["active_ports"]:
                port = port_info["port"]
                service = port_info.get("detected_service", "Unknown")
                pid = port_info.get("pid", "N/A")
                proc_name = port_info.get("process_name", "Unknown")
                print(f"- Port {port}: {service} (PID {pid}, {proc_name})")
    
    if args.cleanup:
        print("Cleaning up stale processes...")
        result = check_all_process_health()
        stale_count = len(result["stale_processes"])
        
        if stale_count > 0:
            print(f"Found {stale_count} stale processes to clean up")
            
            for proc in result["stale_processes"]:
                proc_type = proc.get("type", "Unknown")
                
                if proc_type == "fastapi":
                    port = proc.get("port", 8001)
                    print(f"Cleaning up FastAPI process on port {port}")
                    cleanup_result = cleanup_process_conflict("fastapi", port, proc.get("reason"))
                    
                    if cleanup_result["success"]:
                        print("Cleanup successful")
                    else:
                        print(f"Cleanup failed: {', '.join(cleanup_result['errors'])}")
                        
                elif proc_type == "data_collection":
                    port = proc.get("port", 7862)
                    print(f"Cleaning up Data Collection UI process on port {port}")
                    cleanup_result = cleanup_process_conflict("data_collection", port, proc.get("reason"))
                    
                    if cleanup_result["success"]:
                        print("Cleanup successful")
                    else:
                        print(f"Cleanup failed: {', '.join(cleanup_result['errors'])}")
                        
                elif proc_type == "inference_ui":
                    port = proc.get("port", 7861)
                    print(f"Cleaning up Inference UI process on port {port}")
                    cleanup_result = cleanup_process_conflict("inference_ui", port, proc.get("reason"))
                    
                    if cleanup_result["success"]:
                        print("Cleanup successful")
                    else:
                        print(f"Cleanup failed: {', '.join(cleanup_result['errors'])}")
        else:
            print("No stale processes found to clean up")
    
    if args.cleanup_gradio:
        print("Cleaning up Gradio processes...")
        cleanup_result = cleanup_stale_gradio_processes(args.keep_newest)
        
        print(f"Data Collection UI: {cleanup_result['data_collection']['terminated']} terminated, "
              f"{len(cleanup_result['data_collection']['kept'])} kept")
        print(f"Inference UI: {cleanup_result['inference_ui']['terminated']} terminated, "
              f"{len(cleanup_result['inference_ui']['kept'])} kept")
        
        if cleanup_result["errors"]:
            print(f"Errors: {len(cleanup_result['errors'])}")
            for error in cleanup_result["errors"][:5]:  # Show first 5 errors
                print(f"- {error}")
                
            if len(cleanup_result["errors"]) > 5:
                print(f"... and {len(cleanup_result['errors']) - 5} more errors")
    
    if args.resolve:
        if not args.port:
            print("Error: --port is required when using --resolve")
            sys.exit(1)
            
        print(f"Resolving {args.resolve} process conflict on port {args.port}...")
        cleanup_result = cleanup_process_conflict(args.resolve, args.port, "manual_resolution")
        
        if cleanup_result["success"]:
            print("Conflict resolution successful")
            print("Actions taken:")
            for action in cleanup_result["actions_taken"]:
                print(f"- {action}")
        else:
            print("Conflict resolution failed")
            print("Errors:")
            for error in cleanup_result["errors"]:
                print(f"- {error}")