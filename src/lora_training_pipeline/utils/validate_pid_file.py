#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/validate_pid_file.py

"""
PID File Validation and Recovery Tool

This utility validates, fixes, and normalizes PID files to ensure they follow 
the proper JSON schema and prevent TypeErrors when working with them.

The tool can:
1. Detect legacy plain-integer PID files and convert them to proper JSON
2. Validate PID files against a schema
3. Fix common problems with PID files
4. Detect stale PID files and clean them up
"""

import os
import sys
import json
import fcntl
import psutil
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

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

# Enable debug mode by default
DEBUG_MODE = True

def debug_print(*args, **kwargs):
    """Print debug information when DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print("[PID-VALIDATOR]", *args, **kwargs)

def is_process_running(pid: int) -> bool:
    """
    Check if a process with the given PID is running.
    
    Args:
        pid: Process ID to check
        
    Returns:
        bool: True if process is running, False otherwise
    """
    try:
        # Access the process - this will raise an exception if it doesn't exist
        process = psutil.Process(pid)
        
        # Check if zombie - zombies are not considered "running"
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
            
        # Process exists and is not a zombie
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False

def read_pid_file_for_validation(pid_file: Path) -> Tuple[Any, Optional[str]]:
    """
    Read a PID file for validation, returning the raw content.
    
    Args:
        pid_file: Path to the PID file
        
    Returns:
        Tuple[Any, Optional[str]]: Content and error message, if any
    """
    if not pid_file.exists():
        return None, f"PID file {pid_file} does not exist"
        
    try:
        with open(pid_file, 'r') as f:
            # Use file locking to ensure consistent read
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                content = f.read().strip()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                
        if not content:
            return None, f"PID file {pid_file} is empty"
            
        # Try to parse as JSON
        try:
            data = json.loads(content)
            return data, None
        except json.JSONDecodeError:
            # Not JSON, try to parse as integer
            try:
                pid = int(content)
                # Just return the raw integer for validation purposes
                return pid, None
            except ValueError:
                # Return the raw content for validation
                return content, f"PID file contains invalid data (not JSON or integer)"
                
    except Exception as e:
        error_msg = f"Error reading PID file {pid_file}: {e}"
        debug_print(error_msg)
        return None, error_msg

def validate_pid_file(pid_file: Path) -> Dict[str, Any]:
    """
    Validate a PID file against the schema, with detailed reporting.
    
    Args:
        pid_file: Path to the PID file
        
    Returns:
        dict: Validation results and errors
    """
    result = {
        "pid_file": str(pid_file),
        "exists": pid_file.exists(),
        "valid": False,
        "errors": [],
        "warnings": [],
        "data": None,
        "normalized_data": None,
        "needs_fixing": False,
        "pid_running": False
    }
    
    if not result["exists"]:
        result["errors"].append(f"PID file {pid_file} does not exist")
        return result
        
    # Read raw content
    raw_data, error = read_pid_file_for_validation(pid_file)
    
    if error:
        result["errors"].append(error)
        result["needs_fixing"] = True
        return result
        
    result["data"] = raw_data
    
    # Check data type and validate
    if isinstance(raw_data, int):
        # Legacy format - single integer PID
        result["warnings"].append(f"PID file contains legacy format (raw integer: {raw_data})")
        result["normalized_data"] = {
            "pid": raw_data,
            "timestamp": datetime.now().isoformat(),
            "legacy_format": True,
            "converted_at": datetime.now().isoformat()
        }
        result["needs_fixing"] = True
        
        # Check if PID is actually running
        if is_process_running(raw_data):
            result["pid_running"] = True
        else:
            result["warnings"].append(f"PID {raw_data} is not running")
            
    elif isinstance(raw_data, dict):
        # Check for required fields
        if "pid" not in raw_data:
            result["errors"].append("PID file missing required 'pid' field")
            result["needs_fixing"] = True
        else:
            # Check PID value
            try:
                pid = int(raw_data["pid"])
                
                # Normalize data
                result["normalized_data"] = raw_data.copy()
                result["normalized_data"]["pid"] = pid
                
                # Add missing fields
                if "timestamp" not in raw_data:
                    result["normalized_data"]["timestamp"] = datetime.now().isoformat()
                    result["warnings"].append("Added missing 'timestamp' field")
                    result["needs_fixing"] = True
                
                # Check if PID is running
                if is_process_running(pid):
                    result["pid_running"] = True
                else:
                    result["warnings"].append(f"PID {pid} is not running")
                    
                # Check port field if it exists
                if "port" in raw_data:
                    try:
                        # Convert port to integer if it's a string
                        if isinstance(raw_data["port"], str) and raw_data["port"].isdigit():
                            result["normalized_data"]["port"] = int(raw_data["port"])
                            result["warnings"].append(f"Converted port from string to integer")
                            result["needs_fixing"] = True
                    except (ValueError, TypeError):
                        result["warnings"].append(f"Invalid port value: {raw_data['port']}")
                        
                # Valid dictionary with PID
                result["valid"] = True
                
            except (ValueError, TypeError):
                result["errors"].append(f"Invalid pid value: {raw_data.get('pid')}")
                result["needs_fixing"] = True
    else:
        # Unknown format
        result["errors"].append(f"Unknown PID file format: {type(raw_data)}")
        result["needs_fixing"] = True
    
    return result

def fix_pid_file(pid_file: Path, validation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix a PID file based on validation results.
    
    Args:
        pid_file: Path to the PID file
        validation_result: Results from validate_pid_file
        
    Returns:
        dict: Results of the fix operation
    """
    fix_result = {
        "pid_file": str(pid_file),
        "success": False,
        "actions": [],
        "errors": []
    }
    
    # Check if file should be fixed
    if not validation_result["needs_fixing"]:
        fix_result["actions"].append("No fixes needed")
        fix_result["success"] = True
        return fix_result
        
    # If the process is not running and file exists, remove it
    if pid_file.exists() and not validation_result["pid_running"]:
        try:
            pid_file.unlink()
            fix_result["actions"].append(f"Removed stale PID file (process not running)")
            fix_result["success"] = True
            return fix_result
        except Exception as e:
            fix_result["errors"].append(f"Error removing stale PID file: {e}")
            return fix_result
    
    # If we have normalized data, write it to the file
    if validation_result["normalized_data"]:
        try:
            # Make sure the directory exists
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write normalized data with file locking
            with open(pid_file, 'w') as f:
                # Exclusive lock while writing
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    json.dump(validation_result["normalized_data"], f, indent=2)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
                    
            fix_result["actions"].append(f"Wrote normalized data to PID file")
            fix_result["success"] = True
        except Exception as e:
            fix_result["errors"].append(f"Error writing normalized data: {e}")
    
    return fix_result

def scan_and_validate_all_pid_files() -> Dict[str, Any]:
    """
    Scan for and validate all PID files in the current directory.
    
    Returns:
        dict: Results of the scan and validation
    """
    result = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "fixed_files": 0,
        "files": {},
        "errors": []
    }
    
    # Define common PID file paths to check
    pid_file_paths = [
        Path("./inference_process.pid"),
        Path("./data_collection_ui.pid"),
        Path("./inference_ui.pid")
    ]
    
    # Add any additional .pid files in the current directory
    try:
        for pid_file in Path(".").glob("*.pid"):
            if pid_file not in pid_file_paths:
                pid_file_paths.append(pid_file)
    except Exception as e:
        result["errors"].append(f"Error scanning for additional PID files: {e}")
    
    # Validate each PID file
    for pid_file in pid_file_paths:
        try:
            # Skip if file doesn't exist and it's not one of the standard files
            if not pid_file.exists() and pid_file.name not in [
                "inference_process.pid", "data_collection_ui.pid", "inference_ui.pid"
            ]:
                continue
                
            result["total_files"] += 1
            
            # Validate the PID file
            validation_result = validate_pid_file(pid_file)
            result["files"][str(pid_file)] = validation_result
            
            if validation_result["valid"]:
                result["valid_files"] += 1
            else:
                result["invalid_files"] += 1
                
            # Fix the PID file if needed
            if validation_result["needs_fixing"]:
                fix_result = fix_pid_file(pid_file, validation_result)
                result["files"][str(pid_file)]["fix_result"] = fix_result
                
                if fix_result["success"]:
                    result["fixed_files"] += 1
                    
        except Exception as e:
            error_msg = f"Error processing {pid_file}: {e}"
            debug_print(error_msg)
            debug_print(traceback.format_exc())
            result["errors"].append(error_msg)
    
    return result

if __name__ == "__main__":
    # When run directly, validate all PID files
    print("PID File Validation Tool")
    print("=====================\n")
    
    print("Scanning for PID files...")
    results = scan_and_validate_all_pid_files()
    
    print(f"Scanned {results['total_files']} PID files:")
    print(f"  - Valid: {results['valid_files']}")
    print(f"  - Invalid: {results['invalid_files']}")
    print(f"  - Fixed: {results['fixed_files']}")
    
    if results["errors"]:
        print("\nErrors:")
        for error in results["errors"]:
            print(f"  - {error}")
    
    print("\nFile Details:")
    for file_path, file_result in results["files"].items():
        print(f"\n{file_path}:")
        print(f"  - Valid: {file_result['valid']}")
        print(f"  - PID Running: {file_result['pid_running']}")
        
        if file_result.get("data"):
            if isinstance(file_result["data"], dict):
                print(f"  - PID: {file_result['data'].get('pid', 'Unknown')}")
                print(f"  - Port: {file_result['data'].get('port', 'Unknown')}")
            elif isinstance(file_result["data"], int):
                print(f"  - PID: {file_result['data']} (legacy format)")
        
        if file_result.get("errors"):
            print("  - Errors:")
            for error in file_result["errors"]:
                print(f"    - {error}")
                
        if file_result.get("warnings"):
            print("  - Warnings:")
            for warning in file_result["warnings"]:
                print(f"    - {warning}")
                
        if file_result.get("fix_result"):
            print("  - Fix Results:")
            print(f"    - Success: {file_result['fix_result']['success']}")
            for action in file_result['fix_result'].get("actions", []):
                print(f"    - Action: {action}")