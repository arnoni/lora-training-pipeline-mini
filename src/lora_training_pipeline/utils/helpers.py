# LoRA_Training_Pipeline/src/utils/helpers.py
import logging
import sys
import os
import subprocess
import time
import datetime
import threading
from functools import wraps
from typing import Optional, Callable, Any, Union
from tenacity import retry, wait_exponential, stop_after_attempt
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

def setup_logger(name, log_file, level=logging.INFO):
    """Sets up a logger with file and stream handlers."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(stream_handler)

    return logger

def log_exceptions(logger):
    """Decorator to log exceptions in a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {e}")
                raise  # Re-raise the exception after logging
        return wrapper
    return decorator

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def network_request_with_retry(url, params=None):
    """Example function with retry logic."""
    # Replace this with your actual network request logic (e.g., using httpx)
    #  This is a placeholder for demonstration
    print(f"Attempting to access {url}...")
    # Simulate a network request that might fail
    if params and params.get("fail"):
        raise Exception("Simulated network failure")
    return {"status": "success", "url": url}


#Example Usage
# logger = setup_logger('my_module', 'my_module.log')

# @log_exceptions(logger)
# def my_function():
#     # ... function logic ...
#     pass

def validate_process_log_file(log_file="Windows_processes_ML_pipeline.txt"):
    """
    Validates the Windows process log file by checking if processes mentioned in it
    are still running. If processes are found to be defunct, the file is cleaned up.
    
    Args:
        log_file (str): Path to the process log file
        
    Returns:
        bool: True if validation was performed, False otherwise
    """
    if os.name != 'nt':  # Only perform this on Windows
        return False
        
    import re
    import time
    
    print(f"Validating Windows process log file: {log_file}")
    
    # Check if the file exists
    if not os.path.exists(log_file):
        print("Log file does not exist, no validation needed.")
        return True
        
    # Try to read the file with UTF-8 encoding
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try with a fallback encoding
        try:
            with open(log_file, 'r', encoding='latin-1') as f:
                content = f.read()
            print("Note: Used fallback encoding (latin-1) to read log file")
        except Exception as e:
            print(f"[DEBUG] File encoding error type: {type(e).__name__}")
            print(f"[DEBUG] File encoding error details: {e}")
            print(f"Error reading log file: {e}")
            return False
    except Exception as e:
        print(f"[DEBUG] Log file read error type: {type(e).__name__}")
        print(f"[DEBUG] Log file read error details: {e}")
        import traceback
        print(f"[DEBUG] Log file read traceback: {traceback.format_exc()}")
        print(f"Error reading log file: {e}")
        return False
        
    # Extract PIDs from the log file
    pid_pattern = re.compile(r'pid=(\d+)')
    pids = pid_pattern.findall(content)
    
    # Convert to integers and remove duplicates
    unique_pids = set([int(pid) for pid in pids if pid.isdigit()])
    
    if not unique_pids:
        print("No valid PIDs found in log file.")
        return True
        
    print(f"Found {len(unique_pids)} unique PIDs in log file.")
    
    # Check which PIDs are still running
    running_pids = []
    defunct_pids = []
    
    for pid in unique_pids:
        try:
            # Check if the process exists
            # On Windows, using tasklist to check if process exists
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}', '/NH'], 
                capture_output=True, 
                text=True,
                check=False
            )
            
            # If the output contains the PID, the process is running
            if str(pid) in result.stdout:
                running_pids.append(pid)
            else:
                defunct_pids.append(pid)
                
        except Exception as e:
            print(f"[DEBUG] PID check error type: {type(e).__name__}")
            print(f"[DEBUG] PID check error details: {e}")
            print(f"Error checking PID {pid}: {e}")
            defunct_pids.append(pid)
    
    # Report findings
    print(f"Found {len(running_pids)} running processes and {len(defunct_pids)} defunct processes.")
    
    # If all processes are defunct, clear the file
    if defunct_pids and not running_pids:
        print("All processes in log file are defunct. Clearing log file.")
        try:
            # Create a backup of the old file
            backup_file = f"{log_file}.bak"
            with open(log_file, 'r', encoding='utf-8') as src, open(backup_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
                
            # Clear the file but add a header
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"# Windows processes log file - Reset on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Previous log contained {len(defunct_pids)} defunct processes\n")
                f.write(f"# PIDs: {', '.join(map(str, defunct_pids))}\n\n")
                
            print(f"Log file cleared. Backup saved to {backup_file}.")
            
        except Exception as e:
            print(f"[DEBUG] Log file clear error type: {type(e).__name__}")
            print(f"[DEBUG] Log file clear error details: {e}")
            import traceback
            print(f"[DEBUG] Log file clear traceback: {traceback.format_exc()}")
            print(f"Error clearing log file: {e}")
            return False
    
    # If some processes are running and some are defunct, mark the defunct ones
    elif defunct_pids:
        print("Some processes are defunct. Marking them in the log file.")
        try:
            # Create a new file with defunct processes marked
            new_content = []
            for line in content.splitlines():
                marked = False
                for pid in defunct_pids:
                    if f"pid={pid}" in line and "DEFUNCT" not in line:
                        new_content.append(f"{line} [DEFUNCT]")
                        marked = True
                        break
                if not marked:
                    new_content.append(line)
            
            # Write the updated content
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(new_content))
                
            print("Defunct processes marked in log file.")
            
        except Exception as e:
            print(f"[DEBUG] Log file update error type: {type(e).__name__}")
            print(f"[DEBUG] Log file update error details: {e}")
            import traceback
            print(f"[DEBUG] Log file update traceback: {traceback.format_exc()}")
            print(f"Error updating log file: {e}")
            return False
    
    return True

def get_process_name(process):
    """
    Gets a friendly name for a process, trying different methods:
    1. Check for PROCESS_NAME environment variable
    2. Check process.args and look for a friendly name
    3. Fall back to PID if nothing else works
    
    Args:
        process: A subprocess.Popen object
        
    Returns:
        str: A friendly name for the process
    """
    # If process is None, return a default value
    if not process:
        return "UnknownProcess"
        
    # Try to get the process name from environment variables
    if hasattr(process, 'env') and process.env and 'PROCESS_NAME' in process.env:
        return process.env['PROCESS_NAME']
    
    # Try to extract a meaningful name from the command
    if hasattr(process, 'args'):
        args = process.args
        if isinstance(args, list) and len(args) > 0:
            # Check for the special case where we set sys.argv[0]
            if '-c' in args and 'sys.argv[0]' in str(args):
                # Extract the name from the command string
                cmd_str = str(args)
                import re
                match = re.search(r"sys\.argv\[0\] = '([^']+)'", cmd_str)
                if match:
                    return match.group(1)
            
            # Otherwise use the script name if it's a .py file
            for arg in args:
                if isinstance(arg, str) and arg.endswith('.py'):
                    return os.path.basename(arg).replace('.py', '')
    
    # Fall back to PID if we couldn't determine a better name
    return f"Process-{process.pid if hasattr(process, 'pid') else 'unknown'}"

def log_process_activity(action, process_info, log_file="Windows_processes_ML_pipeline.txt"):
    """
    Logs information about Windows process activity to a text file.
    
    Args:
        action (str): The action being performed (e.g., "START", "TERMINATE", "PAUSE")
        process_info (dict): Information about the process
        log_file (str): Path to the log file
    """
    if os.name != 'nt':  # Only track for Windows
        return
        
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Add process name if it's not already included
    if 'process_name' not in process_info and 'command' in process_info:
        # Try to extract a meaningful name from the command
        cmd = process_info['command']
        if 'GradioDataCollection' in str(cmd):
            process_info['process_name'] = 'GradioDataCollection'
        elif 'ZenMLTrainingPipeline' in str(cmd):
            process_info['process_name'] = 'ZenMLTrainingPipeline'
        elif 'GradioInferenceUI' in str(cmd):
            process_info['process_name'] = 'GradioInferenceUI'
        elif 'gradio_app.py' in str(cmd):
            process_info['process_name'] = 'GradioDataCollection'
        elif 'zenml_pipeline.py' in str(cmd):
            process_info['process_name'] = 'ZenMLTrainingPipeline'
        elif 'gradio_inference.py' in str(cmd):
            process_info['process_name'] = 'GradioInferenceUI'
        elif 'fastapi_inference' in str(cmd):
            process_info['process_name'] = 'FastAPIInferenceServer'
    
    # Create log entry
    log_entry = f"[{timestamp}] {action}: "
    for key, value in process_info.items():
        log_entry += f"{key}={value} "
    log_entry += "\n"
    
    # Write to log file with UTF-8 encoding
    try:
        # Create file if it doesn't exist with a header
        if not os.path.exists(log_file):
            with open(log_file, "w", encoding='utf-8') as f:
                f.write(f"# Windows processes log file - Created on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# This file tracks Windows processes created by the ML pipeline\n")
                f.write(f"# Format: [timestamp] ACTION: key1=value1 key2=value2 ...\n\n")
        
        # Append the new log entry
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"[DEBUG] Process log write error type: {type(e).__name__}")
        print(f"[DEBUG] Process log write error details: {e}")
        print(f"⚠️ Warning: Could not write to process log file: {e}")

def check_dependencies(required_modules, processes=None):
    """
    Checks if all required modules are installed. If not, terminates any existing processes
    before exiting.
    
    Args:
        required_modules (list): List of module names to check
        processes (list, optional): List of subprocess.Popen objects to terminate if check fails
        
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("\n" + "="*80)
        print(f"❌ ERROR: Missing required modules: {', '.join(missing_modules)}")
        print("="*80)
        
        # Terminate any running processes if provided
        if processes:
            print("\n" + "="*80)
            print("⚠️ TERMINATING RUNNING PROCESSES")
            print("="*80)
            
            for i, process in enumerate(processes):
                try:
                    if process and process.poll() is None:  # Check if process exists and is still running
                        print(f"ℹ️ Terminating process {i+1}: PID {process.pid}")
                        
                        if os.name == 'nt':  # Windows
                            # On Windows, use taskkill for more reliable process termination
                            try:
                                # Log the termination attempt
                                log_process_activity("TERMINATE", {
                                    "pid": process.pid, 
                                    "command": str(process.args) if hasattr(process, 'args') else "unknown",
                                    "reason": "dependency_check_failed"
                                })
                                
                                # Execute taskkill
                                result = subprocess.run(['taskkill', '/F', '/PID', str(process.pid)], 
                                                     check=False, capture_output=True)
                                
                                # Log the result
                                log_process_activity("TERMINATE_RESULT", {
                                    "pid": process.pid,
                                    "success": result.returncode == 0,
                                    "return_code": result.returncode,
                                    "output": result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
                                })
                                
                                print(f"✅ Successfully terminated Windows process: PID {process.pid}")
                            except Exception as e:
                                print(f"[DEBUG] Windows process termination error type: {type(e).__name__}")
                                print(f"[DEBUG] Windows process termination error details: {e}")
                                print(f"⚠️ Error terminating Windows process {process.pid}: {e}")
                                log_process_activity("TERMINATE_ERROR", {
                                    "pid": process.pid,
                                    "error": str(e)
                                })
                        else:
                            # On Unix systems, we can just use terminate()
                            process.terminate()
                            try:
                                process.wait(timeout=3)
                                print(f"✅ Process {process.pid} terminated successfully")
                            except subprocess.TimeoutExpired:
                                process.kill()
                                print(f"⚠️ Process {process.pid} forcefully killed after timeout")
                except Exception as e:
                    print(f"[DEBUG] Process termination error type: {type(e).__name__}")
                    print(f"[DEBUG] Process termination error details: {e}")
                    import traceback
                    print(f"[DEBUG] Process termination traceback: {traceback.format_exc()}")
                    print(f"⚠️ Error during process termination: {e}")
        
        print("\n" + "="*80)
        print("ℹ️ RECOMMENDED ACTION:")
        print("1. Make sure all dependencies are installed with: uv pip install -e .")
        print("2. If using a virtual environment, ensure it is activated.")
        print("3. Check that your Python path is correct.")
        print("="*80 + "\n")
        
        sys.exit(1)
        
    return True

class ProgressTracker:
    """A utility class to track and display progress for long-running operations."""
    
    def __init__(self, 
                 task_name: str, 
                 total: Optional[int] = None, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize a progress tracker.
        
        Args:
            task_name: Name of the task being tracked.
            total: Total number of items/steps (if known).
            logger: Optional logger to use instead of print statements.
        """
        self.task_name = task_name
        self.total = total
        self.current = 0
        self.start_time = None
        self.logger = logger
        self.pbar = None
        self.timer_thread = None
        self.timer_running = False
    
    def start(self, message: Optional[str] = None) -> None:
        """Start tracking progress."""
        self.start_time = time.time()
        msg = f"Starting {self.task_name}..."
        if message:
            msg = f"{msg} {message}"
            
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"\n>>> {msg}")
            
        if TQDM_AVAILABLE and self.total:
            self.pbar = tqdm(total=self.total, desc=self.task_name, unit="steps")
    
    def update(self, step: int = 1, message: Optional[str] = None) -> None:
        """Update progress by the given number of steps."""
        self.current += step
        
        if self.pbar:
            self.pbar.update(step)
            if message:
                self.pbar.set_description(f"{self.task_name}: {message}")
        else:
            if self.total:
                percentage = min(100, round(100 * self.current / self.total, 2))
                elapsed = time.time() - self.start_time
                
                if elapsed > 0 and self.current > 0:
                    estimated_total = elapsed * self.total / self.current
                    remaining = max(0, estimated_total - elapsed)
                    time_str = f" | ~{format_time(remaining)} remaining"
                else:
                    time_str = ""
                    
                status_msg = f"Progress: {self.current}/{self.total} ({percentage}%){time_str}"
            else:
                status_msg = f"Progress: {self.current} steps completed"
                
            if message:
                status_msg = f"{status_msg} - {message}"
                
            if self.logger:
                self.logger.info(status_msg)
            else:
                print(f"  {status_msg}")
    
    def complete(self, message: Optional[str] = None) -> None:
        """Mark the task as complete."""
        self.stop_timer()
        
        if self.pbar:
            self.pbar.close()
            self.pbar = None
            
        elapsed = time.time() - self.start_time if self.start_time else 0
        msg = f"Completed {self.task_name} in {format_time(elapsed)}"
        if message:
            msg = f"{msg} - {message}"
            
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"\n✓ {msg}")
    
    def start_timer(self, message: str = "Operation in progress", interval: int = 10) -> None:
        """
        Start a timer that prints periodic updates for long-running operations.
        
        Args:
            message: Message to display with the timer.
            interval: Time in seconds between updates.
        """
        self.stop_timer()  # Stop any existing timer
        self.timer_message = message
        self.timer_running = True
        
        def timer_func():
            counter = 0
            while self.timer_running:
                elapsed = time.time() - self.start_time
                status = f"{self.timer_message} - {format_time(elapsed)} elapsed"
                
                if self.logger:
                    if counter % 6 == 0:  # Log less frequently to avoid flooding logs
                        self.logger.info(status)
                else:
                    print(f"  ⏱️ {status}")
                    
                counter += 1
                time.sleep(interval)
        
        self.timer_thread = threading.Thread(target=timer_func, daemon=True)
        self.timer_thread.start()
    
    def stop_timer(self) -> None:
        """Stop the timer if it's running."""
        if self.timer_thread and self.timer_running:
            self.timer_running = False
            self.timer_thread.join(timeout=1)
            self.timer_thread = None

def format_time(seconds: float) -> str:
    """Format seconds into a human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

def with_progress(task_name: str, logger: Optional[logging.Logger] = None):
    """
    Decorator that adds progress tracking to a function.
    
    Args:
        task_name: Name of the task for progress reporting.
        logger: Optional logger to use instead of print statements.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            progress = ProgressTracker(task_name, logger=logger)
            progress.start()
            
            # Inject the progress tracker into the function arguments
            if 'progress' not in kwargs:
                kwargs['progress'] = progress
                
            try:
                result = func(*args, **kwargs)
                progress.complete()
                return result
            except Exception as e:
                progress.complete(f"Failed with error: {str(e)}")
                raise
        return wrapper
    return decorator

def check_zenml_connection(max_retries=3, retry_delay=5, silent=False):
    """
    Checks if the ZenML server connection is available.
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Seconds to wait between retry attempts
        silent: If True, only prints on error, not on success
        
    Returns:
        bool: True if connection is available, False otherwise
    """
    for attempt in range(1, max_retries + 1):
        try:
            # Only import ZenML if needed (avoid circular imports)
            from zenml.client import Client
            
            if attempt > 1 and not silent:
                print(f"\n🔄 ZENML SERVER RECONNECTION ATTEMPT {attempt}/{max_retries}")
            
            # Try to get the client - this will fail if server is unavailable
            client = Client()
            
            # Try to get server info - this confirms we can actually communicate
            server_info = client.zen_store.get_store_info()
            
            # Handle different ZenML versions
            try:
                # Try to get URL safely
                if hasattr(server_info, 'url'):
                    server_url = server_info.url
                elif hasattr(server_info, 'get_url'):
                    # Some versions use a method instead of an attribute
                    server_url = server_info.get_url()
                elif hasattr(server_info, 'url_str'):
                    server_url = server_info.url_str
                else:
                    # As a fallback, use string representation
                    server_url = str(server_info)
            except Exception as url_err:
                print(f"[DEBUG] ZenML URL extraction error type: {type(url_err).__name__}")
                print(f"[DEBUG] ZenML URL extraction error details: {url_err}")
                server_url = "unknown URL (ZenML version compatibility issue)"
            
            if not silent:
                print("✅ ZenML server connection successful")
                print(f"ℹ️ Connected to: {server_url}")
            
            return True
            
        except Exception as e:
            if "Connection refused" in str(e) or "Failed to connect" in str(e):
                print(f"\n❌ ALERT: ZENML SERVER CONNECTION LOST OR UNAVAILABLE")
                print(f"ℹ️ Error: {str(e)}")
                
                if attempt < max_retries:
                    print(f"ℹ️ Waiting {retry_delay}s before reconnection attempt {attempt+1}/{max_retries}")
                    import time
                    time.sleep(retry_delay)
                else:
                    print(f"\n❌ CRITICAL ERROR: ALL {max_retries} ZENML RECONNECTION ATTEMPTS FAILED")
                    print("ℹ️ Check that the ZenML server is running and accessible")
                    print("ℹ️ You can try restarting the ZenML server with: zenml up")
                    print("ℹ️ Continuing with limited functionality")
            else:
                print(f"\n⚠️ ZENML SERVER WARNING: {str(e)}")
                if attempt < max_retries:
                    import time
                    time.sleep(retry_delay)
                else:
                    return False
    
    return False

def monitor_zenml_connection(interval=60, max_retries=3):
    """
    Starts a background thread to monitor ZenML server connection.
    
    Args:
        interval: Check interval in seconds
        max_retries: Retries for each connection check
        
    Returns:
        threading.Thread: The monitoring thread (daemon)
    """
    def monitor_thread():
        while True:
            check_zenml_connection(max_retries=max_retries, silent=True)
            import time
            time.sleep(interval)
    
    import threading
    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()
    print(f"ℹ️ ZenML connection monitoring started (checking every {interval}s)")
    return monitor

def check_pending_errors(error_file="pending_errors.txt", halt_threshold=2):
    """
    Checks the pending_errors.txt file for accumulated errors and halts
    the pipeline if the error count exceeds the threshold.
    
    Args:
        error_file: Path to the pending errors file
        halt_threshold: Number of errors that will trigger pipeline halt
        
    Returns:
        bool: True if pipeline should continue, False if it should halt
    """
    import os
    from pathlib import Path
    
    error_path = Path(error_file)
    
    # If file doesn't exist, create it and continue
    if not error_path.exists():
        try:
            with open(error_path, "w") as f:
                f.write("# Pending errors file - created on %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
                f.write("# This file tracks errors that need attention\n\n")
            return True
        except Exception as e:
            print(f"[DEBUG] Pending errors file creation error type: {type(e).__name__}")
            print(f"[DEBUG] Pending errors file creation error details: {e}")
            print(f"❌ WARNING: Could not create pending errors file: {e}")
            return True
    
    # Read the file and count error entries
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(error_path, "r", encoding=encoding) as f:
                content = f.read()
            
            # Count error entries - look for ERROR or CRITICAL patterns
            error_count = 0
            for line in content.splitlines():
                if "ERROR:" in line or "CRITICAL:" in line or "❌" in line:
                    error_count += 1
                    # Print the actual error for debugging
                    print(f"Found error in pending_errors.txt: {line.strip()}")
            
            if error_count >= halt_threshold:
                print("\n" + "="*80)
                print(f"❌ CRITICAL: {error_count} PENDING ERRORS DETECTED IN {error_file}")
                print("❌ PIPELINE EXECUTION HALTED FOR SAFETY")
                print(f"ℹ️ Please check {error_file} and resolve the issues before continuing")
                print("ℹ️ After resolving issues, you can clear the file to resume normal operation")
                print("="*80 + "\n")
                return False
            
            if error_count > 0:
                print(f"⚠️ WARNING: {error_count} pending errors found in {error_file}")
                print(f"ℹ️ Pipeline will continue, but please check the errors soon")
                
            # If we got here, we successfully read the file
            return True
                
        except UnicodeDecodeError:
            # Try the next encoding
            continue
        except Exception as e:
            print(f"[DEBUG] Pending errors file read error type: {type(e).__name__}")
            print(f"[DEBUG] Pending errors file read error details: {e}")
            print(f"⚠️ WARNING: Could not read pending errors file with {encoding} encoding: {e}")
            if encoding == 'cp1252':  # Last encoding to try
                # If all encodings fail, create a new file
                try:
                    print(f"Creating a new pending_errors.txt file due to read errors")
                    with open(error_path, "w", encoding='utf-8') as f:
                        f.write(f"# Pending errors file - reset on {time.strftime('%Y-%m-%d %H:%M:%S')} due to read errors\n")
                        f.write(f"# Previous file had encoding issues and could not be read\n\n")
                except Exception as write_err:
                    print(f"[DEBUG] New pending errors file creation error type: {type(write_err).__name__}")
                    print(f"[DEBUG] New pending errors file creation error details: {write_err}")
                    print(f"⚠️ WARNING: Could not create new pending errors file: {write_err}")
            
            continue  # Try next encoding if available
            
    # If we've tried all encodings and still failed
    return True

def log_pending_error(error_message, error_file="pending_errors.txt"):
    """
    Logs an error to the pending_errors.txt file.
    
    Args:
        error_message: The error message to log
        error_file: Path to the pending errors file
    """
    import os
    from pathlib import Path
    
    error_path = Path(error_file)
    
    try:
        # Create file with header if it doesn't exist
        if not error_path.exists():
            with open(error_path, "w", encoding='utf-8') as f:
                f.write("# Pending errors file - created on %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
                f.write("# This file tracks errors that need attention\n\n")
        
        # Append the error message with timestamp (try multiple encodings if needed)
        success = False
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                # First try to read the file to make sure we can append correctly
                try:
                    with open(error_path, "r", encoding=encoding) as f:
                        content = f.read()
                except Exception as read_err:
                    # If read fails, this encoding won't work for appending either
                    print(f"[DEBUG] Error file read test error type: {type(read_err).__name__}")
                    print(f"[DEBUG] Error file read test error details: {read_err}")
                    print(f"Warning: Could not read error file with {encoding} encoding: {read_err}")
                    continue
                
                # If read works, try to append
                with open(error_path, "a", encoding=encoding) as f:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] ERROR: {error_message}\n")
                    
                success = True
                break  # Successfully wrote to file, exit the loop
            except Exception as encoding_err:
                print(f"[DEBUG] Error log encoding {encoding} failed: {type(encoding_err).__name__}: {encoding_err}", file=sys.stderr)
                continue  # Try next encoding
        
        if success:
            print(f"❌ ERROR: {error_message}")
            print(f"ℹ️ This error has been logged to {error_file}")
        else:
            # If all encodings failed, try to create a new file
            try:
                with open(error_path, "w", encoding='utf-8') as f:
                    f.write(f"# Pending errors file - reset on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Previous file had encoding issues and could not be appended to\n\n")
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] ERROR: {error_message}\n")
                print(f"❌ ERROR: {error_message}")
                print(f"ℹ️ Created new error log file with this error")
            except Exception as new_file_error:
                print(f"[DEBUG] New error file creation error type: {type(new_file_error).__name__}")
                print(f"[DEBUG] New error file creation error details: {new_file_error}")
                print(f"❌ ERROR: {error_message}")
                print(f"⚠️ WARNING: Could not create or write to error file: {new_file_error}")
    except Exception as e:
        print(f"[DEBUG] Error logging error type: {type(e).__name__}")
        print(f"[DEBUG] Error logging error details: {e}")
        import traceback
        print(f"[DEBUG] Error logging traceback: {traceback.format_exc()}")
        print(f"❌ ERROR: {error_message}")
        print(f"⚠️ WARNING: Could not log error to file: {e}")

def clear_pending_errors(error_file="pending_errors.txt"):
    """
    Clears the pending_errors.txt file and starts fresh.
    
    Args:
        error_file: Path to the pending errors file
        
    Returns:
        bool: True if successful, False otherwise
    """
    import os
    from pathlib import Path
    
    error_path = Path(error_file)
    
    try:
        # If file exists, create a backup
        if error_path.exists():
            backup_path = error_path.with_suffix('.bak')
            import shutil
            shutil.copy2(error_path, backup_path)
            
        # Create fresh file
        with open(error_path, "w", encoding='utf-8') as f:
            f.write("# Pending errors file - reset on %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            f.write("# This file tracks errors that need attention\n")
            f.write("# Previous errors were cleared\n\n")
            
        print(f"✅ Pending errors file cleared: {error_file}")
        if error_path.exists():
            print(f"ℹ️ Backup saved to: {backup_path}")
            
        return True
    
    except Exception as e:
        print(f"[DEBUG] Clear pending errors error type: {type(e).__name__}")
        print(f"[DEBUG] Clear pending errors error details: {e}")
        import traceback
        print(f"[DEBUG] Clear pending errors traceback: {traceback.format_exc()}")
        print(f"⚠️ WARNING: Could not clear pending errors file: {e}")
        return False

def get_pending_errors(error_file="pending_errors.txt", max_errors=20):
    """
    Reads the pending_errors.txt file and returns a list of errors for review.
    This is designed for Claude or other assistants to review the errors.
    
    Args:
        error_file: Path to the pending errors file
        max_errors: Maximum number of errors to return
        
    Returns:
        list: List of error messages found in the file
    """
    from pathlib import Path
    
    error_path = Path(error_file)
    error_list = []
    
    if not error_path.exists():
        return ["No pending_errors.txt file found. No errors have been logged."]
    
    # Try different encodings to read the file
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(error_path, "r", encoding=encoding) as f:
                content = f.read()
                
            # Extract error entries - look for ERROR or CRITICAL patterns
            for line in content.splitlines():
                if "ERROR:" in line or "CRITICAL:" in line or "❌" in line:
                    error_list.append(line.strip())
                    if len(error_list) >= max_errors:
                        break
                        
            # If we successfully read the file, break out of encoding loop
            break
            
        except UnicodeDecodeError:
            # Try the next encoding
            continue
        except Exception as e:
            print(f"[DEBUG] Get pending errors read error type: {type(e).__name__}")
            print(f"[DEBUG] Get pending errors read error details: {e}")
            return [f"Error reading pending_errors.txt: {str(e)}"]
    
    if not error_list:
        return ["No errors found in pending_errors.txt file."]
        
    return error_list

def scan_pending_errors():
    """
    Function for Claude to scan the pending_errors.txt file.
    This will print all pending errors for review.
    """
    from pathlib import Path
    error_file = "pending_errors.txt"
    error_path = Path(error_file)
    
    print("\n" + "="*80)
    print("PENDING ERRORS SCAN")
    print("="*80)
    
    if not error_path.exists():
        print("No pending_errors.txt file found. No errors have been logged.")
        return
    
    errors = get_pending_errors(error_file)
    
    if not errors or errors[0].startswith("No errors found"):
        print("No errors found in pending_errors.txt file.")
    else:
        print(f"Found {len(errors)} errors in pending_errors.txt:")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
            
    print("\nTo clear all errors after resolving them, use:")
    print("from src.lora_training_pipeline.utils.helpers import clear_pending_errors")
    print("clear_pending_errors()")
    print("="*80 + "\n")

def setup_global_exception_handler():
    """
    Sets up a global unhandled exception handler that will log all
    uncaught exceptions to pending_errors.txt before the program exits.
    
    This ensures that even if an exception occurs in a part of the code
    that doesn't have specific error handling, it will still be logged
    for later review.
    """
    import sys
    import traceback
    
    # Store the original excepthook
    original_excepthook = sys.excepthook
    
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        """
        Custom exception handler that logs to pending_errors.txt
        before calling the original exception handler.
        """
        # Get the formatted traceback
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text = ''.join(tb_lines)
        
        # Prepare error message for logging
        error_message = f"UNHANDLED EXCEPTION: {exc_type.__name__}: {exc_value}\n{tb_text}"
        
        # Log to pending_errors.txt
        log_pending_error(error_message)
        
        # Call the original exception handler
        original_excepthook(exc_type, exc_value, exc_traceback)
    
    # Replace the default exception handler
    sys.excepthook = global_exception_handler
    
    print("✅ Global exception handler installed - all unhandled exceptions will be logged to pending_errors.txt")
    return True

def safe_run(func):
    """
    Decorator that catches any exceptions raised by the wrapped function, 
    logs them to pending_errors.txt, and then re-raises the exception.
    
    This is useful for wrapping important functions that should never fail silently
    and ensures all exceptions are properly logged for later review.
    
    Args:
        func: The function to wrap with error logging.
    
    Returns:
        Wrapped function that logs exceptions to pending_errors.txt.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import traceback
            tb_text = traceback.format_exc()
            error_message = f"ERROR IN {func.__name__}: {str(e)}\n{tb_text}"
            log_pending_error(error_message)
            raise  # Re-raise the exception after logging
    return wrapper