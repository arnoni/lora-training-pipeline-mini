# LoRA_Training_Pipeline/src/data_collection/gradio_app.py

# Check required dependencies first
try:
    import sys
    from pathlib import Path
    import os
    import time
    import traceback
    import datetime
    
    # Install global exception handler for Gradio data collection
    def gradio_data_exception_handler(exc_type, exc_value, exc_traceback):
        """Handle any unhandled exceptions in Gradio data collection with comprehensive logging."""
        error_msg = f"""
CRITICAL UNHANDLED EXCEPTION IN GRADIO DATA COLLECTION
====================================================
Time: {datetime.datetime.now().isoformat()}
Exception Type: {exc_type.__name__}
Exception Value: {exc_value}
Process ID: {os.getpid()}
Working Directory: {os.getcwd()}
Python Executable: {sys.executable}

FULL TRACEBACK:
{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}
"""
        
        # Print to stderr for immediate visibility
        print(error_msg, file=sys.stderr)
        
        # Also try to write to error log file
        try:
            error_file = Path("gradio_data_critical_errors.log")
            with open(error_file, "a", encoding="utf-8") as f:
                f.write(error_msg + "\n" + "="*80 + "\n")
            print(f"[CRITICAL] Gradio data error logged to {error_file}", file=sys.stderr)
        except Exception as log_err:
            print(f"[CRITICAL] Failed to write Gradio data error log: {log_err}", file=sys.stderr)
        
        # Call the original exception handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    # Install the exception handler
    sys.excepthook = gradio_data_exception_handler
    print("[DEBUG] Gradio data collection global exception handler installed")
    
    # Add the project root to PATH so we can import our module
    root_dir = Path(__file__).resolve().parents[3]  # Go up 3 levels from current file
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.lora_training_pipeline.utils.helpers import check_dependencies, log_pending_error
except ImportError as e:
    print(f"ERROR: Missing critical dependency: {e}")
    print("Please make sure all dependencies are installed with: uv pip install -e .")
    
    # Try to log the error directly to pending_errors.txt if helpers module isn't available
    try:
        error_path = Path("pending_errors.txt")
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(error_path, "a") as f:
            f.write(f"[{timestamp}] ERROR: Missing critical dependency in gradio_app.py: {e}\n")
    except Exception as file_error:
        print(f"CRITICAL: Failed to log error to file: {file_error}")
        print(f"Original error was: Missing critical dependency in gradio_app.py: {e}")
    
    sys.exit(1)

# Check specific dependencies for this script
check_dependencies(['gradio', 'fastapi', 'httpx', 'pandas', 'psutil'])

import gradio as gr

# Print Gradio version information and set up compatibility flags
try:
    print("\n" + "="*80)
    print("GRADIO VERSION INFORMATION")
    print("="*80)
    
    # Get and display version
    GRADIO_VERSION = gr.__version__
    print(f"Gradio Package Version: {GRADIO_VERSION}")
    
    # Try to parse the version into components
    version_parts = GRADIO_VERSION.split('.')
    MAJOR_VERSION = int(version_parts[0]) if len(version_parts) > 0 and version_parts[0].isdigit() else 0
    MINOR_VERSION = int(version_parts[1]) if len(version_parts) > 1 and version_parts[1].isdigit() else 0
    
    # Set compatibility flags based on version
    SUPPORTS_THEMES = MAJOR_VERSION >= 3
    # Gradio doesn't consistently support concurrency_count across versions
    # We know from the error message that our current version doesn't support it
    SUPPORTS_QUEUE_CONFIG = False  # Default to False regardless of version
    SUPPORTS_JS_LOAD = MAJOR_VERSION >= 3 and MINOR_VERSION >= 20
    SUPPORTS_ELEM_ID = MAJOR_VERSION >= 3
    
    print(f"Parsed Version: Major={MAJOR_VERSION}, Minor={MINOR_VERSION}")
    print(f"Compatibility Flags:")
    print(f"- Supports Themes: {SUPPORTS_THEMES}")
    print(f"- Supports Queue Config: {SUPPORTS_QUEUE_CONFIG}")
    print(f"- Supports JS Load: {SUPPORTS_JS_LOAD}")
    print(f"- Supports Element IDs: {SUPPORTS_ELEM_ID}")
    
    # Try to check for additional version details
    if hasattr(gr, 'version'):
        print(f"Gradio Version Details: {gr.version}")
    
    # Print the module location
    print(f"Gradio Module Location: {gr.__file__}")
    
    # Print some available components for debugging
    print(f"Available Gradio Components (sample): {str(list(filter(lambda x: not x.startswith('_'), dir(gr))))[:200]}...")
    print("="*80 + "\n")
except Exception as e:
    print(f"Error getting Gradio version information: {e}")
    # Set default compatibility flags for older versions
    SUPPORTS_THEMES = False
    SUPPORTS_QUEUE_CONFIG = False
    SUPPORTS_JS_LOAD = False
    SUPPORTS_ELEM_ID = False
    GRADIO_VERSION = "Unknown"
from fastapi import FastAPI, HTTPException
import httpx
import pandas as pd
from pathlib import Path
import datetime
import os
from typing import Dict

# --- Configuration ---
DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
print(f"📁 Data directory created/verified at: {DATA_DIR.absolute()}")
DATASET_NAME = "user_data_ui"
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
print(f"🔗 FastAPI URL set to: {FASTAPI_URL}")

# --- FastAPI App ---
app = FastAPI()

# --- Data Counter ---
def get_data_count():
    """Counts the total number of data points collected."""
    count = 0
    try:
        # Make sure we're using the absolute path
        data_dir_abs = DATA_DIR.absolute()
        print(f"📊 Counting data in directory: {data_dir_abs}")
        
        # Check if directory exists
        if not data_dir_abs.exists():
            print(f"📊 Data directory doesn't exist. Creating it now.")
            data_dir_abs.mkdir(parents=True, exist_ok=True)
            return "0"
            
        # Check if directory is readable
        if not os.access(data_dir_abs, os.R_OK):
            error_msg = f"Data directory {data_dir_abs} is not readable!"
            print(f"⚠️ WARNING: {error_msg}")
            log_pending_error(f"Data count failed - {error_msg}")
            return "0"
        
        # Look for parquet files
        pattern = f"{DATASET_NAME}_v1_original_*.parquet"
        print(f"📊 Searching for pattern: {pattern}")
        all_files = list(data_dir_abs.glob(pattern))
        
        if not all_files:
            print("📊 No existing data files found. This appears to be the first run.")
            print("📊 New data files will be created when users submit text.")
            return 0
            
        print(f"📊 Found {len(all_files)} existing data files:")
        for file in all_files:
            print(f"   - {file.name}")
            try:
                df = pd.read_parquet(file)
                file_count = len(df)
                count += file_count
                print(f"     Contains {file_count} data points")
            except Exception as file_e:
                print(f"[DEBUG] File reading error type: {type(file_e).__name__}")
                print(f"[DEBUG] File reading error details: {file_e}")
                file_error = f"Error reading file {file.name}: {file_e}"
                print(f"     ⚠️ {file_error}")
                log_pending_error(f"Data count issue - {file_error}")
                import traceback
                print(f"[DEBUG] File reading error traceback: {traceback.format_exc()}")
                # Continue to next file
                
    except Exception as e:
        error_msg = f"Error reading data files: {e}"
        print(f"❌ {error_msg}")
        
        # Get detailed traceback
        import traceback
        tb = traceback.format_exc()
        print(f"Traceback: {tb}")
        
        # Log the error with traceback
        log_pending_error(f"Data count failed - {error_msg}\nTraceback: {tb}")
        return "0"
    
    print(f"📊 Total data points found: {count}")
    return str(count)

@app.post("/submit_data/")
async def submit_data(data: Dict):  # Removed authentication dependency
    """Receives data from Gradio, saves it as Parquet."""
    try:
        text = data.get("text")
        if not text or not isinstance(text, str) or text.strip() == "":
            error_msg = "Invalid text input"
            log_pending_error(f"API submission failed - {error_msg}")
            raise ValueError(error_msg)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version = "v1"
        variation = "original"
        filename = f"{DATASET_NAME}_{version}_{variation}_{timestamp}.parquet"
        
        # Use absolute path
        data_dir_abs = DATA_DIR.absolute()
        filepath = data_dir_abs / filename
        
        print(f"📝 API saving data to: {filepath}")
        
        # Ensure directory exists and is writable
        data_dir_abs.mkdir(parents=True, exist_ok=True)
        if not os.access(data_dir_abs, os.W_OK):
            error_msg = f"API directory not writable: {data_dir_abs}"
            print(f"⚠️ {error_msg}")
            log_pending_error(error_msg)
            raise PermissionError(error_msg)

        df = pd.DataFrame([{"text": text, "timestamp": timestamp}])
        df.to_parquet(filepath)
        
        print(f"✅ API save successful to: {filepath}")
        return {"message": f"Data saved to {filepath}", "filename": filename}

    except ValueError as ve:
        error_msg = f"Validation error: {ve}"
        print(f"⚠️ {error_msg}")
        log_pending_error(f"API validation error - {error_msg}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        error_msg = f"Error saving data: {e}"
        print(f"❌ {error_msg}")
        
        # Get detailed traceback
        import traceback
        tb = traceback.format_exc()
        print(f"Traceback: {tb}")
        
        # Log the error with traceback
        log_pending_error(f"API save failed - {error_msg}\nTraceback: {tb}")
        
        raise HTTPException(status_code=500, detail=error_msg)


# --- Direct Data Saving ---
def save_data_directly(text: str) -> str:
    """Saves data directly to a parquet file if the API is unavailable."""
    try:
        if not text or not isinstance(text, str) or text.strip() == "":
            error_msg = "Error: Text input was empty, cannot save"
            print(f"⚠️ {error_msg}")
            log_pending_error(f"Data save failed - {error_msg}")
            return error_msg
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version = "v1"
        variation = "original"
        filename = f"{DATASET_NAME}_{version}_{variation}_{timestamp}.parquet"
        
        # Create absolute path to ensure we know where it's going
        data_dir_abs = DATA_DIR.absolute()
        filepath = data_dir_abs / filename
        
        print(f"💾 Attempting to save data to: {filepath}")
        
        # Ensure data directory exists
        data_dir_abs.mkdir(parents=True, exist_ok=True)
        
        # Verify directory is writable
        if not os.access(data_dir_abs, os.W_OK):
            writable_err = f"Directory {data_dir_abs} is not writable!"
            print(f"⚠️ WARNING: {writable_err}")
            log_pending_error(f"Data directory not writable - {writable_err}")
            
            # Try creating an alternate directory in user's home
            alt_dir = Path.home() / "lora_data"
            alt_dir.mkdir(parents=True, exist_ok=True)
            filepath = alt_dir / filename
            print(f"📁 Trying alternate directory: {alt_dir}")
            log_pending_error(f"Attempting to use alternate directory: {alt_dir}")
            
        # Create dataframe and save
        df = pd.DataFrame([{"text": text, "timestamp": timestamp}])
        df.to_parquet(filepath)
        
        print(f"✅ Data successfully saved to: {filepath}")
        return f"Data saved directly to {filepath}"
    except Exception as e:
        error_msg = f"Error saving data directly: {e}"
        print(f"❌ {error_msg}")
        
        # Get traceback for detailed error information
        import traceback
        tb = traceback.format_exc()
        print(f"Traceback: {tb}")
        
        # Log the error with traceback to pending_errors.txt
        log_pending_error(f"Data save failed - {error_msg}\nTraceback: {tb}")
        
        return error_msg

# --- Gradio Interface ---
def send_data(text): # Removed auth
    """
    Sends data to FastAPI, or saves directly if API is unavailable.
    This ensures the data collection app works independently of other services.
    """
    print(f"\n{'='*50}")
    print(f"📤 SEND DATA REQUEST RECEIVED")
    print(f"{'='*50}")
    
    if not text or not isinstance(text, str) or text.strip() == "":
        error_msg = "Empty text received, cannot process"
        print(f"⚠️ Error: {error_msg}")
        log_pending_error(f"Data submission failed - {error_msg}")
        return "Error: Text cannot be empty", get_data_count(), text
        
    print(f"📋 Text to save: {text[:30]}{'...' if len(text) > 30 else ''}")
    print(f"📁 Current data directory: {DATA_DIR.absolute()}")
    print(f"🔢 Current data count before save: {get_data_count()}")
    
    # Try API first
    try:
        print(f"🔗 Attempting to send data to API: {FASTAPI_URL}/submit_data/")
        response = httpx.post(f"{FASTAPI_URL}/submit_data/", json={"text": text}, timeout=5)  # Shorter timeout
        response.raise_for_status()
        message = response.json()["message"]
        print(f"✅ API save successful: {message}")
        
        # Update count after saving
        new_count = get_data_count()
        print(f"🔢 Updated data count: {new_count}")
        
        return f"✅ {message}", new_count, ""  # Return updated count and clear input
    except (httpx.RequestError, httpx.HTTPStatusError, Exception) as e:
        # If API fails, save directly
        api_error = f"API error: {e}"
        print(f"🔴 {api_error}")
        log_pending_error(f"API submission failed - {api_error}")
        print(f"💾 Falling back to direct save method...")
        
        message = save_data_directly(text)
        
        # Get updated count after save attempt
        new_count = get_data_count()
        print(f"🔢 Data count after direct save attempt: {new_count}")
        
        if message.startswith("Error"):
            print(f"⚠️ Direct save failed: {message}")
            # Error already logged in save_data_directly
            return f"⚠️ {message}", new_count, text
        else:
            print(f"✅ Direct save successful: {message}")
            # Explicitly return empty string for text_input to clear it
            return f"✅ {message} (API was unavailable, used direct save)", new_count, ""

# Create a persistent Gradio interface that won't auto-refresh
# Use theme if supported by the Gradio version
if SUPPORTS_THEMES and hasattr(gr, 'themes') and hasattr(gr.themes, 'Soft'):
    ui = gr.Blocks(title="Data Collection UI - LoRA Training Pipeline", theme=gr.themes.Soft())
else:
    ui = gr.Blocks(title="Data Collection UI - LoRA Training Pipeline")

with ui:
    # Create a state variable to store the data count and status
    state = gr.State({"count": get_data_count(), "last_message": "", "success": True})
    
    # Define status update function that works with state
    def update_state(new_state, current_state):
        current_state.update(new_state)
        return current_state
    
    # Header with important information
    gr.Markdown("""
    # 📊 Data Collection Interface
    
    Use this interface to submit text samples for training the LoRA model.
    """)
    
    # Add persistent status indicators
    with gr.Row(equal_height=True):
        count_display = gr.Textbox(
            label="Total Data Points", 
            value=get_data_count(),
            interactive=False
        )
        
        status_display = gr.Textbox(
            label="System Status",
            value="✅ Ready to collect data",
            interactive=False
        )
    
    # Main data collection area
    gr.Markdown("### Enter your text sample below:")
    
    # Text input area - conditionally add elem_id based on Gradio version support
    text_input_args = {
        "label": "",
        "placeholder": "Type your text here...",
        "lines": 8,
        "max_lines": 15
    }
    
    # Add elem_id if supported
    if SUPPORTS_ELEM_ID:
        text_input_args["elem_id"] = "text_input_box"
        
    text_input = gr.Textbox(**text_input_args)
    
    # Result display - conditionally add elem_id based on Gradio version support
    result_display_args = {
        "label": "Last Operation Result:",
        "value": "Enter text and click 'Submit' to add data",
        "interactive": False,
        "lines": 2
    }
    
    # Add elem_id if supported
    if SUPPORTS_ELEM_ID:
        result_display_args["elem_id"] = "result_display"
        
    result_display = gr.Textbox(**result_display_args)
    
    # Submit button
    submit_btn = gr.Button("Submit Text Sample", variant="primary")
    
    # Technical details (collapsed by default)
    with gr.Accordion("Technical Details", open=False):
        gr.Markdown(f"""
        - Data Directory: `{DATA_DIR.absolute()}`
        - Last Server Start: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        - Directory Status: {'✅ Writable' if DATA_DIR.exists() and os.access(DATA_DIR, os.W_OK) else '⚠️ Not writable'}
        """)
        
        # Add a manual refresh button for the data count
        refresh_btn = gr.Button("Refresh Data Count")
    
    # Define the submission handler with state management
    def handle_submission(text, state_data):
        try:
            print(f"Processing submission with state: {state_data}")
            
            # Call the send_data function to process the submission
            result, count, _ = send_data(text)
            
            # Update the state
            new_state = {
                "count": count,
                "last_message": result,
                "success": not result.startswith("Error") and not result.startswith("⚠️")
            }
            
            # Determine status display
            status = "✅ Ready to collect data"
            if not new_state["success"]:
                status = "⚠️ Last operation had an error"
            
            # Clear the text input only on success
            new_input = "" if new_state["success"] else text
            
            # Update the state
            updated_state = update_state(new_state, state_data)
            
            # Return all updated values
            return new_input, result, count, status, updated_state
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error_msg = f"UI Error: {e}\n{tb}"
            print(f"❌ {error_msg}")
            log_pending_error(error_msg)
            
            # Return error state
            return text, f"⚠️ UI Error: {e}", state_data["count"], "⚠️ System error occurred", state_data
    
    # Define a refresh function that only updates the count
    def refresh_count(state_data):
        new_count = get_data_count()
        print(f"Refreshing data count: {new_count}")
        state_data["count"] = new_count
        return new_count, state_data
    
    # Connect the submit button
    submit_btn.click(
        handle_submission,
        inputs=[text_input, state],
        outputs=[text_input, result_display, count_display, status_display, state]
    )
    
    # Connect the refresh button
    refresh_btn.click(
        refresh_count,
        inputs=[state],
        outputs=[count_display, state]
    )
    
    # Set initial values
    initial_count = get_data_count()
    count_display.value = str(initial_count)
    status_display.value = "✅ Ready to collect data"
    
    # Add a simple refresh function
    def refresh_values():
        current_count = get_data_count()
        return current_count

if __name__ == "__main__":
    # Handle signal management for Gradio
    # Disable default Gradio signal handlers if requested through env var
    disable_signal_handlers = os.environ.get("GRADIO_DISABLE_SIGNAL_HANDLERS", "").lower() == "1"
    if disable_signal_handlers:
        print("🛡️ Disabling default Gradio signal handlers as requested by environment variable")
        # Import signal module and set up minimal handlers
        import signal

        # Define a custom signal handler that just logs the signal
        def custom_signal_handler(sig, frame):
            print(f"SIGNAL RECEIVED: {signal.Signals(sig).name} - Handled by custom handler")

        # Register for common signals but don't actually terminate
        # This prevents the KeyboardInterrupt during module imports
        signal.signal(signal.SIGINT, custom_signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows-specific
            signal.signal(signal.SIGBREAK, custom_signal_handler)
        signal.signal(signal.SIGTERM, custom_signal_handler)

    # Check if startup delay is specified to prevent race conditions
    import time
    startup_delay = os.environ.get("GRADIO_STARTUP_DELAY", "0")
    try:
        delay_seconds = float(startup_delay)
        if delay_seconds > 0:
            print(f"Startup delay of {delay_seconds} seconds requested by environment variable")
            time.sleep(delay_seconds)
    except ValueError as delay_err:
        print(f"[DEBUG] Startup delay parsing error type: {type(delay_err).__name__}")
        print(f"[DEBUG] Startup delay parsing error details: {delay_err}")
        print(f"⚠️ WARNING: Invalid GRADIO_STARTUP_DELAY value '{startup_delay}', ignoring delay")
        pass

    # Print Python environment information for debugging
    print("\n" + "="*80)
    print("PYTHON ENVIRONMENT INFORMATION")
    print("="*80)
    import sys
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Path: {sys.path}")

    # Print environment variables for debugging
    import os
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"Working Directory: {os.getcwd()}")
    print("="*80 + "\n")

    import socket
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        print("⚠️ Warning: psutil module not available. Limited diagnostics will be shown.")
        print("ℹ️ Install psutil with: pip install psutil")
        PSUTIL_AVAILABLE = False

    print("\n" + "="*80)
    print("GRADIO DATA COLLECTION STARTUP DIAGNOSTICS")
    print("="*80)

    # Get port from environment or use default
    DEFAULT_PORT = 7862  # Change from 7860 to 7862
    GRADIO_PORT = int(os.environ.get("GRADIO_PORT", DEFAULT_PORT))
    print(f"📡 Using Gradio port: {GRADIO_PORT}")

    # Use ProcessWatchdog to ensure only one instance runs and properly manage process resources
    try:
        from src.lora_training_pipeline.utils.process_watchdog import ProcessWatchdog

        print(f"Checking for existing Gradio Data Collection instances on port {GRADIO_PORT}...")

        # Create process watchdog for data collection UI
        watchdog = ProcessWatchdog("data_collection", GRADIO_PORT)
        
        # Check if we can start the service
        if not watchdog.can_start():
            print(f"❌ ERROR: {watchdog.error_message}")
            print("The Data Collection UI cannot start.")
            print("To check running services and clean up stale processes:")
            print("python -m src.lora_training_pipeline.utils.process_watchdog --status")
            print("python -m src.lora_training_pipeline.utils.process_watchdog --cleanup")
            sys.exit(1)

        print(f"✅ No other Gradio Data Collection instances running on port {GRADIO_PORT}")

        # Register the current process PID
        pid = os.getpid()
        if not watchdog.register_pid(pid, {
            "app_type": "gradio",
            "app_name": "data_collection",
            "start_time": datetime.datetime.now().isoformat()
        }):
            print(f"⚠️ Warning: Failed to register PID: {watchdog.error_message}")
            print("The application will continue, but process management may be affected.")

        # The watchdog automatically registers cleanup handlers for signals and atexit

    except ImportError as e:
        print(f"Warning: ProcessWatchdog not available: {e}")
        print("Falling back to basic port checking")

        # Basic port check
        def check_port(port, host='127.0.0.1', timeout=1.0, retries=2):
            """
            Check if a port is in use with retries for reliable detection.
            
            Args:
                port: Port number to check
                host: Host to check (default: 127.0.0.1)
                timeout: Socket timeout in seconds
                retries: Number of connection attempts
                
            Returns:
                bool: True if port is in use, False otherwise
            """
            for attempt in range(retries):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(timeout)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    
                    # If connection succeeded, port is definitely in use
                    if result == 0:
                        return True
                        
                    # If we've made all attempts and still not connected, port is available
                    if attempt == retries - 1:
                        return False
                        
                    # Short delay between retries for more reliable detection
                    time.sleep(0.5)
                    
                except socket.error as e:
                    print(f"Socket error checking port {port}: {e}")
                    # Continue to next retry
                    time.sleep(0.5)
            
            # If we reach here, all retries failed - assume port might be in use to be safe
            return True

        # Debug: Check if configured port is already in use
        port_in_use = check_port(GRADIO_PORT)
        print(f"📡 Port {GRADIO_PORT} status: {'IN USE' if port_in_use else 'AVAILABLE'}")

        if port_in_use:
            print(f"❌ ERROR: Port {GRADIO_PORT} is already in use by another process")
            print("The UI cannot start because the port is already in use.")
            print("Try using a different port or close the existing process.")
            sys.exit(1)

        # Simple PID file management
        try:
            pid_file = "./data_collection_ui.pid"
            pid = os.getpid()

            # Write PID file
            with open(pid_file, 'w') as f:
                import json
                import time
                json.dump({
                    "pid": pid,
                    "port": GRADIO_PORT,
                    "process_type": "gradio_data_collection",
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')
                }, f, indent=2)

            print(f"✅ Wrote PID file {pid_file} with PID {pid}")

            # Register cleanup handler
            import atexit
            def cleanup_pid_file():
                try:
                    if os.path.exists(pid_file):
                        os.unlink(pid_file)
                        print(f"Removed PID file {pid_file}")
                except Exception as e:
                    print(f"Error removing PID file: {e}")

            atexit.register(cleanup_pid_file)

            # Register signal handlers
            import signal
            def signal_handler(sig, frame):
                print(f"Received signal {sig}, cleaning up")
                cleanup_pid_file()
                sys.exit(0)

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

        except Exception as e:
            print(f"Warning: Cannot write PID file: {e}")

    # Debug: Display network interfaces
    print("\n📡 Network interfaces:")
    try:
        hostname = socket.gethostname()
        print(f"   - Hostname: {hostname}")
        print(f"   - Localhost: 127.0.0.1")

        # Get all available IP addresses
        ipaddrs = []

        if PSUTIL_AVAILABLE:
            # Use psutil if available
            for iface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:  # IPv4
                        ipaddrs.append((iface, addr.address))
                        print(f"   - Interface {iface}: {addr.address}")
        else:
            # Fallback: Try to get at least the hostname IP
            try:
                host_ip = socket.gethostbyname(hostname)
                ipaddrs.append(("default", host_ip))
                print(f"   - Default IP: {host_ip}")
            except Exception as ip_error:
                print(f"   - Unable to determine IP address: {ip_error}")

        if not ipaddrs:
            print("   - No IPv4 addresses found!")
    except Exception as e:
        print(f"   - Error identifying network interfaces: {e}")

    # Debug: Check data directory
    print(f"\n📂 Data directory: {DATA_DIR.absolute()}")
    print(f"   - Directory exists: {DATA_DIR.exists()}")
    print(f"   - Directory is writable: {os.access(DATA_DIR, os.W_OK) if DATA_DIR.exists() else 'N/A'}")

    # Use 0.0.0.0 to allow connections from any IP address, making the UI accessible from the internet
    # Use share=True to create a public URL through Gradio's sharing service
    print("\n🚀 LAUNCHING DATA COLLECTION UI: Starting Gradio interface...")
    print("ℹ️ The interface will be accessible at:")
    print(f"ℹ️ - Locally: http://localhost:{GRADIO_PORT}")
    print(f"ℹ️ - Network: http://{ipaddrs[0][1] if ipaddrs else '<your-ip-address>'}:{GRADIO_PORT}")
    print("ℹ️ - Public: A public URL will be displayed when Gradio starts")
    print("="*80)
    
    # Try different combinations of server names and ports
    server_options = [
        {"server_name": "0.0.0.0", "description": "All interfaces", "ports": [GRADIO_PORT, GRADIO_PORT+1, GRADIO_PORT+2]},
        {"server_name": "127.0.0.1", "description": "Localhost only", "ports": [GRADIO_PORT, GRADIO_PORT+1, GRADIO_PORT+2]},
    ]
    
    # Add the actual IP address of the machine if available
    if ipaddrs:
        server_options.append({
            "server_name": ipaddrs[0][1], 
            "description": f"Specific interface ({ipaddrs[0][0]})", 
            "ports": [7860, 7861, 7862]
        })
    
    # Detailed diagnostic mode - try to launch with multiple configurations
    success = False
    for option in server_options:
        if success:
            break
            
        for port in option["ports"]:
            # Check if this port is available
            if check_port(port):
                print(f"\n⚠️ Port {port} is already in use, skipping...")
                continue
                
            try:
                print(f"\n🔄 Trying: server_name={option['server_name']} port={port} ({option['description']})")
                # Basic launch parameters supported by all Gradio versions
                launch_params = {
                    "server_name": option["server_name"],
                    "server_port": port,
                    "share": True,
                    "prevent_thread_lock": True,  # Allows the app to run in parallel with other Gradio apps
                    "quiet": False,               # Show all messages for debugging
                    "debug": True,                # Enable debug mode for more information
                    "inbrowser": True             # Try to open in browser automatically
                }
                
                # Advanced parameters for newer Gradio versions
                if MAJOR_VERSION >= 2:
                    # These parameters should work with Gradio 2.x and 3.x
                    launch_params.update({
                        "favicon_path": None,     # Use default favicon to prevent file locking issues
                        "show_api": False         # Disable API view to prevent any potential conflicts
                    })
                
                # We know from the error message that concurrency_count isn't supported
                # in this version of Gradio, so we'll use a simpler queue configuration
                try:
                    # Try the simplest launch method first
                    print("Trying direct launch without queue...")
                    ui.launch(**launch_params)
                except Exception as e1:
                    print(f"Direct launch failed: {e1}")
                    try:
                        # Then try with queue max_size
                        print("Trying with queue(max_size=20)...")
                        ui.queue(max_size=20).launch(**launch_params)
                    except TypeError as e2:
                        print(f"Queue with max_size failed: {e2}")
                        # If that also fails, use the simplest queue configuration
                        print("Falling back to basic queue configuration")
                        try:
                            ui.queue().launch(**launch_params)
                        except Exception as e3:
                            print(f"All queue configurations failed: {e3}")
                            # Final minimal fallback
                            print("Using absolute minimal configuration")
                            try:
                                print("Trying 127.0.0.1 as server name...")
                                ui.launch(server_name="127.0.0.1", server_port=port)
                            except Exception as e4:
                                print(f"127.0.0.1 also failed: {e4}")
                                print("Final attempt with localhost and different port...")
                                # Use a different port as last resort
                                fallback_port = port + 10  # Use port 7872 instead of 7862
                                print(f"Using fallback port {fallback_port}")
                                ui.launch(server_name="localhost", server_port=fallback_port)
                
                print(f"\n✅ SUCCESS: Gradio server started on {option['server_name']}:{port}")
                print(f"✅ The interface should now be accessible at http://{option['server_name']}:{port}")
                print(f"✅ If using WSL, try http://localhost:{port} in your Windows browser")
                
                success = True
                break
            except Exception as e:
                print(f"❌ ERROR: Failed to start with {option['description']} on port {port}: {e}")
                
    if not success:
        print("\n" + "="*80)
        print("❌ ALL LAUNCH ATTEMPTS FAILED")
        print("="*80)
        print("The Gradio Data Collection UI couldn't be started. Please check:")
        print("1. Is Gradio already running on all the attempted ports?")
        print("2. Do you have proper network permissions?")
        print("3. Is there a firewall blocking the connection?")
        print(f"4. Are you using WSL? Try http://localhost:{GRADIO_PORT} in your Windows browser")
        print("\nAdditional troubleshooting steps:")
        print(f"- Run 'netstat -ano | findstr {GRADIO_PORT}' (Windows) or 'sudo netstat -tulpn | grep {GRADIO_PORT}' (Linux)")
        print("- Try changing the port manually in the code")
        print("- Check if other Gradio instances are running")