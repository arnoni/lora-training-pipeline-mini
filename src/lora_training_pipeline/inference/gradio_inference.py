# LoRA_Training_Pipeline/src/inference/gradio_inference.py

# Check required dependencies first
try:
    import sys
    from pathlib import Path
    import os
    # Add the project root to PATH so we can import our module
    root_dir = Path(__file__).resolve().parents[3]  # Go up 3 levels from current file
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.lora_training_pipeline.utils.helpers import check_dependencies
except ImportError as e:
    print(f"ERROR: Missing critical dependency: {e}")
    print("Please make sure all dependencies are installed with: uv pip install -e .")
    sys.exit(1)

# Check specific dependencies for this script
check_dependencies(['gradio', 'httpx'])

import gradio as gr
import httpx
import os
import datetime

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

# --- Configuration ---
# Get configuration from environment variables with detailed logging
print("\n" + "="*80)
print("GRADIO INFERENCE UI CONFIGURATION")
print("="*80)

# Check all possible environment variables that might be set
possible_url_vars = [
    "FASTAPI_INFERENCE_URL", 
    "INFERENCE_API_URL", 
    "FASTAPI_URL"
]

# Print all environment variables that might contain the URL
print("Environment variables that might contain FastAPI URL:")
for var in possible_url_vars:
    print(f"  {var}: {os.environ.get(var, 'Not set')}")

# Get FastAPI port
inference_port = os.environ.get("FASTAPI_INFERENCE_PORT", "8001")
print(f"FASTAPI_INFERENCE_PORT: {inference_port}")

# Set the URL with fallbacks in order of preference
FASTAPI_INFERENCE_URL = (
    os.environ.get("FASTAPI_INFERENCE_URL") or 
    os.environ.get("INFERENCE_API_URL") or 
    f"http://127.0.0.1:{inference_port}"
)

print(f"Using FastAPI URL: {FASTAPI_INFERENCE_URL}")

# Check model path
OUTPUT_DIR = os.path.join(".", "output", "best_model")
print(f"Model directory path: {OUTPUT_DIR}")
print(f"Model directory exists: {os.path.exists(OUTPUT_DIR)}")
if os.path.exists(OUTPUT_DIR):
    try:
        print(f"Model directory contents: {os.listdir(OUTPUT_DIR)}")
    except Exception as e:
        print(f"Error listing model directory contents: {e}")
else:
    print("Model directory does not exist yet.")
    # Create parent directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
        print(f"Created parent directory for model: {os.path.dirname(OUTPUT_DIR)}")
    except Exception as e:
        print(f"Error creating parent directory: {e}")

# Check active network interfaces to help with connectivity issues
try:
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Local hostname: {hostname}")
    print(f"Local IP address: {local_ip}")
except Exception as e:
    print(f"Error getting network information: {e}")

print("="*80)

# --- Server Status Management ---
def check_server_status():
    """
    Checks if the inference server is running and has a model loaded.
    
    Returns:
        tuple: (server_running, model_loaded, status_message)
    """
    # Print debug information about the server URL configuration
    print(f"\n" + "="*80)
    print(f"SERVER CONNECTION CHECK")
    print(f"="*80)
    print(f"FASTAPI_INFERENCE_URL: {FASTAPI_INFERENCE_URL}")
    # Print ALL environment variables for maximum debugging
    print(f"ALL Environment variables:")
    for key, value in sorted(os.environ.items()):
        print(f"- {key}: {value}")
    print(f"="*80)
    print(f"SERVER CONNECTION SETTINGS:")
    print(f"- FASTAPI_INFERENCE_URL: {os.environ.get('FASTAPI_INFERENCE_URL', 'Not set')}")
    print(f"- INFERENCE_API_URL: {os.environ.get('INFERENCE_API_URL', 'Not set')}")
    print(f"- FASTAPI_INFERENCE_PORT: {os.environ.get('FASTAPI_INFERENCE_PORT', 'Not set')}")
    print(f"- FASTAPI_HOST: {os.environ.get('FASTAPI_HOST', 'Not set')}")
    print(f"- FASTAPI_URL: {os.environ.get('FASTAPI_URL', 'Not set')}")
    print(f"- DEBUG_SERVER_CONNECTION: {os.environ.get('DEBUG_SERVER_CONNECTION', 'Not set')}") 
    # Print Python module info
    import sys
    print(f"Python version: {sys.version}")
    print(f"Python exec: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    
    # Verify network connectivity with enhanced reliability for Windows/WSL
    try:
        import socket
        
        # Parse URL with special handling for localhost
        url_parts = FASTAPI_INFERENCE_URL.split('://')
        if len(url_parts) > 1:
            host_part = url_parts[1].split(':')[0]
        else:
            host_part = "localhost"  # Default if URL format is unexpected
            
        # Special handling for localhost/127.0.0.1
        if host_part == "localhost" or host_part == "127.0.0.1":
            # Try multiple localhost variants since Windows/WSL can be picky
            hosts_to_try = ["localhost", "127.0.0.1", "::1"]
        else:
            hosts_to_try = [host_part]
            
        # Get port with fallback
        try:
            port_part = FASTAPI_INFERENCE_URL.split(':')[-1]
            if '/' in port_part:
                port_part = port_part.split('/')[0]
            port = int(port_part)
        except Exception as e:
            print(f"Warning: Could not parse port from URL: {e}")
            port = 8001  # Default port if parsing fails
        
        print(f"Testing connectivity to FastAPI server on port {port}...")
        
        # Try multiple hostnames that might work for localhost
        connection_successful = False
        for host in hosts_to_try:
            try:
                print(f"  - Trying to connect to {host}:{port}...")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(3)  # Longer timeout for better reliability
                result = s.connect_ex((host, port))
                s.close()
                
                if result == 0:
                    print(f"✅ Successfully connected to {host}:{port}")
                    connection_successful = True
                    break
                else:
                    print(f"❌ Could not connect to {host}:{port} (error code: {result})")
            except Exception as conn_err:
                print(f"❌ Error connecting to {host}:{port}: {conn_err}")
                
        if not connection_successful:
            print(f"❌ Failed to connect to FastAPI server on any hostname variant")
    except Exception as e:
        print(f"❌ Error testing port: {e}")
    
    # If we found a successful host connection above, use that for HTTP calls
    server_url = FASTAPI_INFERENCE_URL
    if 'connection_successful' in locals() and connection_successful and 'host' in locals():
        # Replace the hostname in the URL with the one that worked
        if "localhost" in server_url or "127.0.0.1" in server_url:
            # Only modify localhost URLs
            server_url = f"http://{host}:{port}"
            print(f"Using working connection: {server_url}")
    
    try:
        # Test HTTP connectivity after basic TCP check
        print(f"Attempting HTTP connection to inference server...")
        
        # First check if server is running at all via health endpoint
        print(f"GET {server_url}/health")
        start_time = time.time()
        try:
            # Use the server_url that worked in the socket test
            response = httpx.get(f"{server_url}/health", timeout=5)
            end_time = time.time()
            print(f"✅ Server health check succeeded: {response.status_code} (took {end_time - start_time:.2f}s)")
            print(f"Response headers: {response.headers}")
            print(f"Response body: {response.text}")
            
            # Now check if model is loaded
            print(f"GET {server_url}/model-info")
            start_time = time.time()
            model_response = httpx.get(f"{server_url}/model-info", timeout=5)
            end_time = time.time()
            print(f"✅ Model info check: {model_response.status_code} (took {end_time - start_time:.2f}s)")
            print(f"Response headers: {model_response.headers}")
            print(f"Response body: {model_response.text}")
            
            # Parse the JSON response
            model_data = model_response.json()
            print(f"Model loaded: {model_data.get('model_loaded', False)}")
            print(f"Model path: {model_data.get('model_path', 'Not available')}")
            
            if model_response.status_code == 200 and model_data.get("model_loaded"):
                return True, True, "Model loaded and ready for inference"
            else:
                # Server is running but model is not loaded - this is a normal state
                # before training has completed
                status = model_data.get("status", "unknown")
                if status == "no_model":
                    return True, False, "Server running, waiting for model training to complete"
                else:
                    return True, False, f"Server running, model status: {status}"
        except httpx.ConnectError as e:
            print(f"❌ Connection refused: {e}")
            print(f"The server at {server_url} is not accepting connections")
            print(f"Please check if the FastAPI inference server is running")
        except httpx.TimeoutException as e:
            print(f"❌ Request timed out: {e}")
            print(f"The server at {server_url} took too long to respond")
            print(f"This could indicate a server overload or connectivity issue")
        
        return True, False, "Inference server started - waiting for model"
    except httpx.RequestError as e:
        # Server not running or not responding
        print(f"❌ Connection error: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        
        # Check if model files exist (as a fallback)
        try:
            if os.path.exists(OUTPUT_DIR) and any(os.listdir(OUTPUT_DIR)):
                return False, False, "Inference server starting up - model files detected"
            else:
                return False, False, "Waiting for model training to complete"
        except Exception as dir_err:
            print(f"❌ Error checking model directory: {dir_err}")
            return False, False, "Waiting for model training to complete"
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, False, "Server status check failed - system initializing"
    finally:
        print("="*80)

# --- Gradio Interface ---
def generate_response(text):
    """Sends text to FastAPI inference endpoint and gets the response."""
    # Start with a basic connectivity check
    try:
        # Quick TCP check before full server status check
        import socket
        
        # Parse URL with special handling for localhost
        url_parts = FASTAPI_INFERENCE_URL.split('://')
        if len(url_parts) > 1:
            host_part = url_parts[1].split(':')[0]
        else:
            host_part = "localhost"  # Default if URL format is unexpected
            
        # Special handling for localhost/127.0.0.1
        if host_part == "localhost" or host_part == "127.0.0.1":
            # Try multiple localhost variants since Windows/WSL can be picky
            hosts_to_try = ["localhost", "127.0.0.1", "::1"]
        else:
            hosts_to_try = [host_part]
            
        # Get port with fallback
        try:
            port_part = FASTAPI_INFERENCE_URL.split(':')[-1]
            if '/' in port_part:
                port_part = port_part.split('/')[0]
            server_port = int(port_part)
        except Exception as e:
            print(f"Warning: Could not parse server port from URL: {e}")
            server_port = 8001  # Default port if parsing fails
        
        # Try multiple connection attempts
        connection_successful = False
        error_results = []
        
        print(f"Checking connectivity to FastAPI server on port {server_port}...")
        for host in hosts_to_try:
            try:
                print(f"  - Trying host: {host}...")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)  # Longer timeout for reliability
                result = s.connect_ex((host, server_port))
                s.close()
                
                if result == 0:
                    print(f"✅ Connected successfully to {host}:{server_port}")
                    connection_successful = True
                    break
                else:
                    print(f"❌ Could not connect to {host}:{server_port} (error: {result})")
                    error_results.append((host, result))
            except Exception as conn_err:
                print(f"❌ Error connecting to {host}:{server_port}: {conn_err}")
                error_results.append((host, str(conn_err)))
        
        # Use the best connection result
        if connection_successful:
            connection_result = 0
        elif error_results:
            # Use the first error code as representative
            connection_result = error_results[0][1] if isinstance(error_results[0][1], int) else -1
        
        if connection_result != 0:
            # Prepare a detailed error message with all connection attempts
            attempts_info = ""
            for host, result in error_results:
                attempts_info += f"  - {host}:{server_port} - {'Error code: ' + str(result) if isinstance(result, int) else str(result)}\n"
                
            error_message = (
                f"⚠️ Cannot connect to FastAPI inference server\n\n"
                f"The inference server appears to be offline or not listening on port {server_port}.\n\n"
                f"Connection attempts:\n{attempts_info}\n"
                f"Common causes:\n"
                f"1. The FastAPI server hasn't been started\n"
                f"2. The server is running on a different port\n"
                f"3. A firewall is blocking the connection\n\n"
                f"Debug info:\n"
                f"- URL being used: {FASTAPI_INFERENCE_URL}\n"
                f"- Model path: {OUTPUT_DIR}\n\n"
                f"You can start the FastAPI server manually with:\n"
                f"python -m uvicorn src.lora_training_pipeline.inference.fastapi_inference:app --reload --port {server_port}"
            )
            return error_message
    except Exception as socket_err:
        print(f"Error in socket check: {socket_err}")
        # Continue with regular status check even if socket check fails
    
    # Full server status check
    server_running, model_loaded, status_message = check_server_status()
    
    if not server_running:
        # If server isn't running, show error
        error_message = (
            f"⚠️ {status_message}\n\n"
            f"The inference server is not running.\n\n"
            f"Technical details:\n"
            f"- Server running: {server_running}\n"
            f"- Server URL: {FASTAPI_INFERENCE_URL}\n\n"
            f"Try restarting the pipeline or manually starting the server with:\n"
            f"python -m uvicorn src.lora_training_pipeline.inference.fastapi_inference:app --reload --port {server_port}"
        )
        return error_message
    elif not model_loaded:
        # If server is running but model isn't loaded, show friendly waiting message
        waiting_message = (
            f"✅ Server Status: {status_message}\n\n"
            f"The FastAPI server is running, but no model is loaded yet.\n\n"
            f"This is normal when:\n"
            f"1. The pipeline has just started\n"
            f"2. No training has been completed yet\n"
            f"3. There isn't enough data for training\n\n"
            f"Next steps:\n"
            f"- Continue submitting data through the data collection interface\n"
            f"- Once enough data is collected, training will start automatically\n"
            f"- After training completes, the model will be available for inference here\n\n"
            f"The inference UI is working correctly!"
        )
        return waiting_message
    
    try:
        # Use server_url from status check if we have a working connection
        request_url = server_url if 'server_url' in locals() else FASTAPI_INFERENCE_URL
        
        print(f"\nSending inference request to: {request_url}/generate/")
        start_time = time.time()
        response = httpx.post(
            f"{request_url}/generate/", 
            json={"prompt": text}, 
            timeout=30
        )
        end_time = time.time()
        print(f"Server response status: {response.status_code} (took {end_time - start_time:.2f}s)")
        print(f"Response headers: {response.headers}")
        
        response.raise_for_status()
        result = response.json()["generated_text"]
        print(f"Generated text: {result[:50]}...")
        return result
    except httpx.ConnectError as e:
        # Most common error - server not running or wrong address
        error_message = (
            f"⚠️ CONNECTION REFUSED: Could not connect to FastAPI server\n\n"
            f"The server at {FASTAPI_INFERENCE_URL} is not accepting connections.\n\n"
            f"This usually means:\n"
            f"1. The FastAPI server process has stopped or crashed\n"
            f"2. The server is running on a different port or address\n\n"
            f"Technical details:\n"
            f"- Error: {str(e)}\n"
            f"- Server URL: {FASTAPI_INFERENCE_URL}\n\n"
            f"You can manually restart the server with:\n"
            f"python -m uvicorn src.lora_training_pipeline.inference.fastapi_inference:app --reload --port {server_port}"
        )
        return error_message
    except httpx.TimeoutException as e:
        # Server taking too long to respond
        error_message = (
            f"⚠️ TIMEOUT: Server took too long to respond\n\n"
            f"The request to {FASTAPI_INFERENCE_URL}/generate/ timed out after 30 seconds.\n\n"
            f"This could be caused by:\n"
            f"1. The server is overloaded or processing a large request\n"
            f"2. The model generation is taking too long\n"
            f"3. Network connectivity issues\n\n"
            f"Technical details:\n"
            f"- Error: {str(e)}\n"
            f"- Server URL: {FASTAPI_INFERENCE_URL}\n"
            f"- Timeout: 30 seconds"
        )
        return error_message
    except httpx.RequestError as e:
        # Generic request error
        error_message = (
            f"⚠️ CONNECTION ERROR: Problem communicating with server\n\n"
            f"Unable to connect to the FastAPI server at {FASTAPI_INFERENCE_URL}.\n\n"
            f"Technical details:\n"
            f"- Error type: {type(e).__name__}\n"
            f"- Error message: {str(e)}\n"
            f"- Server URL: {FASTAPI_INFERENCE_URL}"
        )
        return error_message
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            # User-friendly message with training status when model isn't loaded
            training_status = get_training_status()
            return (
                "⚠️ The model is not loaded yet. Please wait for training to complete.\n\n"
                "What's happening?\n"
                "- The FastAPI server is running, but no model has been loaded yet\n"
                "- This is normal when training hasn't completed or hasn't started\n\n"
                f"Training Status:\n{training_status}\n\n"
                "Next steps:\n"
                "1. Make sure you've collected enough data through the data collection interface\n"
                "2. Run the training pipeline: python run_pipeline.py\n"
                "3. Wait for training to complete successfully\n"
                "4. Click the Refresh Status button above"
            )
        try:
            error_text = e.response.text
            # Try to parse JSON error
            try:
                import json
                error_data = json.loads(error_text)
                if isinstance(error_data, dict) and 'detail' in error_data:
                    error_text = error_data['detail']
            except Exception as json_err:
                # Log the error but continue with the raw text
                print(f"Warning: Failed to parse error JSON: {json_err}")
                # Don't modify error_text, use it as-is
                
            return f"⚠️ HTTP ERROR {e.response.status_code}:\n{error_text}"
        except Exception as response_err:
            # Log the specific error
            print(f"Error processing HTTP error response: {response_err}")
            import traceback
            print(f"Stack trace:\n{traceback.format_exc()}")
            return f"⚠️ HTTP error: {e.response.status_code} (Failed to get response details)"
    except Exception as e:
        # Unexpected error with stack trace
        import traceback
        stack_trace = traceback.format_exc()
        error_message = (
            f"⚠️ UNEXPECTED ERROR:\n{str(e)}\n\n"
            f"Error type: {type(e).__name__}\n\n"
            f"Stack trace:\n{stack_trace}"
        )
        return error_message

def clear_inputs(text_input, output_label):
    return "", ""

def get_training_status():
    """Check for training artifacts to provide status information to the user."""
    status_messages = []
    
    # Check model directory
    if os.path.exists(OUTPUT_DIR):
        try:
            files = os.listdir(OUTPUT_DIR)
            if files:
                status_messages.append(f"✅ Model directory exists with {len(files)} files")
                # Check file modification times to estimate when training completed
                try:
                    newest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)))
                    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(OUTPUT_DIR, newest_file)))
                    status_messages.append(f"📅 Most recent model file '{newest_file}' from {mod_time.strftime('%Y-%m-%d %H:%M')}")
                except Exception as e:
                    print(f"Error checking file timestamps: {e}")
            else:
                status_messages.append("⚠️ Model directory exists but is empty - training may have failed")
        except Exception as e:
            status_messages.append(f"⚠️ Error checking model directory: {str(e)}")
    else:
        status_messages.append("⚠️ Model directory does not exist yet - training hasn't completed")
    
    # Check for ZenML tracking info
    try:
        zenml_dir = os.path.join(".", ".zenml")
        if os.path.exists(zenml_dir):
            status_messages.append("ℹ️ ZenML tracking directory found - pipeline has been initialized")
    except Exception as zenml_err:
        print(f"[DEBUG] Error checking ZenML directory: {type(zenml_err).__name__}: {zenml_err}")
        pass
        
    # Check data collection directory
    try:
        data_dir = os.path.join(".", "data")
        if os.path.exists(data_dir):
            data_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
            if data_files:
                status_messages.append(f"📊 Data directory contains {len(data_files)} data files")
    except Exception as data_err:
        print(f"[DEBUG] Error checking data directory: {type(data_err).__name__}: {data_err}")
        pass
    
    return "\n".join(status_messages) or "No training status information available yet."

def refresh_status():
    """Refreshes the status of the inference server and updates the UI."""
    try:
        server_running, model_loaded, status_message = check_server_status()
        
        if not server_running:
            status = f"⏳ Inference server starting: {status_message}"
            # Always enable the UI - we'll show a helpful message about the server status
            is_enabled = True
            show_training_status = True
        elif not model_loaded:
            status = f"⏳ Server running but no model loaded yet: {status_message}"
            # Enable the UI even without a model - we'll show a friendly message
            is_enabled = True
            show_training_status = True
        else:
            status = "✅ Model loaded and ready for inference"
            is_enabled = True
            show_training_status = False
    except Exception as e:
        import traceback
        print(f"Error in refresh_status: {e}")
        print(traceback.format_exc())
        status = "⚠️ Checking server status... Please wait or refresh"
        is_enabled = True  # Always enable so user can try again
        show_training_status = True
    
    # Get training status information
    training_status = get_training_status() if show_training_status else ""
    
    return status, is_enabled, training_status, not show_training_status

# Initialize Blocks with theme if supported
if SUPPORTS_THEMES and hasattr(gr, 'themes') and hasattr(gr.themes, 'Soft'):
    inference_ui = gr.Blocks(title="LoRA Inference UI", theme=gr.themes.Soft())
else:
    inference_ui = gr.Blocks(title="LoRA Inference UI")

with inference_ui:
    # Add an informational notice for users about training
    with gr.Row():
        info_box = gr.Markdown("""
        ### 🤖 LoRA Inference UI
        This interface allows you to interact with your fine-tuned language model.
        
        **If the model isn't ready:**
        1. Make sure the training pipeline has completed successfully
        2. Check that the FastAPI server is running (`python -m uvicorn src.lora_training_pipeline.inference.fastapi_inference:app --reload --port 8001`)
        3. Use the refresh button to check for updates
        
        **Training required before inference!** Collect data and run the training pipeline first if you haven't already.
        """)
    
    with gr.Row():
        status_indicator = gr.Textbox(
            value="Checking model status...", 
            label="Model Status",
            interactive=False
        )
        refresh_button = gr.Button("🔄 Refresh Status")
        
    # Add a training status component that shows when model isn't ready
    with gr.Row(visible=True) as training_status_row:
        training_status_box = gr.Textbox(
            value="Checking training status...",
            label="Training Status Information",
            interactive=False,
            lines=4
        )
    
    with gr.Row():
        text_input = gr.Textbox(
            label="Enter your prompt:",
            placeholder="The model will be available after training completes...",
            interactive=True,
            lines=3
        )
    
    with gr.Row():
        generate_button = gr.Button("Generate", variant="primary")
        clear_button = gr.Button("Clear")
    
    output_label = gr.Textbox(label="Generated Text", lines=5)
    
    # Set up event handlers
    generate_button.click(
        generate_response,
        inputs=[text_input],
        outputs=[output_label],
    )
    
    # Add event handler for the text input to update on Enter key
    text_input.submit(
        generate_response,
        inputs=[text_input],
        outputs=[output_label],
    )
    clear_button.click(
        clear_inputs,
        inputs=[text_input, output_label],
        outputs=[text_input, output_label]
    )
    refresh_button.click(
        refresh_status,
        inputs=[],
        outputs=[status_indicator, generate_button, training_status_box, training_status_row]
    )
    
    # Check status on load
    inference_ui.load(
        refresh_status,
        inputs=[],
        outputs=[status_indicator, generate_button, training_status_box, training_status_row]
    )

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
            sig_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else f"Signal {sig}"
            print(f"SIGNAL RECEIVED: {sig_name} - Handled by custom handler")

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
    except ValueError as val_err:
        print(f"[DEBUG] Error parsing startup delay value '{startup_delay}': {type(val_err).__name__}: {val_err}")
        pass

    # Print Python environment information for debugging
    print("\n" + "="*80)
    print("PYTHON ENVIRONMENT INFORMATION")
    print("="*80)
    import sys
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Path: {sys.path}")

    # Print ALL environment variables for debugging
    print("\nALL ENVIRONMENT VARIABLES:")
    for key, value in sorted(os.environ.items()):
        print(f"- {key}: {value}")

    print(f"\nKEY VARIABLES:")
    print(f"- PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"- Working Directory: {os.getcwd()}")

    # Print available modules for debugging
    print("\nCHECKING FOR CRITICAL MODULES:")
    for module_name in ['gradio', 'httpx', 'sockets', 'uvicorn', 'fastapi']:
        try:
            __import__(module_name)
            print(f"- ✅ {module_name} is available")
        except ImportError:
            print(f"- ❌ {module_name} is NOT available")
    
    # Print network info
    print("\nNETWORK DIAGNOSTICS:")
    try:
        import socket
        hostname = socket.gethostname()
        print(f"- Hostname: {hostname}")
        try:
            local_ip = socket.gethostbyname(hostname)
            print(f"- Local IP: {local_ip}")
        except Exception as ip_err:
            print(f"- Error getting local IP: {ip_err}")
            
        # Try to connect to FastAPI server
        fastapi_port = int(os.environ.get('FASTAPI_INFERENCE_PORT', '8001'))
        print(f"- Attempting connection to FastAPI server on port {fastapi_port}")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex(('localhost', fastapi_port))
            print(f"  - Connection to localhost:{fastapi_port} result: {result} (0=success)")
            sock.close()
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex(('127.0.0.1', fastapi_port))
            print(f"  - Connection to 127.0.0.1:{fastapi_port} result: {result} (0=success)")
            sock.close()
        except Exception as sock_err:
            print(f"  - Socket connection error: {sock_err}")
            
        # Check if our own UI port is available
        ui_port = int(os.environ.get('GRADIO_INFERENCE_PORT', '7861'))
        print(f"- Checking if UI port {ui_port} is available")
        try:
            # Try to bind to the port to see if it's available
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.bind(('0.0.0.0', ui_port))
            print(f"  - ✅ Port {ui_port} is available for binding")
            test_socket.close()
        except Exception as bind_err:
            print(f"  - ❌ Port {ui_port} is NOT available: {bind_err}")
            print("  - This could prevent the UI from starting!")
            
            # Try alternate ports
            for alt_port in [ui_port+1, ui_port+2, ui_port+10, 8080, 8000]:
                try:
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.bind(('0.0.0.0', alt_port))
                    print(f"  - ✅ Alternate port {alt_port} IS available")
                    test_socket.close()
                except Exception as alt_port_err:
                    print(f"[DEBUG] Alternate port {alt_port} test error type: {type(alt_port_err).__name__}")
                    print(f"[DEBUG] Alternate port {alt_port} test error details: {alt_port_err}")
                    print(f"  - ❌ Alternate port {alt_port} is NOT available")
                    
        # List all network interfaces
        print("- Network interfaces:")
        try:
            import psutil
            for iface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:  # IPv4
                        print(f"  - {iface}: {addr.address}")
        except ImportError:
            print("  - psutil module not available for interface listing")
    except Exception as net_err:
        print(f"Error in network diagnostics: {net_err}")
    
    print("="*80 + "\n")
    
    # Use ServiceManager to ensure only one instance runs
    try:
        from src.lora_training_pipeline.utils.service_manager import ServiceManager

        port = int(os.environ.get("GRADIO_INFERENCE_PORT", 7861))
        print(f"Checking for existing Gradio Inference UI instances on port {port}...")

        service_manager = ServiceManager("gradio_inference", port)
        if not service_manager.can_start():
            print(f"ERROR: {service_manager.error_message}")
            print("Another instance of the Inference UI is already running.")
            print("To check running services and clean up stale processes:")
            print("python -m src.lora_training_pipeline.utils.service_manager --status")
            print("python -m src.lora_training_pipeline.utils.service_manager --cleanup")
            sys.exit(1)

        print(f"✅ No other Gradio Inference UI instances running on port {port}")

        # Register service manager cleanup on exit
        import atexit
        atexit.register(service_manager.release)

        # Register signal handlers for cleanup
        try:
            import signal
            def signal_handler(sig, frame):
                print(f"Received signal {sig}, cleaning up")
                service_manager.release()
                sys.exit(0)

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except Exception as e:
            print(f"Warning: Could not register signal handlers: {e}")

    except ImportError as e:
        print(f"Warning: ServiceManager not available: {e}")
        print("Using fallback process management - multiple instances may be started")

        # Basic port check as fallback
        try:
            port = int(os.environ.get("GRADIO_INFERENCE_PORT", 7861))

            # Check if port is already in use
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                result = s.connect_ex(('127.0.0.1', port))
                if result == 0:
                    print(f"❌ ERROR: Port {port} is already in use by another process")
                    print("The UI cannot start because the port is already in use.")
                    print("Try using a different port or close the existing process.")
                    sys.exit(1)
                else:
                    print(f"✅ Port {port} is available for binding")

            # Simple PID file management
            try:
                pid_file = "./inference_ui.pid"
                pid = os.getpid()

                # Write PID file
                with open(pid_file, 'w') as f:
                    import json
                    import time
                    json.dump({
                        "pid": pid,
                        "port": port,
                        "process_type": "gradio_inference",
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
        except Exception as e:
            print(f"Error checking port availability: {e}")

    # Use 0.0.0.0 to allow connections from any IP address, making the UI accessible from the internet
    # Use share=True to create a public URL through Gradio's sharing service
    # Get port from environment variable or use default for display
    port = int(os.environ.get("GRADIO_INFERENCE_PORT", 7861))
    
    print("ℹ️ LAUNCHING INFERENCE UI: Starting Gradio interface...")
    print("ℹ️ The interface will be accessible at:")
    print(f"ℹ️ - Locally: http://localhost:{port}") 
    print(f"ℹ️ - Network: http://<your-ip-address>:{port}")
    print("ℹ️ - Public: A public URL will be displayed when Gradio starts")
    print(f"ℹ️ - Process ID: {os.getpid()}")
    
    # Additional debug info
    print("\nProcess Details:")
    try:
        import psutil
        process = psutil.Process()
        print(f"- Process creation time: {datetime.fromtimestamp(process.create_time()).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"- Parent PID: {process.ppid()}")
        print(f"- Command line: {' '.join(process.cmdline())}")
        print(f"- CPU usage: {process.cpu_percent()}%")
        print(f"- Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    except Exception as proc_err:
        print(f"- Error getting process details: {proc_err}")
    
    # Custom configuration to prevent conflicts with other Gradio instances
    try:
        # Basic launch parameters supported by all Gradio versions
        launch_params = {
            "server_name": "0.0.0.0",
            "server_port": port,
            "share": True,
            "prevent_thread_lock": True,  # Allows the app to run in parallel with other Gradio apps
            "quiet": False                # Show all messages for debugging
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
            # Try without queue for maximum compatibility
            print(f"\n{'='*50}\nLAUNCHING GRADIO INFERENCE UI\n{'='*50}")
            print(f"Launch parameters: {launch_params}")
            print(f"Server name: {launch_params.get('server_name')}")
            print(f"Server port: {launch_params.get('server_port')}")
            print(f"Share: {launch_params.get('share')}")
            print(f"Starting Gradio interface without queue for maximum compatibility...")
            try:
                # Check if port is already in use before launching
                import socket
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('0.0.0.0', port))
                        s.close()
                    print(f"✅ Port {port} is available for binding")
                except OSError as e:
                    print(f"⚠️ Port {port} is already in use: {e}")
                    print(f"⚠️ This may cause the Gradio UI to fail or use a different port")
                    
                    # Try to identify process using this port
                    try:
                        from src.lora_training_pipeline.utils.fix_port_issues import is_port_in_use
                        is_port_in_use(port)  # This function logs the process using the port
                    except ImportError as import_err:
                        print(f"[DEBUG] Error importing port check utilities: {type(import_err).__name__}: {import_err}")
                        pass
                
                # Launch the UI
                inference_ui.launch(**launch_params)
            except Exception as launch_err:
                print(f"ERROR IN GRADIO LAUNCH:\n{type(launch_err).__name__}: {launch_err}")
                import traceback
                print(traceback.format_exc())
                
                # Check if this is a port conflict
                if "address already in use" in str(launch_err).lower():
                    print("This appears to be a port conflict. Another process is already using this port.")
                    print("Use the port conflict resolver to fix this issue:")
                    print("python -m src.lora_training_pipeline.utils.port_conflict_resolver --resolve")
                
                raise launch_err
        except Exception as e1:
            print(f"First launch attempt failed: {e1}")
            try:
                # Try with just max_size
                print("Trying with queue(max_size=20)...")
                inference_ui.queue(max_size=20).launch(**launch_params)
            except TypeError as e2:
                # If that also fails, use the simplest queue configuration
                print(f"Queue with max_size failed: {e2}")
                print("Falling back to basic queue configuration")
                try:
                    inference_ui.queue().launch(**launch_params)
                except Exception as e3:
                    print(f"All queue configurations failed: {e3}")
                    # Final fallback with minimal options
                    print("Using absolute minimal configuration")
                    try:
                        print("Trying 127.0.0.1 as server name...")
                        inference_ui.launch(server_name="127.0.0.1", server_port=port)
                    except Exception as e4:
                        print(f"127.0.0.1 also failed: {e4}")
                        print("Final attempt with localhost and different port...")
                        # Use a different port as last resort
                        fallback_port = port + 10  # Use port 7871 instead of 7861
                        print(f"Using fallback port {fallback_port}")
                        inference_ui.launch(server_name="localhost", server_port=fallback_port)
            
    except Exception as e:
        print(f"❌ ERROR: Failed to start Inference UI: {e}")
        print(f"ℹ️ If port {port} is in use, you can modify the port number using the GRADIO_INFERENCE_PORT environment variable.")
        print(f"ℹ️ Use the port conflict resolver to fix port conflicts:")
        print(f"ℹ️ python -m src.lora_training_pipeline.utils.port_conflict_resolver --resolve")