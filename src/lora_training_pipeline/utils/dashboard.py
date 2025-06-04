# LoRA_Training_Pipeline/src/utils/dashboard.py
from flask import Flask, jsonify, render_template, send_from_directory
import os
import sys
import time
import json
import threading
import pandas as pd
from pathlib import Path
import psutil
import socket
import subprocess
import torch
from datetime import datetime

# Add project root to path if needed
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import project modules
from src.lora_training_pipeline.config import BASE_MODEL_NAME, OLLAMA_MODEL_NAME, DATASET_NAME, DATA_VERSION
from src.lora_training_pipeline.utils.helpers import check_zenml_connection, get_pending_errors
from src.lora_training_pipeline.training.lora_finetune import llama_lora_config

# Constants
DASHBOARD_PORT = 7863  # Avoid conflicts with existing services
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
TRAINING_CYCLE_FILE = Path("./training_cycle_info.json")
BEST_MODEL_METRICS_FILE = Path("./best_model_metrics.json")
TRAINING_LOCK_FILE = Path("./.training_lock")
UPDATE_INTERVAL = 5  # seconds

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Create a custom JSON encoder to handle sets and Path objects
def custom_json_encoder(obj):
    """Custom JSON encoder for dashboard data."""
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    # Add more custom encoding as needed
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Global cache for dashboard data with default values
dashboard_data = {
    "last_update": time.time(),
    "system_info": {
        "cpu_usage": 0,
        "memory_usage": 0,
        "disk_usage": 0,
        "hostname": "unknown",
        "platform": "unknown",
        "python_version": "unknown",
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_available": False
    },
    "model_info": {
        "base_model_name": BASE_MODEL_NAME,
        "ollama_model_name": OLLAMA_MODEL_NAME,
        "lora_config": {
            "r": llama_lora_config.r,
            "lora_alpha": llama_lora_config.lora_alpha,
            "target_modules": llama_lora_config.target_modules,
            "lora_dropout": llama_lora_config.lora_dropout,
            "task_type": llama_lora_config.task_type,
        },
        "best_model": None
    },
    "training_info": {
        "training_lock": False,
        "cycle_count": 0,
        "current_threshold": 10,
        "total_valid_data_points": 0,
        "last_training_time": 0,
        "last_valid_count": 0,
        "time_since_last_training": 0,
        "time_since_formatted": "Never trained",
        "last_training_formatted": "Never",
        "zenml_connected": False
    },
    "data_info": {
        "dataset_name": DATASET_NAME,
        "data_version": DATA_VERSION,
        "collected_files_count": 0,
        "total_collected": 0,
        "cleaned_files_count": 0,
        "total_valid": 0,
        "rejected_files_count": 0,
        "total_rejected": 0,
        "validation_rate": 0
    },
    "processes_info": {
        "processes": []
    },
    "errors_info": {
        "pending_errors": ["No pending errors found."]
    }
}

def get_system_info():
    """Gets system information like CPU, memory, GPU usage."""
    info = {
        "cpu_usage": psutil.cpu_percent(interval=0.1),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "hostname": socket.gethostname(),
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Get GPU info if available
    try:
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            # Try to get GPU memory usage
            try:
                import subprocess
                result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'])
                memory_used, memory_total = map(int, result.decode('utf-8').split(','))
                info["gpu_memory_used"] = memory_used
                info["gpu_memory_total"] = memory_total
                info["gpu_memory_percent"] = round((memory_used / memory_total) * 100, 2)
            except Exception as nvidia_err:
                print(f"[DEBUG] nvidia-smi error type: {type(nvidia_err).__name__}")
                print(f"[DEBUG] nvidia-smi error details: {nvidia_err}")
                print(f"Warning: nvidia-smi check failed: {nvidia_err}")
                # If nvidia-smi fails, use torch
                info["gpu_memory_allocated"] = round(torch.cuda.memory_allocated(0)/1024**3, 2)
                info["gpu_memory_reserved"] = round(torch.cuda.memory_reserved(0)/1024**3, 2)
        else:
            info["gpu_available"] = False
    except Exception as gpu_err:
        print(f"Warning: GPU availability check failed: {gpu_err}")
        info["gpu_available"] = False
    
    return info

def get_model_info():
    """Gets information about the models being used."""
    info = {
        "base_model_name": BASE_MODEL_NAME,
        "ollama_model_name": OLLAMA_MODEL_NAME,
        "lora_config": {
            "r": llama_lora_config.r,
            "lora_alpha": llama_lora_config.lora_alpha,
            "target_modules": llama_lora_config.target_modules,
            "lora_dropout": llama_lora_config.lora_dropout,
            "task_type": llama_lora_config.task_type,
        }
    }
    
    # Check if best model exists and get its metrics
    if BEST_MODEL_METRICS_FILE.exists():
        try:
            with open(BEST_MODEL_METRICS_FILE, "r") as f:
                best_model_data = json.load(f)
                info["best_model"] = {
                    "val_loss": best_model_data.get("best_val_loss", "N/A"),
                    "path": best_model_data.get("model_path", "N/A"),
                    "last_update": os.path.getmtime(BEST_MODEL_METRICS_FILE),
                    "formatted_time": datetime.fromtimestamp(os.path.getmtime(BEST_MODEL_METRICS_FILE)).strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            info["best_model_error"] = str(e)
    else:
        info["best_model"] = None
    
    return info

def get_training_info():
    """Gets information about training cycles and thresholds."""
    info = {
        "training_lock": TRAINING_LOCK_FILE.exists(),
    }
    
    # Get training cycle info
    if TRAINING_CYCLE_FILE.exists():
        try:
            with open(TRAINING_CYCLE_FILE, "r") as f:
                cycle_info = json.load(f)
                info["cycle_count"] = cycle_info.get("cycle_count", 0)
                info["total_valid_data_points"] = cycle_info.get("total_valid_data_points", 0)
                info["last_training_attempt_time"] = cycle_info.get("last_training_attempt_time", 
                                                                    cycle_info.get("last_training_time", 0))
                info["last_successful_training_time"] = cycle_info.get("last_successful_training_time", 0)
                info["training_success"] = cycle_info.get("training_success", False)
                info["last_valid_count"] = cycle_info.get("last_valid_count", 0)
                
                # Calculate next threshold (base threshold is 10)
                base_threshold = 10
                info["current_threshold"] = (info["cycle_count"] + 1) * base_threshold
                
                # Time since last training attempt
                if info["last_training_attempt_time"] > 0:
                    time_since_attempt = time.time() - info["last_training_attempt_time"]
                    info["time_since_last_attempt"] = time_since_attempt
                    # Format nicely
                    if time_since_attempt < 60:
                        info["time_since_attempt_formatted"] = f"{time_since_attempt:.1f} seconds"
                    elif time_since_attempt < 3600:
                        info["time_since_attempt_formatted"] = f"{time_since_attempt/60:.1f} minutes"
                    else:
                        info["time_since_attempt_formatted"] = f"{time_since_attempt/3600:.1f} hours"
                else:
                    info["time_since_last_attempt"] = 0
                    info["time_since_attempt_formatted"] = "Never attempted"
                
                # Time since last successful training
                if info["last_successful_training_time"] > 0:
                    time_since_success = time.time() - info["last_successful_training_time"]
                    info["time_since_last_success"] = time_since_success
                    # Format nicely
                    if time_since_success < 60:
                        info["time_since_success_formatted"] = f"{time_since_success:.1f} seconds"
                    elif time_since_success < 3600:
                        info["time_since_success_formatted"] = f"{time_since_success/60:.1f} minutes"
                    else:
                        info["time_since_success_formatted"] = f"{time_since_success/3600:.1f} hours"
                else:
                    info["time_since_last_success"] = 0
                    info["time_since_success_formatted"] = "Never successfully trained"
                
                # Format last training attempt time
                if info["last_training_attempt_time"] > 0:
                    info["last_attempt_formatted"] = datetime.fromtimestamp(info["last_training_attempt_time"]).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    info["last_attempt_formatted"] = "Never"
                    
                # Format last successful training time
                if info["last_successful_training_time"] > 0:
                    info["last_success_formatted"] = datetime.fromtimestamp(info["last_successful_training_time"]).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    info["last_success_formatted"] = "Never"
                    
                # For backward compatibility with the UI
                info["last_training_time"] = info["last_training_attempt_time"]
                info["last_training_formatted"] = info["last_attempt_formatted"]
                info["time_since_last_training"] = info["time_since_last_attempt"]
                info["time_since_formatted"] = info["time_since_attempt_formatted"]
        except Exception as e:
            info["training_cycle_error"] = str(e)
    else:
        info["cycle_count"] = 0
        info["current_threshold"] = 10
        info["total_valid_data_points"] = 0
        info["last_training_attempt_time"] = 0
        info["last_successful_training_time"] = 0
        info["last_valid_count"] = 0
        info["time_since_last_attempt"] = 0
        info["time_since_attempt_formatted"] = "Never attempted"
        info["time_since_last_success"] = 0
        info["time_since_success_formatted"] = "Never successfully trained"
        info["last_attempt_formatted"] = "Never"
        info["last_success_formatted"] = "Never"
        info["training_success"] = False
        
        # For backward compatibility with the UI
        info["last_training_time"] = 0
        info["last_training_formatted"] = "Never"
        info["time_since_last_training"] = 0
        info["time_since_formatted"] = "Never trained"
    
    # Check ZenML connection
    info["zenml_connected"] = check_zenml_connection(max_retries=1, silent=True)
    
    return info

def get_data_info(print_messages=False):
    """Gets information about collected and processed data."""
    info = {
        "dataset_name": DATASET_NAME,
        "data_version": DATA_VERSION,
    }
    
    # Count all collected data
    if print_messages:
        print(f"Counting data in directory: {DATA_DIR}")
        print(f"📊 Searching for pattern: {DATASET_NAME}_{DATA_VERSION}_original_*.parquet")
    
    all_files = list(DATA_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_original_*.parquet"))
    info["collected_files_count"] = len(all_files)
    
    # Read and count all collected data points
    total_collected = 0
    if all_files:
        try:
            df_list = []
            for file in all_files:
                try:
                    df = pd.read_parquet(file)
                    df_list.append(df)
                except Exception as e:
                    if print_messages:
                        print(f"⚠️ Warning: Could not read file {file}: {e}")
                    continue
            
            if df_list:
                collected_df = pd.concat(df_list)
                total_collected = len(collected_df)
        except Exception as e:
            info["collected_data_error"] = str(e)
            if print_messages:
                print(f"⚠️ Error counting collected data: {e}")
    
    info["total_collected"] = total_collected
    
    # Count valid data
    if print_messages:
        print(f"📊 Searching for pattern: {DATASET_NAME}_{DATA_VERSION}_valid_*.parquet")
    
    # Look in both the main directory and the valid subdirectory
    valid_files_main = list(DATA_DIR.glob(f"{DATASET_NAME}_{DATA_VERSION}_valid_*.parquet"))
    valid_files_subdir = list(Path(DATA_DIR / "valid").glob(f"{DATASET_NAME}_{DATA_VERSION}_valid_*.parquet"))
    valid_files = valid_files_main + valid_files_subdir
    
    info["valid_files_count"] = len(valid_files)
    
    # Read and count all valid data points
    total_valid = 0
    if valid_files:
        try:
            valid_df_list = []
            for file in valid_files:
                try:
                    df = pd.read_parquet(file)
                    valid_df_list.append(df)
                except Exception as e:
                    if print_messages:
                        print(f"⚠️ Warning: Could not read valid file {file}: {e}")
                    continue
            
            if valid_df_list:
                valid_df = pd.concat(valid_df_list)
                total_valid = len(valid_df)
        except Exception as e:
            info["valid_data_error"] = str(e)
            if print_messages:
                print(f"⚠️ Error counting valid data: {e}")
    
    info["total_valid"] = total_valid
    
    # Count invalid data
    if print_messages:
        print(f"📊 Searching for pattern: {DATASET_NAME}_{DATA_VERSION}_invalid_*.parquet")
    invalid_files = list(Path(DATA_DIR / "rejected").glob(f"{DATASET_NAME}_{DATA_VERSION}_invalid_*.parquet"))
    info["invalid_files_count"] = len(invalid_files)
    
    # Read and count all invalid data points
    total_invalid = 0
    if invalid_files:
        try:
            invalid_df_list = []
            for file in invalid_files:
                try:
                    df = pd.read_parquet(file)
                    invalid_df_list.append(df)
                except Exception as e:
                    if print_messages:
                        print(f"⚠️ Warning: Could not read invalid file {file}: {e}")
                    continue
            
            if invalid_df_list:
                invalid_df = pd.concat(invalid_df_list)
                total_invalid = len(invalid_df)
        except Exception as e:
            info["invalid_data_error"] = str(e)
            if print_messages:
                print(f"⚠️ Error counting invalid data: {e}")
    
    info["total_invalid"] = total_invalid
    
    # Calculate validation rate
    if total_collected > 0:
        info["validation_rate"] = round((total_valid / total_collected) * 100, 2)
    else:
        info["validation_rate"] = 0
        
    # Also count training files
    training_dir = DATA_DIR / "training"
    training_files = []
    if training_dir.exists():
        training_files = list(training_dir.glob(f"{DATASET_NAME}_{DATA_VERSION}_training_*.parquet"))
    
    info["training_files_count"] = len(training_files)
    
    if print_messages:
        print(f"\n📊 DATA SUMMARY:")
        print(f"  - Total collected: {total_collected} data points in {len(all_files)} files")
        print(f"  - Valid data: {total_valid} data points in {len(valid_files)} files")
        print(f"  - Rejected data: {total_invalid} data points in {len(invalid_files)} files")
        print(f"  - Training files: {len(training_files)} files")
    
    return info

def get_processes_info():
    """Gets information about running processes related to the pipeline."""
    info = {
        "processes": []
    }
    
    # Define process name patterns to look for
    process_patterns = {
        "GradioDataCollection": ["gradio_app.py", "GradioDataCollection"],
        "ZenMLTrainingPipeline": ["zenml_pipeline.py", "ZenMLTrainingPipeline"],
        "GradioInferenceUI": ["gradio_inference.py", "GradioInferenceUI"],
        "FastAPIInferenceServer": ["fastapi_inference", "uvicorn"],
        "Dashboard": ["dashboard.py"]
    }
    
    # Get all running Python processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'status']):
        try:
            # Skip if not a Python process
            try:
                proc_cmdline = proc.info["cmdline"] if proc.info["cmdline"] else []
                # Make sure cmdline is a list before joining
                if not isinstance(proc_cmdline, list):
                    proc_cmdline = []
                    
                cmdline_str = " ".join(str(item) for item in proc_cmdline)
                
                if not ("python" in proc.info["name"].lower() or "python" in cmdline_str.lower()):
                    continue
                    
                # Check if this is one of our processes
                process_info = {
                    "pid": proc.info["pid"],
                    "cmdline": cmdline_str,
                    "status": proc.info["status"],
                    "running_time": time.time() - proc.info["create_time"],
                    "memory_mb": round(proc.memory_info().rss / (1024 * 1024), 2)
                }
            except (TypeError, AttributeError) as e:
                # Skip this process if there's an error processing its info
                print(f"Warning: Error processing process {proc.info['pid'] if 'pid' in proc.info else 'unknown'}: {e}")
                continue
            
            # Format running time
            if process_info["running_time"] < 60:
                process_info["running_time_formatted"] = f"{process_info['running_time']:.1f} seconds"
            elif process_info["running_time"] < 3600:
                process_info["running_time_formatted"] = f"{process_info['running_time']/60:.1f} minutes"
            else:
                process_info["running_time_formatted"] = f"{process_info['running_time']/3600:.1f} hours"
            
            # Identify process type
            process_type = "Other"
            for name, patterns in process_patterns.items():
                try:
                    if any(pattern in process_info["cmdline"] for pattern in patterns):
                        process_type = name
                        break
                except (TypeError, AttributeError):
                    # Skip pattern checking if cmdline is not properly formatted
                    continue
            
            process_info["type"] = process_type
            
            # Only add if it's one of our processes
            if process_type != "Other":
                info["processes"].append(process_info)
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    # Sort by process type
    info["processes"].sort(key=lambda x: x["type"])
    
    # Check for expected processes that aren't running
    running_types = {p["type"] for p in info["processes"]}
    expected_types = set(process_patterns.keys())
    # Convert sets to lists for JSON serialization
    info["missing_processes"] = list(expected_types - running_types - {"Dashboard"})
    # Make sure there are no sets in the info dictionary
    info["running_processes"] = list(running_types)
    info["expected_processes"] = list(expected_types)
    
    return info

def get_errors_info():
    """Gets information about pending errors."""
    info = {
        "pending_errors": get_pending_errors(max_errors=50)
    }
    
    return info

# Note: We've replaced this function with individual section updates in background_updater
    
def background_updater():
    """Background thread to update dashboard data periodically."""
    while True:
        try:
            # Update each section individually to isolate errors
            try:
                dashboard_data["last_update"] = time.time()
                dashboard_data["system_info"] = get_system_info()
            except Exception as e:
                print(f"Error updating system info: {e}")
                
            try:
                dashboard_data["model_info"] = get_model_info()
            except Exception as e:
                print(f"Error updating model info: {e}")
                
            try:
                dashboard_data["training_info"] = get_training_info()
            except Exception as e:
                print(f"Error updating training info: {e}")
                
            try:
                dashboard_data["data_info"] = get_data_info(print_messages=False)
            except Exception as e:
                print(f"Error updating data info: {e}")
                
            try:
                dashboard_data["processes_info"] = get_processes_info()
            except Exception as e:
                print(f"Error updating processes info: {e}")
                
            try:
                dashboard_data["errors_info"] = get_errors_info()
            except Exception as e:
                print(f"Error updating errors info: {e}")
                
        except Exception as e:
            print(f"Error in background updater: {e}")
            # Ensure we have at least minimal data
            if "last_update" not in dashboard_data:
                dashboard_data["last_update"] = time.time()
                
        # Sleep before next update
        time.sleep(UPDATE_INTERVAL)

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/dashboard-data')
def get_dashboard_data():
    """API endpoint to get current dashboard data."""
    # Make a copy of the dashboard data to avoid modifying the original
    dashboard_copy = json.loads(json.dumps(dashboard_data, default=custom_json_encoder))
    return jsonify(dashboard_copy)

@app.route('/api/delete-datasets', methods=['POST'])
def delete_datasets():
    """API endpoint to delete all datasets."""
    try:
        # Import and execute the delete function
        import sys
        from pathlib import Path
        
        # Get the root directory of the project
        root_dir = Path(__file__).resolve().parents[3]
        
        # Make sure run_pipeline.py can be imported properly
        sys.path.insert(0, str(root_dir))
        
        # Import our delete function
        from run_pipeline import delete_all_datasets
        
        # Call the delete function
        result = delete_all_datasets()
        
        # Force update of dashboard data
        dashboard_data["data_info"] = get_data_info(print_messages=False)
        
        # Convert any set objects in the result to lists
        serializable_result = json.loads(json.dumps(result, default=custom_json_encoder))
        
        return jsonify({
            "success": True,
            "message": "Datasets deleted successfully",
            "details": serializable_result
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        return jsonify({
            "success": False,
            "message": f"Error deleting datasets: {error_msg}",
            "error": error_trace
        }), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serves static files."""
    return send_from_directory(app.static_folder, filename)

def create_templates_directory():
    """Creates the templates directory if it doesn't exist."""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    return templates_dir

def create_static_directory():
    """Creates the static directory if it doesn't exist."""
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    return static_dir

def run_dashboard(host='0.0.0.0', port=DASHBOARD_PORT, debug=False):
    """Runs the dashboard server."""
    print(f"Starting LoRA Training Pipeline Dashboard on http://{host}:{port}")
    print(f"You can access the dashboard from your browser at http://localhost:{port}")
    
    # Create directories if they don't exist
    templates_dir = create_templates_directory()
    static_dir = create_static_directory()
    
    # Initial data fetch with verbose output
    print("\n" + "="*50)
    print("📊 INITIALIZING DASHBOARD DATA")
    print("="*50)
    
    # Print data info once at startup only
    dashboard_data["data_info"] = get_data_info(print_messages=True)
    
    # Start the background updater thread (silent updates)
    update_thread = threading.Thread(target=background_updater, daemon=True)
    update_thread.start()
    
    # Run the Flask app
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_dashboard()