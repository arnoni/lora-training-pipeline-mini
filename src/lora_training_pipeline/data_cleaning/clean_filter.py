#!/usr/bin/env python
# LoRA_Training_Pipeline/src/data_cleaning/clean_filter.py

# Import error handling tools first to avoid NameError
from src.lora_training_pipeline.utils.helpers import safe_run, log_pending_error

import pandas as pd
import datetime
import json
import os
from pathlib import Path
import shutil
import time
from zenml import step  # Import step directly from zenml
from typing import Tuple, Dict, Any, List, Optional, Union, Annotated  # Use Annotated for named outputs
# Don't import ollama here - we'll import it dynamically in the validation function
# to handle different versions of the library
# safe_run and log_pending_error already imported at the top of the file

# Import configuration 
from src.lora_training_pipeline.config import OLLAMA_MODEL_NAME, OLLAMA_FALLBACK_MODELS

# --- Configuration ---
DATA_DIR = Path("./data")
REJECTED_DIR = Path("./data/rejected")
VALID_DIR = Path("./data/valid")
CHECKPOINT_FILE = Path("./data/validation_checkpoint.json")

# Create required directories if they don't exist
REJECTED_DIR.mkdir(parents=True, exist_ok=True)
VALID_DIR.mkdir(parents=True, exist_ok=True)

def validate_text_with_ollama(text: str) -> int:
    """Sends text to Ollama for validation using the original ollama library."""
    # Debug timestamp to track when validation starts
    import datetime
    validation_start_time = datetime.datetime.now()
    print(f"\n[DEBUG-OLLAMA] Starting validation at: {validation_start_time}")
    print(f"[DEBUG-OLLAMA] Text length: {len(text)} characters")
    
    # Debug: Print the first 50 chars of text
    preview = text[:50] + "..." if len(text) > 50 else text
    print(f"[DEBUG-OLLAMA] Text preview: {preview}")
    
    # Check if Ollama server is running first
    import socket
    try:
        print("[DEBUG-OLLAMA] Checking if Ollama server is running...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        result = s.connect_ex(('localhost', 11434))
        if result == 0:
            print("[DEBUG-OLLAMA] ✅ Ollama server is running on port 11434")
            
            # Try to check Ollama version
            try:
                print("[DEBUG-OLLAMA] Checking Ollama version...")
                import urllib.request
                import json
                import urllib.error
                try:
                    response = urllib.request.urlopen('http://localhost:11434/api/version', timeout=2)
                    version_data = json.loads(response.read().decode())
                    print(f"[DEBUG-OLLAMA] ✅ Ollama version: {version_data}")
                except urllib.error.URLError as url_err:
                    print(f"[DEBUG-OLLAMA] ⚠️ Could not get version via API: {url_err}")
                except Exception as api_err:
                    print(f"[DEBUG-OLLAMA] ⚠️ Error reading version data: {api_err}")
            except Exception as ver_err:
                print(f"[DEBUG-OLLAMA] ⚠️ Error checking Ollama version: {ver_err}")
        else:
            print("[DEBUG-OLLAMA] ❌ Ollama server not detected on port 11434")
        s.close()
    except Exception as conn_err:
        print(f"[DEBUG-OLLAMA] ⚠️ Could not check Ollama server connection: {conn_err}")
    
    prompt = f"""Analyze the given text: {text} and determine if it represents
a valid online chat between two people. Output 1 if the text is a coherent and
appropriate conversation. Output 0 if the text contains any invalid content,
such as hate speech, gibberish, trolling, abuse, or other inappropriate or
incoherent elements. A valid online chat can contain slang or street language,
nicknames, abbreviations, acronyms, and initialisms. A valid online chat
can also contain outrageous or provoking content.

You must respond with a single digit only: either 1 or 0.
"""
    print(f"[DEBUG-OLLAMA] Prompt prepared, length: {len(prompt)} characters")

    try:
        # Import ollama library with detailed error checking
        try:
            print("[DEBUG-OLLAMA] Attempting to import ollama library...")
            import sys
            print(f"[DEBUG-OLLAMA] Python executable: {sys.executable}")
            print(f"[DEBUG-OLLAMA] Python path: {sys.path}")
            
            import ollama
            
            # Get ollama module info
            module_path = getattr(ollama, "__file__", "Unknown")
            module_version = getattr(ollama, "__version__", "Unknown")
            print(f"[DEBUG-OLLAMA] ✅ Ollama library imported successfully")
            print(f"[DEBUG-OLLAMA] Ollama library path: {module_path}")
            print(f"[DEBUG-OLLAMA] Ollama library version: {module_version}")
            print(f"[DEBUG-OLLAMA] Ollama module attributes: {dir(ollama)[:10]}...")
        except ImportError as e:
            print(f"[DEBUG-OLLAMA] ❌ Ollama import error: {e}")
            print(f"[DEBUG-OLLAMA] Error type: {type(e).__name__}")
            print(f"[DEBUG-OLLAMA] Error details: {str(e)}")
            print("[DEBUG-OLLAMA] Falling back to simulation mode.")
            raise ImportError(f"Ollama library not installed: {e}")

        # Connect to Ollama server
        print(f"⏳ Connecting to Ollama server...")
        
        # Check if models are available with detailed debugging
        try:
            print("[DEBUG-OLLAMA] Checking if models are available...")
            print("[DEBUG-OLLAMA] Calling ollama.list() function...")
            
            # Try to print ollama.list function info before calling
            list_func = getattr(ollama, "list", None)
            if list_func:
                print(f"[DEBUG-OLLAMA] ollama.list function exists: {list_func}")
            else:
                print("[DEBUG-OLLAMA] ⚠️ ollama.list function not found!")
                # Check what functions are available
                print(f"[DEBUG-OLLAMA] Available ollama functions: {[f for f in dir(ollama) if not f.startswith('_')]}")
            
            # Attempt to call the list function with exception tracking
            try:
                print("[DEBUG-OLLAMA] Executing ollama.list()...")
                models_response = ollama.list()
                print(f"[DEBUG-OLLAMA] ollama.list() returned: {models_response[:100]}..." if isinstance(models_response, str) else f"[DEBUG-OLLAMA] ollama.list() returned non-string: {type(models_response)}")
            except Exception as list_err:
                print(f"[DEBUG-OLLAMA] ❌ Error calling ollama.list(): {list_err}")
                print(f"[DEBUG-OLLAMA] Error type: {type(list_err).__name__}")
                raise
            
            # Parse the models response
            import json
            try:
                print("[DEBUG-OLLAMA] Parsing models response...")
                if isinstance(models_response, str):
                    models_data = json.loads(models_response)
                else:
                    print(f"[DEBUG-OLLAMA] models_response is not a string, it's a {type(models_response)}")
                    # If it's already a dict, use it directly
                    if isinstance(models_response, dict):
                        models_data = models_response
                    else:
                        print("[DEBUG-OLLAMA] Cannot parse non-string, non-dict response")
                        raise ValueError(f"Unexpected response type: {type(models_response)}")
                
                print(f"[DEBUG-OLLAMA] Models data type: {type(models_data)}")
                
                available_models = []
                
                if isinstance(models_data, dict) and 'models' in models_data:
                    print(f"[DEBUG-OLLAMA] Found 'models' key with {len(models_data['models'])} models")
                    available_models = [m.get('name', '') for m in models_data['models'] if isinstance(m, dict)]
                else:
                    # Try to extract models from other formats
                    print(f"[DEBUG-OLLAMA] Models data keys: {models_data.keys() if hasattr(models_data, 'keys') else 'No keys'}")
                
                print(f"[DEBUG-OLLAMA] Found {len(available_models)} available models: {available_models}")
                
                print(f"[DEBUG-OLLAMA] Checking for exact match of {OLLAMA_MODEL_NAME} in available models")
                # Try to use the configured model first, then fall back to alternatives
                # First check if our main model is available
                if OLLAMA_MODEL_NAME in available_models:
                    model_name = OLLAMA_MODEL_NAME
                    print(f"✅ Using configured model: {model_name}")
                else:
                    print(f"[DEBUG-OLLAMA] Primary model {OLLAMA_MODEL_NAME} not found, trying fallbacks")
                    # Try each fallback model in order of preference
                    found_model = False
                    for fallback in OLLAMA_FALLBACK_MODELS:
                        print(f"[DEBUG-OLLAMA] Checking for fallback model containing '{fallback}'")
                        matching_models = [m for m in available_models if fallback.lower() in m.lower()]
                        if matching_models:
                            model_name = matching_models[0]
                            print(f"⚠️ Using fallback model: {model_name}")
                            found_model = True
                            break
                            
                    # If we still don't have a model, try common prefixes
                    if not found_model:
                        print(f"[DEBUG-OLLAMA] No fallbacks found, trying common prefixes")
                        for prefix in ['gemma', 'deepseek', 'phi', 'qwen', 'mistral']:
                            print(f"[DEBUG-OLLAMA] Looking for model with prefix '{prefix}'")
                            matches = [m for m in available_models if prefix in m.lower()]
                            if matches:
                                model_name = matches[0]
                                print(f"⚠️ Using available {prefix.title()} model: {model_name}")
                                found_model = True
                                break
                                
                    # Just use any available model as last resort
                    if not found_model and available_models:
                        model_name = available_models[0]
                        print(f"⚠️ Using first available model: {model_name}")
                    elif not found_model:
                        # No models available, try model from your list
                        model_name = 'gemma3:4b'  # Try gemma3:4b which you have locally
                        print(f"⚠️ No models found, trying default: {model_name}")
            except json.JSONDecodeError as json_err:
                print(f"[DEBUG] JSON decode error type: {type(json_err).__name__}")
                print(f"[DEBUG] JSON decode error details: {json_err}")
                print("⚠️ Could not parse models response, using default model")
                model_name = 'gemma3:4b'  # Use gemma3:4b which is available on your system
        except Exception as e:
            print(f"⚠️ Error checking models: {e}")
            model_name = 'gemma3:4b'  # Use gemma3:4b which is available on your system
            
        # Generate response using the original ollama library with detailed debugging
        print(f"🤖 Using Ollama model: {model_name} for text validation")
        print(f"⏳ Validating text with Ollama (length: {len(text)} chars)...")
        
        # Debug generate function
        print("[DEBUG-OLLAMA] Checking ollama.generate function...")
        generate_func = getattr(ollama, "generate", None)
        if generate_func:
            print(f"[DEBUG-OLLAMA] ollama.generate function exists: {generate_func}")
        else:
            print("[DEBUG-OLLAMA] ⚠️ ollama.generate function not found!")
            print(f"[DEBUG-OLLAMA] Available functions: {[f for f in dir(ollama) if not f.startswith('_')]}")
        
        try:
            # Debug timestamp before calling generate
            import datetime
            generate_start = datetime.datetime.now()
            print(f"[DEBUG-OLLAMA] Calling ollama.generate() at {generate_start}")
            print(f"[DEBUG-OLLAMA] Parameters: model={model_name}, prompt_length={len(prompt)}")
            
            # Try to call generate with parameters
            try:
                response = ollama.generate(model=model_name, prompt=prompt)
                
                # Debug the response
                generate_end = datetime.datetime.now()
                generate_duration = (generate_end - generate_start).total_seconds()
                print(f"[DEBUG-OLLAMA] generate() completed in {generate_duration:.2f} seconds")
                print(f"[DEBUG-OLLAMA] Response type: {type(response)}")
                print(f"[DEBUG-OLLAMA] Response preview: {str(response)[:100]}..." if isinstance(response, (str, bytes)) else f"[DEBUG-OLLAMA] Non-string response: {response}")
            except Exception as gen_err:
                print(f"[DEBUG-OLLAMA] ❌ Error in ollama.generate(): {gen_err}")
                print(f"[DEBUG-OLLAMA] Error type: {type(gen_err).__name__}")
                raise
            
            # Parse the response
            try:
                print("[DEBUG-OLLAMA] Parsing response...")
                if isinstance(response, str):
                    try:
                        response_data = json.loads(response)
                        print(f"[DEBUG-OLLAMA] Response parsed as JSON: {type(response_data)}")
                        if isinstance(response_data, dict):
                            print(f"[DEBUG-OLLAMA] Response keys: {list(response_data.keys())}")
                        generated_text = response_data.get('response', '')
                        print(f"[DEBUG-OLLAMA] Generated text: {generated_text}")
                    except json.JSONDecodeError as json_err:
                        print(f"[DEBUG-OLLAMA] Response is not valid JSON: {json_err}")
                        # If not JSON, use the raw response
                        generated_text = response.strip()
                        print(f"[DEBUG-OLLAMA] Using raw response: {generated_text}")
                else:
                    # Handle non-string responses
                    print(f"[DEBUG-OLLAMA] Response is not a string, trying to extract text...")
                    if hasattr(response, 'response'):
                        # Some Ollama libraries return objects with a response attribute
                        generated_text = getattr(response, 'response', '')
                        print(f"[DEBUG-OLLAMA] Extracted from response object: {generated_text}")
                    elif hasattr(response, 'get') and callable(response.get):
                        # Some return dict-like objects
                        generated_text = response.get('response', '')
                        print(f"[DEBUG-OLLAMA] Extracted from dict-like object: {generated_text}")
                    else:
                        generated_text = str(response)
                        print(f"[DEBUG-OLLAMA] Converting to string: {generated_text}")
            except Exception as e:
                print(f"[DEBUG-OLLAMA] Error parsing response: {e}")
                generated_text = str(response) if response else ""
                print(f"[DEBUG-OLLAMA] Falling back to string conversion: {generated_text[:100]}...")
        except Exception as e:
            print(f"⚠️ Error generating with {model_name}: {e}")
            # Try with a different model if the first one fails
            try:
                # Choose a fallback model from the ones we know are available
                fallback_options = ["gemma3:1b", "deepseek-r1:1.5b", "phi4", "qwen2.5:1.5b"]
                fallback_model = None
                
                # Try to find a different model than the one that failed
                for option in fallback_options:
                    if option != model_name:
                        fallback_model = option
                        break
                
                # If we couldn't find a different model, use gemma3:4b as last resort
                if not fallback_model:
                    fallback_model = "gemma3:4b"
                    
                print(f"⏳ Trying with alternate model: {fallback_model}...")
                response = ollama.generate(model=fallback_model, prompt=prompt)
                
                # Parse the response
                if isinstance(response, str):
                    try:
                        response_data = json.loads(response)
                        generated_text = response_data.get('response', '')
                    except json.JSONDecodeError:
                        # If not JSON, use the raw response
                        generated_text = response.strip()
                else:
                    # Handle non-string responses
                    if hasattr(response, 'response'):
                        generated_text = getattr(response, 'response', '')
                    elif hasattr(response, 'get') and callable(response.get):
                        generated_text = response.get('response', '')
                    else:
                        generated_text = str(response)
            except Exception as e2:
                print(f"⚠️ Error with alternate model: {e2}")
                # If both models fail, use a simple heuristic
                print(f"⚠️ Both models failed, using length-based validation")
                if 10 <= len(text) <= 2000:
                    return 1  # Valid
                else:
                    return 0  # Invalid

        # Log summary of response
        print(f"📝 Ollama response summary: {generated_text[:100]}...")
        
        # Check for validation signal (looking for '1' anywhere in the response)
        if "1" in generated_text:
            print(f"✅ Text validated successfully (VALID)")
            return 1
        else:
            print(f"❌ Text validation failed (INVALID)")
            return 0

    except Exception as e:  # Catch any Ollama-related errors
        error_msg = f"Error validating text with Ollama: {e}"
        print(f"❌ {error_msg}")
        log_pending_error(error_msg)
        
        # Log additional details to help debug the issue
        import socket
        try:
            # Try to connect to Ollama's default port to see if it's running
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            result = s.connect_ex(('localhost', 11434))
            if result == 0:
                print("✅ Ollama server is running on port 11434")
                log_pending_error("Ollama server is running but text validation failed")
            else:
                print("❌ Ollama server not detected on port 11434")
                log_pending_error("Ollama server not detected on default port 11434")
            s.close()
        except Exception as conn_err:
            print(f"⚠️ Could not check Ollama server connection: {conn_err}")
        
        # As per requirements, we do NOT use fallback validation methods
        # If Ollama validation fails, the data point is considered invalid
        print("❌ Ollama validation failed - marking as INVALID")
        print("ℹ️ Not using fallback validation as per requirements")
        return 0  # Always return 0 (invalid) if Ollama validation fails

def synchronize_checkpoint_with_filesystem(checkpoint, dataset_name, version):
    """
    Synchronizes the checkpoint with the actual files on disk to ensure accuracy.
    Implements a robust approach with transaction tracking and detailed logging to
    ensure state consistency between checkpoint and filesystem.
    
    Args:
        checkpoint: The loaded checkpoint dictionary
        dataset_name: Name of the dataset
        version: Version of the dataset
        
    Returns:
        Updated checkpoint dictionary with synchronized state
    """
    print("[DEBUG-CLEAN] ℹ️ Synchronizing checkpoint with filesystem...")
    
    # Create a transaction marker to ensure atomicity
    transaction_id = int(time.time())
    checkpoint["sync_transaction"] = transaction_id
    checkpoint["transaction_status"] = "sync_in_progress"
    
    # Save checkpoint state before synchronization (for debugging/recovery)
    checkpoint_before_sync = checkpoint.copy()
    
    # Define data directories with error handling
    valid_dir = Path("./data/valid")
    invalid_dir = Path("./data/rejected")
    data_dir = Path("./data")
    
    # Ensure directories exist
    for dir_path in [data_dir, valid_dir, invalid_dir]:
        if not dir_path.exists():
            print(f"[DEBUG-CLEAN] Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # List files on disk with detailed logging and error handling
    valid_files_on_disk = []
    invalid_files_on_disk = []
    original_files_on_disk = []
    
    # Use try/except blocks to handle potential directory access issues
    try:
        valid_files_on_disk = [f.name for f in valid_dir.glob(f"{dataset_name}_{version}_valid_*.parquet")]
        print(f"[DEBUG-CLEAN] Found {len(valid_files_on_disk)} valid files on disk")
    except Exception as e:
        print(f"[DEBUG-CLEAN] ⚠️ Error accessing valid files directory: {e}")
    
    try:
        invalid_files_on_disk = [f.name for f in invalid_dir.glob(f"{dataset_name}_{version}_invalid_*.parquet")]
        print(f"[DEBUG-CLEAN] Found {len(invalid_files_on_disk)} invalid files on disk")
    except Exception as e:
        print(f"[DEBUG-CLEAN] ⚠️ Error accessing invalid files directory: {e}")
    
    try:
        original_files_on_disk = [f.name for f in data_dir.glob(f"{dataset_name}_{version}_original_*.parquet")]
        print(f"[DEBUG-CLEAN] Found {len(original_files_on_disk)} original files on disk")
    except Exception as e:
        print(f"[DEBUG-CLEAN] ⚠️ Error accessing original files directory: {e}")
    
    # Ensure checkpoint has required fields with default values
    if "valid_files" not in checkpoint:
        checkpoint["valid_files"] = []
    if "invalid_files" not in checkpoint:
        checkpoint["invalid_files"] = []
    if "processed_files" not in checkpoint:
        checkpoint["processed_files"] = []
    
    # Compare with checkpoint using sets for efficient operations
    checkpoint_valid = set(checkpoint.get("valid_files", []))
    checkpoint_invalid = set(checkpoint.get("invalid_files", []))
    checkpoint_processed = set(checkpoint.get("processed_files", []))
    
    # Track all detected differences for reconciliation
    valid_files_missing_from_checkpoint = set(valid_files_on_disk) - checkpoint_valid
    invalid_files_missing_from_checkpoint = set(invalid_files_on_disk) - checkpoint_invalid
    valid_files_missing_from_disk = checkpoint_valid - set(valid_files_on_disk)
    invalid_files_missing_from_disk = checkpoint_invalid - set(invalid_files_on_disk)
    
    # Log the sync details (time-stamped for debugging in logs)
    sync_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DEBUG-CLEAN] Checkpoint-Filesystem sync started at {sync_timestamp} (transaction {transaction_id})")
    
    # Track stats for summary report
    change_count = 0
    
    # First handle files missing from disk (checkpoint refers to non-existent files)
    if valid_files_missing_from_disk:
        print(f"[DEBUG-CLEAN] ⚠️ Found {len(valid_files_missing_from_disk)} valid files in checkpoint but missing from disk")
        for file in sorted(valid_files_missing_from_disk):
            print(f"[DEBUG-CLEAN]   - Missing valid file: {file}")
            # Log into processing history for audit trail
            if "processing_history" in checkpoint and file in checkpoint.get("processing_history", {}):
                checkpoint["processing_history"][file]["status"] = "missing_from_disk"
        
        # Remove missing files from checkpoint
        checkpoint["valid_files"] = list(checkpoint_valid - valid_files_missing_from_disk)
        print(f"[DEBUG-CLEAN] ✅ Removed {len(valid_files_missing_from_disk)} missing valid files from checkpoint")
        change_count += len(valid_files_missing_from_disk)
    
    if invalid_files_missing_from_disk:
        print(f"[DEBUG-CLEAN] ⚠️ Found {len(invalid_files_missing_from_disk)} invalid files in checkpoint but missing from disk")
        for file in sorted(invalid_files_missing_from_disk):
            print(f"[DEBUG-CLEAN]   - Missing invalid file: {file}")
            # Log into processing history for audit trail
            if "processing_history" in checkpoint and file in checkpoint.get("processing_history", {}):
                checkpoint["processing_history"][file]["status"] = "missing_from_disk"
        
        # Remove missing files from checkpoint
        checkpoint["invalid_files"] = list(checkpoint_invalid - invalid_files_missing_from_disk)
        print(f"[DEBUG-CLEAN] ✅ Removed {len(invalid_files_missing_from_disk)} missing invalid files from checkpoint")
        change_count += len(invalid_files_missing_from_disk)
        
    # Then handle files missing from checkpoint (files exist but checkpoint doesn't know about them)
    if valid_files_missing_from_checkpoint:
        print(f"[DEBUG-CLEAN] ⚠️ Found {len(valid_files_missing_from_checkpoint)} valid files on disk not in checkpoint")
        for file in sorted(valid_files_missing_from_checkpoint):
            print(f"[DEBUG-CLEAN]   - Adding valid file to checkpoint: {file}")
        
        # Add missing files to checkpoint
        checkpoint["valid_files"] = list(checkpoint_valid.union(valid_files_missing_from_checkpoint))
        print(f"[DEBUG-CLEAN] ✅ Updated valid_files in checkpoint (now {len(checkpoint['valid_files'])} files)")
        change_count += len(valid_files_missing_from_checkpoint)
        
    if invalid_files_missing_from_checkpoint:
        print(f"[DEBUG-CLEAN] ⚠️ Found {len(invalid_files_missing_from_checkpoint)} invalid files on disk not in checkpoint")
        for file in sorted(invalid_files_missing_from_checkpoint):
            print(f"[DEBUG-CLEAN]   - Adding invalid file to checkpoint: {file}")
        
        # Add missing files to checkpoint
        checkpoint["invalid_files"] = list(checkpoint_invalid.union(invalid_files_missing_from_checkpoint))
        print(f"[DEBUG-CLEAN] ✅ Updated invalid_files in checkpoint (now {len(checkpoint['invalid_files'])} files)")
        change_count += len(invalid_files_missing_from_checkpoint)
    
    # Build file mapping with improved error handling
    all_processed_files = set()
    file_to_output_mapping = {}  # Will track which original file maps to which output file
    
    # Track validation categories for reporting
    mapping_success_count = 0
    mapping_error_count = 0
    
    # Build timestamp lookup tables for the full timestamp format
    valid_timestamps = {}
    for f in valid_files_on_disk:
        try:
            # Extract the timestamp part after "_valid_"
            timestamp_part = f.split("_valid_")[1].split(".")[0]
            valid_timestamps[timestamp_part] = f
        except Exception as e:
            print(f"[DEBUG-CLEAN] ⚠️ Error parsing valid filename {f}: {e}")
            mapping_error_count += 1
    
    invalid_timestamps = {}
    for f in invalid_files_on_disk:
        try:
            # Extract the timestamp part after "_invalid_"
            timestamp_part = f.split("_invalid_")[1].split(".")[0]
            invalid_timestamps[timestamp_part] = f
        except Exception as e:
            print(f"[DEBUG-CLEAN] ⚠️ Error parsing invalid filename {f}: {e}")
            mapping_error_count += 1
    
    # Map original files to their processed outputs
    for original_file in original_files_on_disk:
        # Extract timestamp from original file
        try:
            timestamp = original_file.split("_original_")[1].split(".")[0]
            
            # Direct lookup by timestamp - more accurate than string manipulation
            if timestamp in valid_timestamps or timestamp in invalid_timestamps:
                all_processed_files.add(original_file)
                
                # Track which output file corresponds to this original file
                if timestamp in valid_timestamps:
                    valid_output = valid_timestamps[timestamp]
                    file_to_output_mapping[original_file] = valid_output
                    
                    # Update processing history for this file if not already present
                    if "processing_history" not in checkpoint:
                        checkpoint["processing_history"] = {}
                    
                    if original_file not in checkpoint.get("processing_history", {}):
                        # Create a new history entry for this file
                        checkpoint["processing_history"][original_file] = {
                            "processed_at": int(time.time()),
                            "outputs": {
                                "valid": valid_output,
                                "invalid": None
                            },
                            "counts": {
                                "valid": 1, 
                                "invalid": 0,
                                "total": 1
                            },
                            "status": "valid_detected_during_sync"
                        }
                        print(f"[DEBUG-CLEAN] ℹ️ Added missing processing history for {original_file}")
                    
                    mapping_success_count += 1
                    
                if timestamp in invalid_timestamps:
                    invalid_output = invalid_timestamps[timestamp]
                    file_to_output_mapping[original_file] = invalid_output
                    
                    # Update processing history for this file if not already present
                    if "processing_history" not in checkpoint:
                        checkpoint["processing_history"] = {}
                    
                    if original_file not in checkpoint.get("processing_history", {}):
                        # Create a new history entry for this file
                        checkpoint["processing_history"][original_file] = {
                            "processed_at": int(time.time()),
                            "outputs": {
                                "valid": None,
                                "invalid": invalid_output
                            },
                            "counts": {
                                "valid": 0, 
                                "invalid": 1,
                                "total": 1
                            },
                            "status": "invalid_detected_during_sync"
                        }
                        print(f"[DEBUG-CLEAN] ℹ️ Added missing processing history for {original_file}")
                    
                    mapping_success_count += 1
        except Exception as e:
            print(f"[DEBUG-CLEAN] ⚠️ Error extracting timestamp from {original_file}: {e}")
            mapping_error_count += 1
    
    # Handle files marked as processed but with no output files
    processed_without_outputs = checkpoint_processed - all_processed_files
    if processed_without_outputs:
        print(f"[DEBUG-CLEAN] ⚠️ Found {len(processed_without_outputs)} files in checkpoint marked as processed but with no output files")
        
        # Configurable option through environment variable
        remove_orphaned_entries = os.environ.get("CLEAN_ORPHANED_ENTRIES", "").lower() == "true"
        
        if remove_orphaned_entries:
            # Option 1: Remove these from processed list to allow reprocessing
            checkpoint["processed_files"] = list(checkpoint_processed - processed_without_outputs)
            print(f"[DEBUG-CLEAN] ✅ Removed inconsistent files from processed list (CLEAN_ORPHANED_ENTRIES=true)")
            
            # Mark these files in the processing history
            for file in processed_without_outputs:
                if "processing_history" in checkpoint and file in checkpoint.get("processing_history", {}):
                    checkpoint["processing_history"][file]["status"] = "output_missing_reprocessing_scheduled"
            
            change_count += len(processed_without_outputs)
        else:
            # Option 2: Keep them in processed list to prevent endless processing cycles
            print(f"[DEBUG-CLEAN] ℹ️ Keeping these in processed list to prevent reprocessing loops (set CLEAN_ORPHANED_ENTRIES=true to change this)")
            
            # Still mark these files in the processing history
            for file in processed_without_outputs:
                if "processing_history" in checkpoint and file in checkpoint.get("processing_history", {}):
                    checkpoint["processing_history"][file]["status"] = "output_missing_but_preserved"
    
    # Track files that have outputs but aren't marked as processed
    processed_files_missing_from_checkpoint = all_processed_files - checkpoint_processed
    if processed_files_missing_from_checkpoint:
        print(f"[DEBUG-CLEAN] ⚠️ Found {len(processed_files_missing_from_checkpoint)} processed files not in checkpoint")
        
        # Add these files to the processed list
        checkpoint["processed_files"] = list(checkpoint_processed.union(processed_files_missing_from_checkpoint))
        print(f"[DEBUG-CLEAN] ✅ Updated processed_files in checkpoint (now {len(checkpoint['processed_files'])} files)")
        change_count += len(processed_files_missing_from_checkpoint)
    
    # Save the mapping for later use during processing
    checkpoint["file_to_output_mapping"] = file_to_output_mapping
    
    # Add a verification timestamp and sync stats
    checkpoint["last_verified"] = int(time.time())
    checkpoint["last_sync_transaction"] = transaction_id
    checkpoint["transaction_status"] = "sync_complete"
    checkpoint["last_sync_changes"] = change_count
    checkpoint["last_sync_successful_mappings"] = mapping_success_count
    checkpoint["last_sync_error_mappings"] = mapping_error_count
    
    # Print sync summary
    print(f"[DEBUG-CLEAN] ✅ Checkpoint-Filesystem sync complete")
    print(f"[DEBUG-CLEAN]   - Changes made: {change_count}")
    print(f"[DEBUG-CLEAN]   - Successful file mappings: {mapping_success_count}")
    print(f"[DEBUG-CLEAN]   - Mapping errors: {mapping_error_count}")
    print(f"[DEBUG-CLEAN]   - Transaction ID: {transaction_id}")
    
    return checkpoint

def atomic_load_checkpoint():
    """
    Load the validation checkpoint using atomic operations.
    
    This function provides robust error handling:
    1. Tries to load the main checkpoint file
    2. If that fails, tries to load the backup file
    3. If that fails, creates a new default checkpoint
    
    Returns:
        dict: The loaded checkpoint data
    """
    # Define default checkpoint structure with all required fields
    default_checkpoint = {
        "processed_files": [],      # Files already processed (each file = one data point)
        "last_processed_timestamp": "", 
        "valid_files": [],          # Files containing valid data points
        "invalid_files": [],        # Files containing invalid data points
        "checkpoint_version": 1,    # Track checkpoint schema version for upgrades
        "last_reprocessed": 0,      # Unix timestamp of the last reprocessing run
        "transaction_id": None,     # Transaction ID for atomic operations
        "transaction_status": "complete"  # Status of the last transaction
    }
    
    # Create backup file path
    backup_file = Path(str(CHECKPOINT_FILE) + ".bak")
    fixed_file = Path(str(CHECKPOINT_FILE) + "_fixed.json")
    
    # Check corruption flag (for recovery)
    corruption_detected = False
    recovery_source = None
    
    # Try to load main checkpoint file
    loaded_checkpoint = None
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                loaded_checkpoint = json.load(f)
                recovery_source = "main"
                print(f"[DEBUG-CHECKPOINT] ✅ Successfully loaded checkpoint from main file")
        except Exception as e:
            print(f"[DEBUG-CHECKPOINT] ⚠️ Warning: Could not read main checkpoint file: {e}")
            corruption_detected = True
    
    # If main file failed, try backup
    if loaded_checkpoint is None and backup_file.exists():
        try:
            with open(backup_file, 'r') as f:
                loaded_checkpoint = json.load(f)
                recovery_source = "backup"
                print(f"[DEBUG-CHECKPOINT] ⚠️ Recovered checkpoint from backup file")
        except Exception as e:
            print(f"[DEBUG-CHECKPOINT] ⚠️ Warning: Could not read backup checkpoint file: {e}")
    
    # If fixed version exists, try that as last resort
    if loaded_checkpoint is None and fixed_file.exists():
        try:
            with open(fixed_file, 'r') as f:
                loaded_checkpoint = json.load(f)
                recovery_source = "fixed"
                print(f"[DEBUG-CHECKPOINT] ⚠️ Recovered checkpoint from fixed backup file")
        except Exception as e:
            print(f"[DEBUG-CHECKPOINT] ⚠️ Warning: Could not read fixed checkpoint file: {e}")
    
    # If all files failed or don't exist, use default
    if loaded_checkpoint is None:
        loaded_checkpoint = default_checkpoint
        print(f"[DEBUG-CHECKPOINT] ℹ️ No valid checkpoint found, using default")
        recovery_source = "default"
    
    # Apply schema upgrades and defaults
    final_checkpoint = default_checkpoint.copy()
    for key, value in loaded_checkpoint.items():
        if key in final_checkpoint or key == "processing_history":
            final_checkpoint[key] = value
    
    # Handle transaction recovery if needed
    if final_checkpoint.get("transaction_status") == "in_progress":
        print(f"[DEBUG-CHECKPOINT] ⚠️ Detected incomplete transaction {final_checkpoint.get('transaction_id')}")
        final_checkpoint["transaction_status"] = "recovered"
        corruption_detected = True
    
    # If corruption was detected, save the fixed version
    if corruption_detected:
        print(f"[DEBUG-CHECKPOINT] 🔄 Checkpoint corruption detected, creating fixed backup")
        try:
            with open(fixed_file, 'w') as f:
                json.dump(final_checkpoint, f, indent=2)
            print(f"[DEBUG-CHECKPOINT] ✅ Saved fixed checkpoint to {fixed_file}")
        except Exception as e:
            print(f"[DEBUG-CHECKPOINT] ⚠️ Warning: Could not save fixed checkpoint: {e}")
    
    # Log recovery information
    if recovery_source != "main":
        print(f"[DEBUG-CHECKPOINT] ℹ️ Checkpoint recovery complete from {recovery_source} source")
    
    return final_checkpoint

def atomic_save_checkpoint(checkpoint_data):
    """
    Save the validation checkpoint using atomic file operations.
    
    This approach ensures that the checkpoint file is either completely
    written or not changed at all, preventing corruption during crashes.
    
    Args:
        checkpoint_data (dict): The checkpoint data to save
    """
    # Create a unique transaction ID if not already present
    if "transaction_id" not in checkpoint_data or checkpoint_data["transaction_id"] is None:
        checkpoint_data["transaction_id"] = int(time.time())
    
    # Mark transaction as in progress
    checkpoint_data["transaction_status"] = "in_progress"
    checkpoint_data["last_modified"] = int(time.time())
    
    # Create temp file path (same directory as checkpoint)
    temp_file = Path(str(CHECKPOINT_FILE) + ".tmp")
    backup_file = Path(str(CHECKPOINT_FILE) + ".bak")
    
    # Create parent directory if it doesn't exist
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # First write to temp file
        with open(temp_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Create backup of current file if it exists
        if CHECKPOINT_FILE.exists():
            try:
                shutil.copy2(CHECKPOINT_FILE, backup_file)
            except Exception as e:
                print(f"[DEBUG-CHECKPOINT] ⚠️ Warning: Could not create backup: {e}")
        
        # Atomic rename of temp file to final file
        try:
            # On Windows, we need to remove destination first
            if CHECKPOINT_FILE.exists():
                CHECKPOINT_FILE.unlink()
            temp_file.rename(CHECKPOINT_FILE)
        except Exception as e:
            # If rename fails, try copy and delete approach
            print(f"[DEBUG-CHECKPOINT] ⚠️ Atomic rename failed: {e}")
            shutil.copy2(temp_file, CHECKPOINT_FILE)
            temp_file.unlink()
        
        # Update transaction status to complete
        checkpoint_data["transaction_status"] = "complete"
        
        # Update the saved file with completed status
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        print(f"[DEBUG-CHECKPOINT] ✅ Checkpoint saved successfully (transaction {checkpoint_data['transaction_id']})")
        return True
        
    except Exception as e:
        print(f"[DEBUG-CHECKPOINT] ❌ Error saving checkpoint: {e}")
        # Try to clean up temp file
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception as cleanup_error:
            print(f"[DEBUG-CHECKPOINT] ⚠️ Warning: Could not clean up temp file: {cleanup_error}")
            # Continue despite the error - this is not critical
        return False

# Legacy functions for backward compatibility
def load_checkpoint():
    """Legacy function that calls atomic_load_checkpoint."""
    return atomic_load_checkpoint()

def save_checkpoint(checkpoint_data):
    """Legacy function that calls atomic_save_checkpoint."""
    return atomic_save_checkpoint(checkpoint_data)

@step
@safe_run
def clean_and_filter_data(dataset_name: str = "user_data_ui", version: str = "v1", disable_cache: bool = False) -> Tuple[
    Annotated[pd.DataFrame, "clean_data"],
    Annotated[bool, "dataset_ok"],
    Annotated[str, "training_file_path"]
]:
    """
    Loads, cleans, and filters data with checkpoint support.
    
    Returns:
        clean_data: DataFrame containing cleaned valid data
        dataset_ok: Boolean indicating if threshold was met
        training_file_path: Path to the training file created (or empty string if none created)
    """
    print("\n" + "="*80)
    print("PIPELINE STATUS: Starting data cleaning and filtering stage")
    print("="*80)
    
    # Create a marker file to indicate cleaning is in progress
    # This will prevent the delete_all_datasets function from running during cleaning
    from pathlib import Path
    import time
    import atexit
    
    CLEANING_IN_PROGRESS_FILE = Path("./.cleaning_in_progress")
    
    # Create an enhanced marker file with timestamp and process ID
    try:
        import os
        import json
        
        clean_marker_data = {
            "start_time": time.time(),
            "pid": os.getpid(),
            "session_id": f"clean_{int(time.time())}_{os.getpid()}"
        }
        
        with open(CLEANING_IN_PROGRESS_FILE, 'w') as f:
            json.dump(clean_marker_data, f)
            
        # Register a function to remove the marker when the process exits
        def remove_cleaning_marker():
            if CLEANING_IN_PROGRESS_FILE.exists():
                try:
                    CLEANING_IN_PROGRESS_FILE.unlink()
                    print("ℹ️ Removed cleaning in progress marker on exit")
                except Exception as e:
                    print(f"⚠️ Warning: Could not remove cleaning marker on exit: {e}")
                    
        atexit.register(remove_cleaning_marker)
        
        print(f"ℹ️ Created cleaning in progress marker with session ID: {clean_marker_data['session_id']}")
    except Exception as e:
        # Fallback to simple marker file if anything fails
        CLEANING_IN_PROGRESS_FILE.touch()
        print(f"ℹ️ Created simple cleaning in progress marker (error: {e})")
    
    # Add timestamp for debugging
    import datetime
    import os
    print(f"⏱️ DATA CLEANING STARTED AT: {datetime.datetime.now()}")
    print(f"⏱️ Running in directory: {os.getcwd()}")
    
    # Print detailed Ollama information
    print("\n" + "="*50)
    print("OLLAMA CONFIGURATION CHECK")
    print("="*50)
    print(f"🤖 Primary Ollama model: {OLLAMA_MODEL_NAME}")
    print(f"🔄 Fallback models: {OLLAMA_FALLBACK_MODELS}")
    
    # Check Ollama server connection
    import socket
    try:
        print("[DEBUG-OLLAMA] 🔌 Checking Ollama server connection...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        result = s.connect_ex(('localhost', 11434))
        if result == 0:
            print("[DEBUG-OLLAMA] ✅ Ollama server is running on port 11434")
            
            # Try to get Ollama version directly
            try:
                import ollama
                if hasattr(ollama, 'version'):
                    version_info = ollama.version()
                    print(f"[DEBUG-OLLAMA] ✅ Ollama version: {version_info}")
                else:
                    print("[DEBUG-OLLAMA] Ollama library doesn't have version() function")
                    
                    # Alternative: try using REST API
                    import urllib.request
                    import json
                    try:
                        response = urllib.request.urlopen('http://localhost:11434/api/version')
                        version_data = json.loads(response.read().decode())
                        print(f"[DEBUG-OLLAMA] ✅ Ollama version from API: {version_data}")
                    except Exception as api_err:
                        print(f"[DEBUG-OLLAMA] ⚠️ Could not get version from API: {api_err}")
            except Exception as ver_err:
                print(f"[DEBUG-OLLAMA] ⚠️ Could not get Ollama version: {ver_err}")
        else:
            print("[DEBUG-OLLAMA] ❌ Ollama server not detected on port 11434")
            print("[DEBUG-OLLAMA] ⚠️ WARNING: Data validation will likely fail without Ollama server")
        s.close()
    except Exception as conn_err:
        print(f"[DEBUG-OLLAMA] ⚠️ Could not check Ollama server connection: {conn_err}")
    print("="*50 + "\n")
    
    # Verify data directory structure
    print(f"[DEBUG-CLEAN] Checking data directory structure")
    if DATA_DIR.exists():
        print(f"[DEBUG-CLEAN] ✅ Data directory exists: {DATA_DIR.absolute()}")
        # List all files in the data directory
        all_files_in_dir = list(DATA_DIR.glob("*"))
        print(f"[DEBUG-CLEAN] Found {len(all_files_in_dir)} total files/dirs in data directory")
        
        # Count different types of parquet files for debugging the "16 parquet files" issue
        parquet_files = [f for f in all_files_in_dir if f.is_file() and f.suffix == '.parquet']
        print(f"[DEBUG-CLEAN] 📊 PARQUET FILE COUNT: {len(parquet_files)} total parquet files")
        
        # Count by pattern
        pattern_counts = {}
        for pattern in [f"{dataset_name}_{version}_original_*.parquet", 
                        f"{dataset_name}_{version}_valid_*.parquet", 
                        f"{dataset_name}_{version}_invalid_*.parquet",
                        f"{dataset_name}_{version}_cleaned_*.parquet",
                        f"{dataset_name}_{version}_consolidated_*.parquet"]:
            count = len(list(DATA_DIR.glob(pattern)))
            pattern_counts[pattern] = count
            print(f"[DEBUG-CLEAN]   - Pattern '{pattern}': {count} files")
        
        # List sample files
        if all_files_in_dir:
            print(f"[DEBUG-CLEAN] Sample files/directories in data dir:")
            for item in all_files_in_dir[:5]:  # Show the first 5 items
                if item.is_file():
                    print(f"[DEBUG-CLEAN]   - File: {item.name} ({item.stat().st_size} bytes)")
                else:
                    print(f"[DEBUG-CLEAN]   - Dir: {item.name}")
    else:
        print(f"[DEBUG-CLEAN] ❌ Data directory does not exist: {DATA_DIR.absolute()}")
        print(f"[DEBUG-CLEAN] Creating data directory...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for required directories
    for dir_path, dir_name in [(REJECTED_DIR, "Rejected"), (VALID_DIR, "Valid")]:
        if dir_path.exists():
            print(f"[DEBUG-CLEAN] ✅ {dir_name} directory exists: {dir_path.absolute()}")
        else:
            print(f"[DEBUG-CLEAN] Creating {dir_name.lower()} directory: {dir_path.absolute()}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint with improved atomic operations and error recovery
    print("[DEBUG-CLEAN] 🔒 Loading checkpoint with atomic operations and recovery support")
    checkpoint = atomic_load_checkpoint()
    
    # Record processing operation start time
    operation_start_time = int(time.time())
    checkpoint["last_operation_start"] = operation_start_time
    checkpoint["operation_type"] = "clean_and_filter"
    
    # If disable_cache is True, handle reprocessing with improved safeguards
    if disable_cache:
        print("[DEBUG-CLEAN] 🔄 Cache disabled - checking if reprocessing is needed")
        
        # Get current time for reprocessing timestamp
        current_time = int(time.time())
        
        # Check for cooldown period between reprocessing runs (prevent endless revalidation)
        reprocessing_cooldown = int(os.environ.get("REPROCESSING_COOLDOWN", "300"))  # Default: 5 minutes between full reprocessing
        time_since_last_reprocessed = current_time - checkpoint.get("last_reprocessed", 0)
        
        # Count unprocessed files by comparing original vs processed
        original_files_on_disk = [f.name for f in DATA_DIR.glob(f"{dataset_name}_{version}_original_*.parquet")]
        processed_files_set = set(checkpoint.get("processed_files", []))
        unprocessed_count = len([f for f in original_files_on_disk if f not in processed_files_set])
        
        # Record counts in checkpoint for diagnostics
        checkpoint["last_scan_original_count"] = len(original_files_on_disk)
        checkpoint["last_scan_processed_count"] = len(processed_files_set)
        checkpoint["last_scan_unprocessed_count"] = unprocessed_count
        
        if unprocessed_count > 0:
            # If there are unprocessed files, we should only process those (don't reprocess everything)
            print(f"[DEBUG-CLEAN] 📈 Detected {unprocessed_count} unprocessed data files")
            print(f"[DEBUG-CLEAN] ℹ️ Will only process new files, skipping already processed ones")
            # Don't clear the processed files list - we'll just process the new ones
            
            # Save diagnostic information
            checkpoint["process_decision"] = "process_new_files_only"
            
        elif time_since_last_reprocessed < reprocessing_cooldown:
            # If all files are already processed and we're in cooldown, don't reprocess
            cooldown_remaining = reprocessing_cooldown - time_since_last_reprocessed
            print(f"[DEBUG-CLEAN] ⚠️ Reprocessing cooldown active: {cooldown_remaining} seconds remaining")
            print(f"[DEBUG-CLEAN] ℹ️ Will use existing checkpoint data to prevent endless revalidation")
            # We don't clear the processed files list to prevent endless reprocessing
            
            # Save diagnostic information
            checkpoint["process_decision"] = "cooldown_active"
            checkpoint["cooldown_remaining"] = cooldown_remaining
            
        else:
            # Only do full reprocessing if cooldown has passed AND force_reprocess is explicitly requested
            force_reprocess = os.environ.get("FORCE_REPROCESS", "").lower() == "true"
            if force_reprocess:
                print(f"[DEBUG-CLEAN] 🔄 Reprocessing cooldown passed and FORCE_REPROCESS=true")
                print(f"[DEBUG-CLEAN] 🔄 Clearing processed files list to force reprocessing all files")
                
                # Create backup of current state before clearing
                checkpoint["pre_reprocess_state"] = {
                    "processed_files_count": len(checkpoint.get("processed_files", [])),
                    "valid_files_count": len(checkpoint.get("valid_files", [])),
                    "invalid_files_count": len(checkpoint.get("invalid_files", [])),
                    "reprocess_timestamp": current_time
                }
                
                # Clear processed files list
                checkpoint["processed_files"] = []
                checkpoint["last_reprocessed"] = current_time
                print(f"[DEBUG-CLEAN] 📝 Full reprocessing triggered at {time.ctime(current_time)}")
                
                # Save diagnostic information
                checkpoint["process_decision"] = "force_reprocess_triggered"
                
            else:
                print(f"[DEBUG-CLEAN] ℹ️ No unprocessed files and no FORCE_REPROCESS flag")
                print(f"[DEBUG-CLEAN] ℹ️ Using existing checkpoint data (set FORCE_REPROCESS=true to override)")
                
                # Save diagnostic information
                checkpoint["process_decision"] = "no_changes_needed"
    
    # Save the checkpoint after cache decision but before synchronization
    # This ensures we capture the decision state
    atomic_save_checkpoint(checkpoint)
    
    # Log synchronization start
    print("[DEBUG-CLEAN] 🔄 Starting checkpoint-filesystem synchronization")
    sync_start_time = int(time.time())
    
    # Synchronize checkpoint with filesystem to ensure accuracy
    # This will detect and reconcile any inconsistencies between checkpoint and files
    checkpoint = synchronize_checkpoint_with_filesystem(checkpoint, dataset_name, version)
    
    # Log synchronization completion and duration
    sync_end_time = int(time.time())
    sync_duration = sync_end_time - sync_start_time
    print(f"[DEBUG-CLEAN] ✅ Checkpoint synchronization completed in {sync_duration} seconds")
    
    # Save the synchronized checkpoint immediately
    # This ensures we don't lose the synchronized state if processing fails
    atomic_save_checkpoint(checkpoint)
    
    # Get updated lists from synchronized checkpoint
    processed_files = set(checkpoint["processed_files"])
    valid_files = set(checkpoint["valid_files"])
    invalid_files = set(checkpoint.get("invalid_files", []))
    
    print(f"[DEBUG-CLEAN] Loaded and synchronized checkpoint")
    print(f"[DEBUG-CLEAN] Checkpoint summary:")
    print(f"[DEBUG-CLEAN]   - Processed files: {len(processed_files)}")
    print(f"[DEBUG-CLEAN]   - Valid files: {len(valid_files)}")
    print(f"[DEBUG-CLEAN]   - Invalid files: {len(invalid_files)}")
    
    # Get all original data files
    glob_pattern = f"{dataset_name}_{version}_original_*.parquet"
    print(f"[DEBUG-CLEAN] Searching for data files with pattern: {glob_pattern}")
    all_files = list(DATA_DIR.glob(glob_pattern))
    
    if not all_files:
        print("\n❌ DATA COLLECTION STATUS: No original data files found")
        print(f"❌ No files matching pattern: {dataset_name}_{version}_original_*.parquet")
        print("ℹ️ RECOMMENDED ACTION: Use the data collection interface to submit more data samples")
        print("="*80 + "\n")
        return pd.DataFrame(), False

    # Sort files by creation time to ensure we process them in order
    all_files.sort(key=lambda f: f.stat().st_mtime)
    
    # Enhanced checkpoint verification step
    # Check both the processed files list and look for corresponding valid/invalid files
    new_files = []
    processed_file_names = set(processed_files)
    valid_file_names = set(valid_files)
    invalid_file_names = set(invalid_files)
    
    # For each original file, check if it should be processed
    for file in all_files:
        file_name = file.name
        original_timestamp = file_name.split('_')[-1].split('.')[0]  # Extract timestamp
        
        # Check if this file is already in processed list
        already_processed = file_name in processed_file_names
        
        # Check if there's a corresponding valid or invalid file with the same timestamp
        valid_name = f"{dataset_name}_{version}_valid_{original_timestamp}.parquet"
        invalid_name = f"{dataset_name}_{version}_invalid_{original_timestamp}.parquet"
        has_valid = valid_name in valid_file_names
        has_invalid = invalid_name in invalid_file_names
        
        # Also physically check if files exist (extra safeguard)
        valid_path = VALID_DIR / valid_name
        invalid_path = REJECTED_DIR / invalid_name
        valid_exists = valid_path.exists()
        invalid_exists = invalid_path.exists()
        
        # Extra logging for checkpoint verification
        if already_processed:
            print(f"[DEBUG-CLEAN] ✓ File {file_name} is already marked as processed in checkpoint")
            
            # Double-check that it has an output file using both methods
            # 1. Check if file_to_output_mapping exists in checkpoint and contains the file
            has_output_via_mapping = False
            output_file = None
            if "file_to_output_mapping" in checkpoint and file_name in checkpoint["file_to_output_mapping"]:
                has_output_via_mapping = True
                output_file = checkpoint["file_to_output_mapping"][file_name]
            
            # 2. Check the various other detection methods as fallback
            has_output_via_checks = has_valid or has_invalid or valid_exists or invalid_exists
            
            if has_output_via_mapping:
                print(f"[DEBUG-CLEAN] ✅ Verified file {file_name} has output: {output_file}")
            elif has_output_via_checks:
                print(f"[DEBUG-CLEAN] ✅ Verified file {file_name} has output (detected via checks)")
            else:
                print(f"[DEBUG-CLEAN] ⚠️ WARNING: File {file_name} is marked as processed but no valid/invalid output file found")
                print(f"[DEBUG-CLEAN] ℹ️ Will reprocess this file to ensure consistency")
                new_files.append(file)
        else:
            # Check if there's a corresponding output file even if not in checkpoint
            if has_valid or has_invalid or valid_exists or invalid_exists:
                print(f"[DEBUG-CLEAN] ⚠️ File {file_name} has output files but is not in processed list")
                print(f"[DEBUG-CLEAN] ℹ️ Adding to processed list to avoid duplicate processing")
                processed_file_names.add(file_name)
            else:
                # Genuinely new file that needs processing
                print(f"[DEBUG-CLEAN] 🆕 New file detected: {file_name}")
                new_files.append(file)
    
    # Update checkpoint with any corrections, removing duplicates
    checkpoint["processed_files"] = list(set(processed_file_names))
    checkpoint["transaction_status"] = "verification_complete"
    atomic_save_checkpoint(checkpoint)
    
    print(f"[DEBUG-CLEAN] Found {len(all_files)} total original files, {len(new_files)} are new and need processing")
    
    # Process new files
    newly_processed_files = []
    newly_valid_files = []
    
    if new_files:
        try:
            # Import Ollama to check availability before validation
            try:
                import ollama
                print(f"[DEBUG-OLLAMA] ✅ Successfully imported Ollama library")
                print(f"[DEBUG-OLLAMA] Ollama version: {getattr(ollama, '__version__', 'unknown')}")
                print(f"[DEBUG-OLLAMA] Ollama path: {getattr(ollama, '__file__', 'unknown')}")
            except ImportError as e:
                print(f"[DEBUG-OLLAMA] ❌ Failed to import Ollama: {e}")
                print(f"[DEBUG-OLLAMA] ⚠️ WARNING: Data validation may fail without Ollama")
            
            for i, file_path in enumerate(new_files):
                print(f"\n{'='*50}")
                print(f"PROCESSING FILE {i+1}/{len(new_files)}: {file_path.name}")
                print(f"{'='*50}")
                
                # Read the file
                print(f"[DEBUG-CLEAN] Reading file: {file_path.name} ({file_path.stat().st_size} bytes)")
                df = pd.read_parquet(file_path)
                print(f"[DEBUG-CLEAN] File contains {len(df)} data points")
                
                # Validate each entry
                valid_results = []
                valid_count = 0
                invalid_count = 0
                
                for j, row in enumerate(df.iterrows()):
                    idx, data_row = row  # df.iterrows() returns (index, Series) tuples
                    text = data_row["text"]
                    print(f"\n{'-'*40}")
                    print(f"🔍 VALIDATING DATA POINT #{j+1}/{len(df)} FROM FILE {i+1}/{len(new_files)}")
                    print(f"{'-'*40}")
                    print(f"📝 Preview: {text[:50]}..." if len(text) > 50 else f"📝 Text: {text}")
                    
                    # Validate
                    print(f"[DEBUG-OLLAMA] Sending data point to Ollama for validation...")
                    valid = validate_text_with_ollama(text)
                    valid_results.append(valid)
                    
                    # Print result with clear separation
                    if valid:
                        valid_count += 1
                        print(f"✅ VALIDATION RESULT: Data point #{j+1} is VALID")
                        print(f"✅ Running total in this file: {valid_count} valid, {invalid_count} invalid")
                    else:
                        invalid_count += 1
                        print(f"❌ VALIDATION RESULT: Data point #{j+1} is INVALID")
                        print(f"❌ Running total in this file: {valid_count} valid, {invalid_count} invalid")
                    print(f"{'-'*40}\n")
                
                # Add validation results to dataframe
                df["valid"] = valid_results
                
                # Split into valid and invalid dataframes
                valid_df = df[df["valid"] == 1].copy()
                invalid_df = df[df["valid"] == 0].copy()
                
                # Generate timestamps for filenames
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save valid data points
                if not valid_df.empty:
                    # Remove the validation column
                    valid_df.drop(columns=["valid"], inplace=True)
                    
                    # Construct valid file path preserving the full original timestamp with date
                    # Extract the full timestamp from the filename (including date)
                    # The format should be like 20250321_162928
                    parts = file_path.stem.split('_')
                    if len(parts) >= 5:  # we expect dataset_version_original_date_time
                        date_part = parts[-2]  # e.g., 20250321
                        time_part = parts[-1]  # e.g., 162928
                        original_timestamp = f"{date_part}_{time_part}"
                    else:
                        # Fallback if filename doesn't match expected format
                        original_timestamp = file_path.stem.split('_')[-1]
                        
                    valid_filename = f"{dataset_name}_{version}_valid_{original_timestamp}.parquet"
                    valid_filepath = VALID_DIR / valid_filename
                    
                    # Save valid data points
                    valid_df.to_parquet(valid_filepath)
                    print(f"✅ {len(valid_df)} valid data points saved to {valid_filepath}")
                    
                    # Add to newly validated files list
                    newly_valid_files.append(valid_filepath.name)
                
                # Save invalid data points
                if not invalid_df.empty:
                    # Remove the validation column
                    invalid_df.drop(columns=["valid"], inplace=True)
                    
                    # Construct invalid file path preserving the full original timestamp with date
                    # Extract the full timestamp from the filename (including date)
                    # The format should be like 20250321_162928
                    parts = file_path.stem.split('_')
                    if len(parts) >= 5:  # we expect dataset_version_original_date_time
                        date_part = parts[-2]  # e.g., 20250321
                        time_part = parts[-1]  # e.g., 162928
                        original_timestamp = f"{date_part}_{time_part}"
                    else:
                        # Fallback if filename doesn't match expected format
                        original_timestamp = file_path.stem.split('_')[-1]
                        
                    invalid_filename = f"{dataset_name}_{version}_invalid_{original_timestamp}.parquet"
                    invalid_filepath = REJECTED_DIR / invalid_filename
                    
                    # Save invalid data points
                    invalid_df.to_parquet(invalid_filepath)
                    print(f"ℹ️ {len(invalid_df)} invalid data points saved to {invalid_filepath}")
                
                # Add file to processed list
                newly_processed_files.append(file_path.name)
                
                # Begin checkpoint update transaction
                checkpoint["transaction_status"] = "file_processing_update"
                checkpoint["transaction_id"] = int(time.time())
                
                # Update checkpoint after each file to ensure we don't reprocess
                # if the pipeline is interrupted - use set to avoid duplicates
                checkpoint["processed_files"] = list(set(list(processed_files) + newly_processed_files))
                checkpoint["valid_files"] = list(set(list(valid_files) + newly_valid_files))
                
                # Also track invalid files more carefully
                if not invalid_df.empty:
                    if "invalid_files" not in checkpoint:
                        checkpoint["invalid_files"] = []
                    
                    # Add the new invalid file to the checkpoint (ensure no duplicates)
                    if invalid_filepath.name not in checkpoint["invalid_files"]:
                        checkpoint["invalid_files"].append(invalid_filepath.name)
                        print(f"[DEBUG-CLEAN]   - Added {invalid_filepath.name} to checkpoint invalid_files list (now {len(checkpoint['invalid_files'])} files)")
                    else:
                        print(f"[DEBUG-CLEAN]   - File {invalid_filepath.name} already in checkpoint invalid_files list")
                
                # Get current time for timestamps
                current_time = int(time.time())
                
                # Update timestamps
                checkpoint["last_processed_timestamp"] = timestamp
                checkpoint["last_file_processed_time"] = current_time
                
                # Track detailed processing history
                if "processing_history" not in checkpoint:
                    checkpoint["processing_history"] = {}
                    
                # Record this processing event with detailed information
                processing_record = {
                    "processed_at": current_time,
                    "outputs": {
                        "valid": valid_filepath.name if not valid_df.empty else None,
                        "invalid": invalid_filepath.name if not invalid_df.empty else None
                    },
                    "counts": {
                        "valid": valid_count,
                        "invalid": invalid_count,
                        "total": len(df)
                    },
                    "status": "processed_successfully"
                }
                
                # Update the processing history
                checkpoint["processing_history"][file_path.name] = processing_record
                
                # Update the file_to_output_mapping for future reference
                if "file_to_output_mapping" not in checkpoint:
                    checkpoint["file_to_output_mapping"] = {}
                
                # Add the mapping for this file
                if not valid_df.empty:
                    checkpoint["file_to_output_mapping"][file_path.name] = valid_filepath.name
                elif not invalid_df.empty:
                    checkpoint["file_to_output_mapping"][file_path.name] = invalid_filepath.name
                
                # Complete transaction and save with atomic operations
                checkpoint["transaction_status"] = "complete"
                atomic_save_checkpoint(checkpoint)
                
                print(f"✅ Processed file {i+1}/{len(new_files)}: {file_path.name}")
                print(f"✅ Valid: {valid_count}, Invalid: {invalid_count}")
                print(f"✅ Running totals from checkpoint:")
                print(f"   - Processed files: {len(checkpoint['processed_files'])}")
                print(f"   - Valid files: {len(checkpoint['valid_files'])}")
                print(f"   - Invalid files: {len(checkpoint.get('invalid_files', []))}")
        
        except Exception as e:
            print(f"\n❌ ERROR: Failed to process new files: {e}")
            print("ℹ️ RECOMMENDED ACTION: Check files for corruption or permission issues")
            print("="*80 + "\n")
            # Continue with already processed files if possible
    else:
        print("\n[DEBUG-CLEAN] No new files to process. Using previously validated data.")
    
    # Now combine all valid data for threshold check
    all_valid_files = list(VALID_DIR.glob(f"{dataset_name}_{version}_valid_*.parquet"))
    print(f"\n[DEBUG-CLEAN] Found {len(all_valid_files)} total valid data files")
    
    # Report skipped files due to checkpoint mechanism
    skipped_files = [f for f in all_files if f.name in processed_files]
    if skipped_files:
        print(f"\n[DEBUG-CLEAN] ℹ️ CHECKPOINT STATUS: SKIPPED {len(skipped_files)} previously processed files due to checkpoint mechanism:")
        
        # Enhance the checkpoint information
        print(f"[DEBUG-CLEAN] ℹ️ CHECKPOINT DETAILS:")
        print(f"[DEBUG-CLEAN]   - Total original files: {len(all_files)}")
        print(f"[DEBUG-CLEAN]   - Previously processed (in checkpoint): {len(processed_files)}")
        print(f"[DEBUG-CLEAN]   - Newly detected (not in checkpoint): {len(new_files)}")
        print(f"[DEBUG-CLEAN]   - Skipped due to checkpoint: {len(skipped_files)}")
        
        # Show checkpoint contents summary
        print(f"[DEBUG-CLEAN] ℹ️ CHECKPOINT CONTENTS:")
        print(f"[DEBUG-CLEAN]   - Last processing timestamp: {checkpoint.get('last_processed_timestamp', 'Not recorded')}")
        print(f"[DEBUG-CLEAN]   - Previously valid files: {len(checkpoint.get('valid_files', []))}")
        
        # Calculate invalid files more accurately from filesystem rather than checkpoint
        # Count actual invalid files on disk
        invalid_dir = Path("./data/rejected")
        actual_invalid_files = []
        if invalid_dir.exists():
            actual_invalid_files = list(invalid_dir.glob(f"{dataset_name}_{version}_invalid_*.parquet"))
        
        # Compare with checkpoint
        checkpoint_invalid_count = len(checkpoint.get('invalid_files', []))
        print(f"[DEBUG-CLEAN]   - Previously invalid files (in checkpoint): {checkpoint_invalid_count}")
        print(f"[DEBUG-CLEAN]   - Actual invalid files (on disk): {len(actual_invalid_files)}")
        
        # Look for discrepancy
        if checkpoint_invalid_count != len(actual_invalid_files):
            print(f"[DEBUG-CLEAN]   ⚠️ Warning: Discrepancy between checkpoint ({checkpoint_invalid_count}) and actual invalid files ({len(actual_invalid_files)})")
            print(f"[DEBUG-CLEAN]   ⚠️ This may indicate the checkpoint is out of sync with the filesystem")
        
        # List specific skipped files
        print(f"[DEBUG-CLEAN] ℹ️ SKIPPED FILES (showing {min(5, len(skipped_files))} of {len(skipped_files)}):")
        for i, file in enumerate(skipped_files[:5]):  # Show first 5 skipped files 
            print(f"[DEBUG-CLEAN]   - Skipped: {file.name}")
        if len(skipped_files) > 5:
            print(f"[DEBUG-CLEAN]   - ... and {len(skipped_files) - 5} more files")
    
    if not all_valid_files:
        print("\n❌ DATA STATUS: No valid data files found")
        print("ℹ️ RECOMMENDED ACTION: Use the data collection interface to submit more data samples")
        print("="*80 + "\n")
        return pd.DataFrame(), False
    
    # Load all valid data files
    try:
        valid_dfs = []
        for file_path in all_valid_files:
            try:
                df = pd.read_parquet(file_path)
                valid_dfs.append(df)
                print(f"[DEBUG-CLEAN] Loaded valid file: {file_path.name} ({len(df)} data points)")
            except Exception as e:
                print(f"[DEBUG-CLEAN] ⚠️ Error loading {file_path.name}: {e}")
        
        if not valid_dfs:
            print("\n❌ DATA STATUS: Failed to load any valid data files")
            return pd.DataFrame(), False
        
        # Combine all valid data
        combined_valid_df = pd.concat(valid_dfs)
        print(f"\n✅ Successfully loaded {len(combined_valid_df)} valid data points from {len(valid_dfs)} files")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load valid data files: {e}")
        print("="*80 + "\n")
        return pd.DataFrame(), False
    
    # Calculate total valid data points (the ONLY thing that matters for threshold)
    valid_data_count = len(combined_valid_df)
    
    # Count invalid data points (only for reporting purposes)
    all_invalid_files = list(REJECTED_DIR.glob(f"{dataset_name}_{version}_invalid_*.parquet"))
    try:
        invalid_count = sum(len(pd.read_parquet(f)) for f in all_invalid_files)
    except Exception as count_err:
        print(f"[DEBUG] Invalid file counting error type: {type(count_err).__name__}")
        print(f"[DEBUG] Invalid file counting error details: {count_err}")
        # If we can't load invalid files, just count the number of files
        invalid_count = len(all_invalid_files)
    
    # Check if enough valid data points exist - threshold is ONLY based on valid data
    min_required = 10  # Minimum number of valid data points required to proceed with training
    dataset_ok = valid_data_count >= min_required
    
    # Make the threshold check highly visible in the logs
    print("\n" + "="*50)
    print("🚨 TRAINING THRESHOLD CHECK 🚨")
    print("="*50)
    print(f"Total validated (valid) data points: {valid_data_count}")
    print(f"Threshold: {min_required}")
    print(f"Result: {'✅ THRESHOLD MET' if dataset_ok else '❌ THRESHOLD NOT MET'}")
    print(f"Training: {'✅ WILL PROCEED' if dataset_ok else '❌ WILL NOT PROCEED'}")
    print("="*50)
    
    # Count files (each file = one data point in this application)
    all_original_files = list(DATA_DIR.glob(f"{dataset_name}_{version}_original_*.parquet"))
    original_file_count = len(all_original_files)
    
    # Count valid files
    valid_file_count = len(all_valid_files)
    
    # Count invalid files
    invalid_file_count = len(all_invalid_files)
    
    # Print detailed summary
    print("\n" + "="*50)
    print("DATASET CLEANING: Final Results")
    print("="*50)
    print("DATA SUMMARY:")
    print(f"  - Total collected files/data points: {original_file_count}")
    print(f"  - Valid files/data points: {valid_file_count}")
    print(f"  - Invalid files/data points: {invalid_file_count}")
    print(f"  - Note: In this application, each file contains exactly one data point")
    print(f"  - Data validation rate: {(valid_file_count/original_file_count*100) if original_file_count > 0 else 0:.1f}%")
    print("\nTRAINING STATUS:")
    print(f"  - Valid data points: {valid_data_count}")
    print(f"  - Required threshold: {min_required}")
    if not dataset_ok:
        print(f"  - ⚠️ Need {min_required - valid_data_count} more valid data points to reach threshold")
    
    if dataset_ok:
        print(f"\n✅ DATA STATUS: Sufficient data available for training")
        print(f"ℹ️ Valid data points: {len(combined_valid_df)} (minimum required: {min_required})")
        print("ℹ️ PIPELINE STATUS: Proceeding to training stage")
    else:
        print(f"\n❌ DATA STATUS: Insufficient data for training")
        print(f"ℹ️ Valid data points: {len(combined_valid_df)} (minimum required: {min_required})")
        print(f"ℹ️ RECOMMENDED ACTION: Collect at least {min_required - len(combined_valid_df)} more valid data samples")
        print("ℹ️ PIPELINE STATUS: Training stage will be skipped")
    
    print("="*80 + "\n")
    
    # Add completion timestamp for debugging
    print(f"⏱️ DATA CLEANING COMPLETED AT: {datetime.datetime.now()}")
    print(f"📊 FINAL STATUS: {'DATASET OK' if dataset_ok else 'INSUFFICIENT DATA'} - {len(combined_valid_df)} valid samples")
    
    # Create variable for tracking training file path
    training_file_path = ""
    
    # Save a consolidated file in the main data directory for the model training
    # but ONLY if we're proceeding with training to avoid redundant files
    if dataset_ok:
        # Create a training directory if it doesn't exist
        TRAINING_DIR = DATA_DIR / "training"
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load training cycle info to get current threshold
        TRAINING_CYCLE_FILE = Path("./training_cycle_info.json")
        base_threshold = 10  # Default minimum threshold
        max_data_ratio = 2.0  # Max allowed ratio of data points to threshold
        
        # Get current threshold from training cycle info
        try:
            if TRAINING_CYCLE_FILE.exists():
                with open(TRAINING_CYCLE_FILE, 'r') as f:
                    cycle_info = json.load(f)
                    cycle_count = cycle_info.get("cycle_count", 0)
                    # Calculate threshold based on cycle count (same formula as in main script)
                    current_threshold = (cycle_count + 1) * base_threshold
            else:
                current_threshold = base_threshold  # Default for first cycle
                
            # Check if we have too much data (2x or more than threshold)
            valid_data_count = len(combined_valid_df)
            max_allowed = current_threshold * max_data_ratio
            
            if valid_data_count > max_allowed:
                print("\n" + "="*80)
                print("🚨 SAFEGUARD: TOO MUCH TRAINING DATA DETECTED")
                print("="*80)
                print(f"⚠️ Current valid data points: {valid_data_count}")
                print(f"⚠️ Current threshold: {current_threshold}")
                print(f"⚠️ Maximum allowed: {max_allowed} (2x threshold)")
                print(f"⚠️ This exceeds the maximum allowed ratio of {max_data_ratio}x threshold")
                print(f"⚠️ This could lead to overfitting or other training instability")
                print("\nℹ️ RECOMMENDED ACTIONS:")
                print("  1. Use a smaller dataset (delete some data points)")
                print("  2. Or increase the cycle count to raise the threshold")
                print("  3. Or modify max_data_ratio in the code if you're sure this is safe")
                print("="*80 + "\n")
                
                # Don't create training file - this will prevent training from starting
                print(f"❌ Training file creation ABORTED to prevent potential issues")
                return combined_valid_df, False, ""  # Return empty path to prevent training
        except Exception as e:
            print(f"⚠️ Warning: Error checking max data threshold: {e}")
            print(f"ℹ️ Continuing with training file creation anyway")
            # Continue with training file creation despite the error
        
        # Check if we need to create a consolidated file
        # First, get the base threshold
        base_threshold = 10  # Default base threshold
        TRAINING_CYCLE_FILE = Path("./training_cycle_info.json")
        
        # Try to read training cycle info to check if threshold is met
        try:
            cycle_count = 0
            if TRAINING_CYCLE_FILE.exists():
                with open(TRAINING_CYCLE_FILE, 'r') as f:
                    try:
                        cycle_info = json.load(f)
                        cycle_count = cycle_info.get("cycle_count", 0)
                    except Exception as e:
                        print(f"⚠️ Warning: Error reading training cycle file: {e}")
        except Exception as e:
            print(f"⚠️ Warning: Error checking training cycle: {e}")
            
        # Calculate current threshold
        current_threshold = (cycle_count + 1) * base_threshold
        
        # Check if threshold is met for creating consolidated file
        if len(combined_valid_df) < current_threshold:
            print(f"⚠️ Threshold not yet met: {len(combined_valid_df)} valid points < {current_threshold} threshold")
            print(f"ℹ️ Will NOT create consolidated file until threshold is met")
            # Create training file only but don't consolidate
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            training_filename = f"{dataset_name}_{version}_training_{timestamp}.parquet"
            training_filepath = TRAINING_DIR / training_filename
            combined_valid_df.to_parquet(training_filepath)
            print(f"✅ Training copy saved to {training_filepath} for model training")
        else:
            # Threshold met - create both consolidated and training files
            print(f"✅ Threshold met: {len(combined_valid_df)} valid points >= {current_threshold} threshold")
            
            # Save consolidated file for general use ONLY when threshold is met
            consolidated_filename = f"{dataset_name}_{version}_consolidated_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            consolidated_filepath = DATA_DIR / consolidated_filename
            combined_valid_df.to_parquet(consolidated_filepath)
            print(f"✅ Consolidated valid data saved to {consolidated_filepath}")
            
            # Also save a copy with "training" prefix in the training directory
            # This is the file that will be used by the training process
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            training_filename = f"{dataset_name}_{version}_training_{timestamp}.parquet"
            training_filepath = TRAINING_DIR / training_filename
            combined_valid_df.to_parquet(training_filepath)
            print(f"✅ Training copy saved to {training_filepath} for model training")
        
        # Track the training file path as a ZenML artifact
        training_file_path = str(training_filepath)
        print(f"ℹ️ Training file path will be passed as ZenML artifact: {training_file_path}")
    else:
        print("ℹ️ Not creating consolidated/training files since threshold was not met")
    
    # Remove the cleaning in progress marker file - done by atexit handler now
    # We keep this as a final fallback but it should rarely be needed
    try:
        CLEANING_IN_PROGRESS_FILE = Path("./.cleaning_in_progress")
        if CLEANING_IN_PROGRESS_FILE.exists():
            try:
                # Read the marker file to check if it's from this process
                import json
                import os
                
                with open(CLEANING_IN_PROGRESS_FILE, 'r') as f:
                    marker_data = json.load(f)
                    if marker_data.get("pid") == os.getpid():
                        # This is our marker, we can safely remove it
                        CLEANING_IN_PROGRESS_FILE.unlink()
                        print("ℹ️ Removed our cleaning in progress marker at completion")
                    else:
                        # This is another process's marker, don't remove it
                        print(f"ℹ️ Found another cleaning process's marker (PID: {marker_data.get('pid')}), not removing")
            except Exception as marker_err:
                print(f"[DEBUG] Marker file reading error type: {type(marker_err).__name__}")
                print(f"[DEBUG] Marker file reading error details: {marker_err}")
                # If we can't read the marker data, just remove it if it's been there too long
                import time
                if time.time() - CLEANING_IN_PROGRESS_FILE.stat().st_mtime > 3600:  # 1 hour timeout
                    CLEANING_IN_PROGRESS_FILE.unlink()
                    print("ℹ️ Removed stale cleaning in progress marker (older than 1 hour)")
                else:
                    print("ℹ️ Found recent cleaning marker from unknown process, not removing")
    except Exception as e:
        print(f"⚠️ Warning: Error handling cleaning in progress marker: {e}")
    
    # Final checkpoint update with completion information
    try:
        # Save the final state of the checkpoint with process completion information
        checkpoint["cleaning_process_completed"] = True
        checkpoint["completion_time"] = int(time.time())
        checkpoint["transaction_status"] = "process_complete"
        checkpoint["valid_data_count"] = valid_data_count
        checkpoint["threshold_met"] = dataset_ok
        checkpoint["dataset_ok"] = dataset_ok
        
        # Log summary of the checkpoint state
        print(f"[DEBUG-CLEAN] 📊 Final checkpoint state:")
        print(f"[DEBUG-CLEAN]   - Total processed files: {len(checkpoint.get('processed_files', []))}")
        print(f"[DEBUG-CLEAN]   - Total valid files: {len(checkpoint.get('valid_files', []))}")
        print(f"[DEBUG-CLEAN]   - Total invalid files: {len(checkpoint.get('invalid_files', []))}")
        print(f"[DEBUG-CLEAN]   - Valid data count: {valid_data_count}")
        print(f"[DEBUG-CLEAN]   - Threshold met: {dataset_ok}")
        
        # Save with atomic operations
        atomic_save_checkpoint(checkpoint)
        print(f"[DEBUG-CLEAN] ✅ Final checkpoint saved successfully")
    except Exception as e:
        print(f"[DEBUG-CLEAN] ⚠️ Warning: Failed to save final checkpoint state: {e}")
    
    return combined_valid_df, dataset_ok, training_file_path