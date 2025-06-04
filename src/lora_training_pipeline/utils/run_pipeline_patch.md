# Patch Instructions for run_pipeline.py

This document provides instructions for patching `run_pipeline.py` to use the parallel service startup capability.

## Overview

The parallel startup integration allows all pipeline services (FastAPI server, Gradio UIs, Dashboard) to start in parallel while respecting their dependencies. This reduces the overall startup time and ensures proper resource allocation.

## Changes to Make

### 1. Add Import Statement

Near the top of the file, where other utility imports are located (around line 7-14), add:

```python
# Import parallel execution module
try:
    from src.lora_training_pipeline.utils.run_pipeline_integration import (
        run_pipeline_with_parallel_execution,
        register_cleanup_handlers,
        PARALLEL_STARTUP_AVAILABLE
    )
    print("✅ Parallel pipeline execution module loaded successfully")
except ImportError:
    PARALLEL_STARTUP_AVAILABLE = False
    print("⚠️ Parallel pipeline execution module not available - using sequential startup")
```

### 2. Modify the Main Function

In the `if __name__ == "__main__":` section (around line 5056), replace:

```python
if __name__ == "__main__":
    try:
        # Process any queued signals before starting
        if not signal_queue.empty():
            process_signal()

        if "--delete-datasets" in sys.argv: # Special flag for dataset deletion
            delete_all_datasets()
        else:
            # Normal execution - use enhanced process management if available
            if ENHANCED_PROCESS_MANAGEMENT:
                print("\n" + "="*80)
                print("STARTING PIPELINE WITH ENHANCED PROCESS MANAGEMENT")
                print("="*80)
                # Run the enhanced version that prevents duplicate processes
                data_collection_process, inference_ui_process, fastapi_process = run_pipeline_with_enhanced_process_management()
                # Add the processes to the global processes list for cleanup on exit
                if data_collection_process:
                    processes.append(data_collection_process)
                if inference_ui_process:
                    processes.append(inference_ui_process)
                if fastapi_process:
                    processes.append(fastapi_process)

                # Continue with the rest of the script, which includes monitoring and other processes
                main()

                # Main loop to process signals and keep the program running
                while True:
                    # Process any queued signals
                    if process_signal():
                        break  # Exit if a signal was processed (it will call exit(0))
                    time.sleep(1)  # Sleep to prevent CPU usage
            else:
                # Run the standard version if enhanced process management is not available
                print("\n" + "="*80)
                print("STARTING PIPELINE WITH STANDARD PROCESS MANAGEMENT")
                print("="*80)
                main()

                # Main loop to process signals and keep the program running
                while True:
                    # Process any queued signals
                    if process_signal():
                        break  # Exit if a signal was processed (it will call exit(0))
                    time.sleep(1)  # Sleep to prevent CPU usage
```

With:

```python
if __name__ == "__main__":
    try:
        # Process any queued signals before starting
        if not signal_queue.empty():
            process_signal()

        if "--delete-datasets" in sys.argv: # Special flag for dataset deletion
            delete_all_datasets()
        else:
            # Normal execution - use parallel startup if available, fall back to enhanced or standard
            if PARALLEL_STARTUP_AVAILABLE:
                print("\n" + "="*80)
                print("STARTING PIPELINE WITH PARALLEL SERVICE EXECUTION")
                print("="*80)
                
                # Register cleanup handlers for parallel execution
                register_cleanup_handlers()
                
                # Start services in parallel with proper dependency management
                data_collection_process, inference_ui_process, fastapi_process = run_pipeline_with_parallel_execution(
                    data_collection_port=DATA_COLLECTION_PORT,
                    inference_ui_port=INFERENCE_UI_PORT,
                    fastapi_inference_port=FASTAPI_INFERENCE_PORT,
                    dashboard_port=DASHBOARD_PORT,
                    session_id=SESSION_ID
                )
                
                # Add the processes to the global processes list for compatibility
                if data_collection_process:
                    processes.append(data_collection_process)
                if inference_ui_process:
                    processes.append(inference_ui_process)
                if fastapi_process:
                    processes.append(fastapi_process)

                # Continue with the rest of the script, which includes monitoring and other processes
                # Skip the service startup parts of main() since they're already handled
                try:
                    # Extract non-startup parts from main function for processing
                    check_for_stale_processes()
                    write_session_info()
                    
                    # Handle other initialization but skip service startup
                    # ...
                    
                    print("\n" + "="*80)
                    print("PARALLEL SERVICE STARTUP COMPLETED")
                    print("="*80)
                    print("Services are running in parallel mode.")
                    print(f"Dashboard URL: http://localhost:{DASHBOARD_PORT}")
                    print(f"Data Collection UI URL: http://localhost:{DATA_COLLECTION_PORT}")
                    print(f"Inference UI URL: http://localhost:{INFERENCE_UI_PORT}")
                    print(f"FastAPI Inference Server URL: http://localhost:{FASTAPI_INFERENCE_PORT}")
                    print("="*80 + "\n")
                except Exception as e:
                    print(f"Error in initialization: {e}")
                    import traceback
                    traceback.print_exc()

                # Main loop to process signals and keep the program running
                while True:
                    # Process any queued signals
                    if process_signal():
                        break  # Exit if a signal was processed (it will call exit(0))
                    time.sleep(1)  # Sleep to prevent CPU usage
            elif ENHANCED_PROCESS_MANAGEMENT:
                print("\n" + "="*80)
                print("STARTING PIPELINE WITH ENHANCED PROCESS MANAGEMENT")
                print("="*80)
                # Run the enhanced version that prevents duplicate processes
                data_collection_process, inference_ui_process, fastapi_process = run_pipeline_with_enhanced_process_management()
                # Add the processes to the global processes list for cleanup on exit
                if data_collection_process:
                    processes.append(data_collection_process)
                if inference_ui_process:
                    processes.append(inference_ui_process)
                if fastapi_process:
                    processes.append(fastapi_process)

                # Continue with the rest of the script, which includes monitoring and other processes
                main()

                # Main loop to process signals and keep the program running
                while True:
                    # Process any queued signals
                    if process_signal():
                        break  # Exit if a signal was processed (it will call exit(0))
                    time.sleep(1)  # Sleep to prevent CPU usage
            else:
                # Run the standard version if neither parallel nor enhanced process management is available
                print("\n" + "="*80)
                print("STARTING PIPELINE WITH STANDARD PROCESS MANAGEMENT")
                print("="*80)
                main()

                # Main loop to process signals and keep the program running
                while True:
                    # Process any queued signals
                    if process_signal():
                        break  # Exit if a signal was processed (it will call exit(0))
                    time.sleep(1)  # Sleep to prevent CPU usage
```

## Testing the Changes

1. Make sure the parallel startup module is installed by running:
   ```
   uv venv .venv && . .venv/bin/activate && uv pip install -e .
   ```

2. Run the pipeline with the new parallel startup:
   ```
   python run_pipeline.py
   ```

3. Check the logs for any issues. The parallel startup module logs to `parallel_pipeline.log`.

## Reverting to Sequential Startup

If you need to use the original sequential startup, you can temporarily disable the parallel startup by setting:

```python
PARALLEL_STARTUP_AVAILABLE = False  # Force sequential startup
```

Near the import statements at the top of the file.

## Performance Impact

The parallel startup mode can significantly reduce the overall startup time of the pipeline, especially on systems with multiple CPU cores. Service dependencies are still respected, ensuring that services start in the correct order when needed.