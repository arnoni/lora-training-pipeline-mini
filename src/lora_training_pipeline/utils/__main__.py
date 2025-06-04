# This file allows running the dashboard module directly with python -m
import sys
import os

if __name__ == "__main__":
    # Add the project root to PATH if needed
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        
    # Import and run the dashboard
    from src.lora_training_pipeline.utils.dashboard import run_dashboard
    run_dashboard()