# LoRA Training Pipeline

A comprehensive pipeline for training and deploying LoRA-based language models.

## Features

- **Data Collection Interface**: Gradio UI for collecting training data
- **Data Cleaning and Validation**: Automated cleaning and validation of collected data
- **Automatic Training**: Pipeline triggers training when sufficient data is collected
- **Progressive Training Thresholds**: Increases data requirements for subsequent training cycles
- **Inference API**: FastAPI server for model inference
- **Inference UI**: Gradio UI for testing the trained model
- **Real-time Dashboard**: Web-based dashboard for monitoring pipeline status

## Getting Started

### Prerequisites

- Python 3.9 or later
- CUDA-compatible GPU (recommended)
- Ollama installed for validation

### Installation

1. Clone the repository
2. Setup the environment:
```bash
uv venv .venv
. .venv/bin/activate  # On Linux/macOS
# OR
.\.venv\Scripts\activate  # On Windows
uv pip install -e .
```

## Running the Pipeline

Run the complete pipeline (data collection, training, and inference):

```bash
python run_pipeline.py
```

This will start:
- Data Collection UI on port 7862
- Inference UI on port 7861
- FastAPI Inference Server on port 8001

## Dashboard

To monitor the pipeline, use the built-in dashboard:

```bash
# On Linux/macOS:
./dashboard_ui.sh

# On Windows:
dashboard_ui.bat
```

The dashboard provides real-time monitoring of:
- System resources (CPU, Memory, GPU)
- Data collection status
- Training cycle progress
- Model configuration
- Process status
- Error monitoring

Access the dashboard in your browser at http://localhost:7863.

## Individual Components

Run components separately:

- **Data Collection**: `python src/lora_training_pipeline/data_collection/gradio_app.py`
- **ZenML Pipeline**: `python src/lora_training_pipeline/training/zenml_pipeline.py`
- **Inference UI**: `python src/lora_training_pipeline/inference/gradio_inference.py`
- **Inference API**: `uvicorn src.lora_training_pipeline.inference.fastapi_inference:app --reload --port 8001`

## Project Structure

- `data/` - Training data storage
- `output/` - Trained models and checkpoints
- `src/lora_training_pipeline/` - Source code
  - `data_collection/` - Data collection interface
  - `data_cleaning/` - Data validation and cleaning
  - `training/` - LoRA fine-tuning code
  - `inference/` - Inference API and UI
  - `utils/` - Utility functions and dashboard

## License

This project is licensed under the MIT License - see the LICENSE file for details.