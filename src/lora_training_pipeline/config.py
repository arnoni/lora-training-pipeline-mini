# LoRA_Training_Pipeline/src/config.py

# --- Base Model Configuration ---
# Using Llama-3.2-1B-Instruct model (requires authentication)
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # Requires Hugging Face authentication

# --- Ollama Model Configuration ---
# Using the models that are actually available on your system
OLLAMA_MODEL_NAME = "gemma3:4b"  # Primary model for validation
# Fallback models in order of preference - updated based on your actual available models
OLLAMA_FALLBACK_MODELS = ["gemma3:4b", "gemma3:1b", "deepseek-r1:14b", "deepseek-r1:1.5b", "phi4", "qwen2.5:1.5b", "mistral-small"]

# --- Data Configuration ---
DATASET_NAME = "user_data_ui"
DATA_VERSION = "v1"