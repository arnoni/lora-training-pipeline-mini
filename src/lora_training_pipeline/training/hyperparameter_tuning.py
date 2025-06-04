# LoRA_Training_Pipeline/src/training/hyperparameter_tuning.py
# ----------------------------------------
# Hyperparameter tuning using Ray Tune and PyTorch Lightning.
# ----------------------------------------

# Import error handling tools first to avoid NameError
from src.lora_training_pipeline.utils.helpers import safe_run
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
# import pytorch_lightning as pl  # Removed
import lightning as L  # Changed to lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # Changed import
import os
import tempfile
from typing import Dict, Tuple, Any, Annotated
import pandas as pd
from zenml import step

# Use modern imports with type annotations
import mlflow
# Import MLflow tracking URI with proper error handling
try:
    from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
except ImportError:
    # Fallback implementation if the import fails
    def get_tracking_uri():
        """Fallback function to get MLflow tracking URI if ZenML import fails."""
        print("Warning: Could not import get_tracking_uri from ZenML, using fallback.")
        # Check for MLflow tracking URI in environment
        import os
        if "MLFLOW_TRACKING_URI" in os.environ:
            return os.environ["MLFLOW_TRACKING_URI"]
        # Use a local file-based store as default fallback
        from pathlib import Path
        mlruns_dir = Path(".zen/.mlruns").absolute()
        # Create directory if it doesn't exist
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        return f"file://{str(mlruns_dir)}"

from src.lora_training_pipeline.training.lora_finetune import LoraFinetuneModule, HfDataModule, llama_lora_config, get_device, prepare_data
from pathlib import Path
from src.lora_training_pipeline.config import BASE_MODEL_NAME


BEST_MODEL_PATH = Path("./output/best_model")

class TuneReportCallback(RayTrainReportCallback):
    """Custom callback to report metrics to Ray Tune."""
    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule): # Changed to L.Trainer
        metrics = self.get_metrics(trainer=trainer, pl_module=pl_module)
        if trainer.is_global_zero:
            metrics = self.get_metrics(trainer, pl_module)
            tune.report(**metrics)

def trainable_func(config: Dict, data_module:HfDataModule, lora_config, run_local:bool = True):
    """Trainable function for Ray Tune."""
    # Import progress tracking
    from src.lora_training_pipeline.utils.helpers import ProgressTracker
    
    # Create a progress tracker for this training run
    trial_progress = ProgressTracker(f"Training Trial")
    trial_progress.start(f"Starting training with lr={config.get('lr'):.1e}, batch_size={config.get('per_device_train_batch_size')}")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode="min",
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=config.get("checkpoint_dir", "checkpoints/"),
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    
    # Create a custom callback to track training progress
    class ProgressCallback(L.Callback):
        def __init__(self, progress_tracker):
            super().__init__()
            self.progress = progress_tracker
            
        def on_train_epoch_start(self, trainer, pl_module):
            self.progress.update(message=f"Starting epoch {trainer.current_epoch+1}/{trainer.max_epochs}")
            
        def on_train_epoch_end(self, trainer, pl_module):
            if hasattr(trainer, 'logged_metrics') and 'train_loss' in trainer.logged_metrics:
                loss = trainer.logged_metrics['train_loss']
                self.progress.update(message=f"Completed epoch {trainer.current_epoch+1}/{trainer.max_epochs} - train_loss: {loss:.4f}")
                
        def on_validation_start(self, trainer, pl_module):
            self.progress.update(message=f"Running validation after epoch {trainer.current_epoch+1}")
            
        def on_validation_end(self, trainer, pl_module):
            if hasattr(trainer, 'logged_metrics') and 'val_loss' in trainer.logged_metrics:
                loss = trainer.logged_metrics['val_loss']
                self.progress.update(message=f"Validation loss: {loss:.4f}")
    
    # Setup trainer with our custom callback
    progress_callback = ProgressCallback(trial_progress)
    
    trainer = L.Trainer( # Changed to L.Trainer
        devices="auto" if not run_local else 1,
        accelerator="auto",
        strategy="auto" if not run_local else "ddp_spawn",
        precision=16 if get_device(run_local) == "cuda" else 32,
        max_epochs=config.get("max_epochs", 10),
        callbacks=[early_stopping, model_checkpoint, TuneReportCallback(), progress_callback],
        enable_progress_bar=True,
        logger=L.loggers.MLFlowLogger(experiment_name="lora_finetuning", tracking_uri=get_tracking_uri()) if config.get("logging", True) else None, # Changed to L.loggers
        log_every_n_steps=1,
    )
    trainer = prepare_trainer(trainer)

    trial_progress.update(message="Initializing model")
    model = LoraFinetuneModule(BASE_MODEL_NAME, lora_config, config=config)
    
    # Start training timer
    trial_progress.start_timer("Training in progress", interval=5)
    trainer.fit(model, datamodule=data_module)
    
    # Update with test phase
    trial_progress.update(message="Running test evaluation")
    trainer.test(model, datamodule=data_module)
    
    # Save checkpoint
    trial_progress.update(message="Saving model checkpoint")
    with tempfile.TemporaryDirectory() as temp_dir:
        if trainer.is_global_zero:
            trainer.save_checkpoint(os.path.join(temp_dir, "best_model.ckpt"))
            checkpoint = Checkpoint.from_directory(temp_dir)
            tune.report(checkpoint=checkpoint)
    
    # Complete this trial
    trial_progress.complete(f"Training completed with final val_loss: {trainer.callback_metrics.get('val_loss', 0):.4f}")


# Remove duplicate import that was causing the NameError

# Path for saving the best model
BEST_MODEL_PATH = Path(os.path.join(".", "output", "best_model"))

# Check if MLflow tracker is available
def check_mlflow_tracker():
    """Check if MLflow tracker is registered with ZenML."""
    try:
        from zenml.client import Client
        client = Client()

        # First method: list_experiment_trackers
        try:
            if hasattr(client, 'list_experiment_trackers') and callable(client.list_experiment_trackers):
                trackers = client.list_experiment_trackers()
                tracker_names = [t.name for t in trackers]
                if "mlflow_tracker" in tracker_names:
                    print(f"Found MLflow tracker in experiment trackers: {tracker_names}")
                    return True
        except Exception as e1:
            print(f"Warning: Could not check MLflow tracker via list_experiment_trackers: {e1}")

        # Second method: active stack components
        try:
            if hasattr(client, 'active_stack') and client.active_stack:
                components = client.active_stack.components
                if "experiment_tracker" in components:
                    tracker = components["experiment_tracker"]
                    # Check if it's MLflow
                    if hasattr(tracker, 'type') and 'mlflow' in str(tracker.type).lower():
                        print(f"Found MLflow tracker in active stack components")
                        return True
        except Exception as e2:
            print(f"Warning: Could not check MLflow tracker via active_stack: {e2}")

        # Third method: try to get the tracking URI directly
        try:
            from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
            tracking_uri = get_tracking_uri()
            if tracking_uri:
                print(f"Found MLflow tracking URI: {tracking_uri}")
                return True
        except Exception as e3:
            print(f"Warning: Could not get MLflow tracking URI: {e3}")

        # If all checks failed, check for environment variable as last resort
        import os
        if "MLFLOW_TRACKING_URI" in os.environ:
            print(f"Found MLFLOW_TRACKING_URI in environment variables")
            return True

        # If stack is "mlflow-stack", assume MLflow is available
        try:
            if hasattr(client, 'active_stack') and client.active_stack and 'mlflow' in client.active_stack.name.lower():
                print(f"Active stack '{client.active_stack.name}' contains 'mlflow', assuming tracker is available")
                return True
        except Exception as stack_err:
            print(f"[DEBUG] Error checking active stack for MLflow: {type(stack_err).__name__}: {stack_err}")
            pass

        print("MLflow tracker not found or not properly configured")
        return False
    except Exception as e:
        print(f"Warning: Could not check MLflow tracker (ZenML client error): {e}")
        # Try to direct check for MLflow
        try:
            import mlflow
            # If we get here, MLflow is at least installed
            print("MLflow is installed but not properly configured with ZenML")
            # Check if we can create a local tracking location
            from pathlib import Path
            mlruns_dir = Path(".zen/.mlruns").absolute()
            mlruns_dir.mkdir(parents=True, exist_ok=True)
            # Set tracking URI to local dir
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{str(mlruns_dir)}"
            print(f"Set MLFLOW_TRACKING_URI to {os.environ['MLFLOW_TRACKING_URI']}")
            return True
        except ImportError:
            print("MLflow is not installed")
        return False

# Use MLflow tracker only if available, otherwise use a fallback
mlflow_available = check_mlflow_tracker()
# Try to safely use the experiment_tracker parameter
try:
    # Check if we can import Client to verify ZenML is fully working
    from zenml.client import Client
    step_decorator = step(enable_cache=True, experiment_tracker="mlflow_tracker") if mlflow_available else step(enable_cache=True)
except Exception as e:
    print(f"Warning: Error with ZenML client: {e}")
    # Use simpler decorator as a fallback
    step_decorator = step(enable_cache=True)

@step_decorator
@safe_run
def hyperparameter_tuning(
    training_file_path: str, 
    run_local: bool = True
) -> Tuple[
    Annotated[Dict, "best_config"],
    Annotated[str, "best_checkpoint_path"],
    Annotated[str, "best_model_path"]
]:
    """Performs hyperparameter tuning using Ray Tune."""
    # Import progress tracking
    from src.lora_training_pipeline.utils.helpers import ProgressTracker
    
    # Create master progress tracker for the entire process
    training_progress = ProgressTracker("Model Training Pipeline", total=5)
    training_progress.start("Starting hyperparameter tuning and model training")
    
    print("\n" + "="*80)
    print("PIPELINE STATUS: Starting hyperparameter tuning and model training")
    print("="*80)

    # Step 1: Data preparation
    training_progress.update(message="Loading data from training file")
    
    # Load the training data from file
    try:
        print(f"ℹ️ DATA STATUS: Loading training data from {training_file_path}")
        clean_data = pd.read_parquet(training_file_path)
        print(f"✅ Successfully loaded training data with {len(clean_data)} data points")
    except Exception as e:
        error_msg = f"Failed to load training data from {training_file_path}: {e}"
        print(f"❌ ERROR: {error_msg}")
        from src.lora_training_pipeline.utils.helpers import log_pending_error
        log_pending_error(error_msg)
        raise
    
    data_prep = ProgressTracker("Data Preparation")
    data_prep.start(f"Processing {len(clean_data)} data points")
    
    # Check for Hugging Face authentication if using a gated model
    if "meta-llama" in BASE_MODEL_NAME:
        import os
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            print("\n⚠️ WARNING: Using a gated model without HUGGING_FACE_HUB_TOKEN environment variable")
            print("You may encounter authentication issues when accessing the model.")
            print("Consider setting the HUGGING_FACE_HUB_TOKEN environment variable.")
        else:
            print(f"✅ HUGGING_FACE_HUB_TOKEN environment variable is set")
            
            # Try to use the token to log in
            try:
                from huggingface_hub import login
                login(token=hf_token)
                print(f"✅ Successfully logged in to Hugging Face Hub")
            except Exception as e:
                print(f"⚠️ Error during Hugging Face login: {e}")
                print("Continuing anyway, but you may encounter authentication issues.")
    
    try:
        data_module = HfDataModule(BASE_MODEL_NAME, clean_data)
        data_prep.update(message="Converting data to Hugging Face format")
        data_module.prepare_data()
        data_prep.update(message="Setting up data loaders")
        data_module.setup(stage="fit")
        data_prep.complete("Data preparation complete")
    except Exception as e:
        # Handle potential authentication errors
        from src.lora_training_pipeline.utils.helpers import log_pending_error
        error_msg = f"Error during data preparation: {e}"
        print(f"\n❌ ERROR: {error_msg}")
        log_pending_error(error_msg)
        
        if "authenticat" in str(e).lower() or "gated" in str(e).lower() or "access" in str(e).lower():
            print("\n" + "="*80)
            print("❌ AUTHENTICATION ERROR: Cannot access the gated model")
            print("ℹ️ This is likely because you don't have access to the meta-llama model")
            print("ℹ️ 1. Set your HUGGING_FACE_HUB_TOKEN in the environment")
            print("ℹ️ 2. Make sure you have access to the model on Hugging Face")
            print("ℹ️ 3. Or change BASE_MODEL_NAME in config.py to a non-gated model")
            print("="*80 + "\n")
        
        # Re-raise the exception to handle it in the step
        raise
    
    # Step 2: Configure hyperparameter search
    training_progress.update(message="Configuring hyperparameter search")
    config_progress = ProgressTracker("Search Configuration")
    config_progress.start("Setting up hyperparameter search space")
    
    # Adjust search space based on data size
    if len(clean_data) < 50:
        config_progress.update(message="Using reduced search space due to limited data")
        print("⚠️ Limited data available. Using reduced hyperparameter search to prevent overfitting.")
        search_space = {
            "lr": tune.loguniform(1e-5, 1e-4),
            "per_device_train_batch_size": tune.choice([2, 4]),
            "per_device_eval_batch_size": tune.choice([2, 4]),
            "warmup_steps": tune.choice([5, 10]),
            "max_epochs": tune.choice([3, 5]),
            "lora_r": 4,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "logging": True,
        }
        num_samples = 1
    else:
        config_progress.update(message="Using full hyperparameter search space")
        print("✅ Sufficient data available. Using full hyperparameter search space.")
        search_space = {
            "lr": tune.loguniform(1e-5, 1e-3),
            "per_device_train_batch_size": tune.choice([4, 8, 16]),
            "per_device_eval_batch_size": tune.choice([4, 8, 16]),
            "warmup_steps": tune.choice([5, 10, 20]),
            "max_epochs": tune.choice([5, 10, 15]),
            "lora_r": tune.choice([4, 8, 16]),
            "lora_alpha": tune.choice([16, 32, 64]),
            "lora_dropout": tune.uniform(0.05, 0.2),
            "logging": True,
        }
        num_samples = 1 if run_local else 3  # Reduced from 10 to be more reasonable
    
    config_progress.update(message="Setting up scheduler and reporter")
    scheduler = ASHAScheduler(
        max_t=15,
        grace_period=1,
        reduction_factor=2,
    )
    reporter = CLIReporter(
        parameter_columns=["lr", "per_device_train_batch_size", "warmup_steps", "max_epochs"],
        metric_columns=["val_loss", "training_iteration"],
        max_report_frequency=30,
    )

    # Initialize Ray
    config_progress.update(message="Initializing Ray for training")
    print(f"ℹ️ TRAINING STATUS: Initializing Ray {'in local mode' if run_local else 'for distributed training'}")
    if run_local:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(local_mode=True)
    
    device_info = f"{'CPU' if get_device(run_local) == 'cpu' else 'GPU'}"
    print(f"ℹ️ HARDWARE STATUS: Training on {device_info}")
    
    # Configure tuner
    config_progress.update(message=f"Creating tuner with {num_samples} trials")
    print(f"ℹ️ TUNING STATUS: Starting hyperparameter search with {num_samples} trials")
    
    # Import helper for logging errors
    from src.lora_training_pipeline.utils.helpers import log_pending_error
    
    try:
        tuner = tune.Tuner(
            tune.with_parameters(trainable_func, data_module=data_module, lora_config=llama_lora_config, run_local=run_local),
            tune_config=tune.TuneConfig(
                metric="val_loss",
                mode="min",
                scheduler=scheduler,
                num_samples=num_samples,
                search_space=search_space,
            ),
            run_config=ray.train.RunConfig(
                name="lora_tune",
                progress_reporter=reporter,
                checkpoint_config=ray.train.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute="val_loss",
                    checkpoint_score_order="min",
                )
            ),
        )
        config_progress.complete("Search configuration complete")
    except Exception as e:
        error_msg = f"Failed to configure hyperparameter tuning: {str(e)}"
        log_pending_error(error_msg)
        config_progress.complete(f"Search configuration failed: {str(e)}")
        raise

    # Step 3: Run hyperparameter search
    training_progress.update(message="Running hyperparameter search")
    tuning_progress = ProgressTracker("Hyperparameter Search")
    tuning_progress.start(f"Running {num_samples} trials")
    tuning_progress.start_timer("Hyperparameter tuning in progress", interval=5)
    
    print("ℹ️ TUNING STATUS: Hyperparameter search in progress (this may take a while)...")
    
    try:
        results = tuner.fit()
        tuning_progress.complete("Hyperparameter search completed")
    except Exception as e:
        error_msg = f"Hyperparameter tuning failed: {str(e)}"
        log_pending_error(error_msg)
        tuning_progress.complete(f"Hyperparameter search failed: {str(e)}")
        raise
    
    # Step 4: Process results and select best model
    training_progress.update(message="Processing tuning results")
    model_selection = ProgressTracker("Model Selection")
    model_selection.start("Identifying best performing model")
    
    try:
        # Get best result
        best_result = results.get_best_result(metric="val_loss", mode="min")
        best_config = best_result.config
        best_checkpoint = best_result.checkpoint
        best_checkpoint_path = os.path.join(best_checkpoint.path, "best_model.ckpt")
        
        model_selection.update(message=f"Found best model with val_loss: {best_result.metrics['val_loss']:.4f}")
        print(f"\n✅ TUNING STATUS: Hyperparameter search completed")
        print(f"ℹ️ Best validation loss: {best_result.metrics['val_loss']:.4f}")
    except Exception as e:
        error_msg = f"Failed to select best model from tuning results: {str(e)}"
        log_pending_error(error_msg)
        model_selection.complete(f"Model selection failed: {str(e)}")
        raise
    
    # Load and save best model
    model_selection.update(message="Loading best model from checkpoint")
    print("ℹ️ MODEL STATUS: Loading best model from checkpoint")
    
    try:
        best_model = LoraFinetuneModule.load_from_checkpoint(
            checkpoint_path=best_checkpoint_path, 
            model_name=BASE_MODEL_NAME, 
            lora_config=llama_lora_config, 
            config=best_config
        )
        model_selection.complete("Best model identified and loaded")
    except Exception as e:
        error_msg = f"Failed to load best model from checkpoint: {str(e)}"
        log_pending_error(error_msg)
        model_selection.complete(f"Model loading failed: {str(e)}")
        raise
    
    # Step 5: Save and log the results
    training_progress.update(message="Saving and logging results")
    save_progress = ProgressTracker("Model Saving")
    save_progress.start("Saving trained model")
    
    # Save model
    best_model_path = str(BEST_MODEL_PATH)
    print(f"ℹ️ MODEL STATUS: Saving best model to {best_model_path}")
    save_progress.start_timer("Saving model to disk")
    
    try:
        # Ensure the directory exists
        Path(best_model_path).mkdir(parents=True, exist_ok=True)
        
        # Save the PEFT model (LoRA adapter)
        best_model.model.save_pretrained(best_model_path)
        print(f"✅ Saved LoRA adapter to {best_model_path}")
        
        # Verify that the adapter files were created
        config_file = Path(best_model_path) / "adapter_config.json"
        weights_file = Path(best_model_path) / "adapter_model.safetensors"
        
        if config_file.exists():
            print(f"✅ LoRA config file created: {config_file}")
        else:
            print(f"⚠️ WARNING: LoRA config file missing: {config_file}")
            
        if weights_file.exists():
            print(f"✅ LoRA weights file created: {weights_file}")
        else:
            print(f"⚠️ WARNING: LoRA weights file missing: {weights_file}")
            # Try to save explicitly using different method
            try:
                best_model.model.save_adapter(best_model_path, "default")
                print(f"✅ Explicitly saved LoRA adapter using save_adapter method")
            except Exception as adapter_save_err:
                print(f"⚠️ Failed to save adapter explicitly: {adapter_save_err}")
        
        # Also save the tokenizer for completeness
        try:
            best_model.tokenizer.save_pretrained(best_model_path)
            print(f"✅ Saved tokenizer to {best_model_path}")
        except Exception as tokenizer_err:
            print(f"⚠️ Warning: Failed to save tokenizer: {tokenizer_err}")
        
        save_progress.update(message="Model saved, logging to MLflow")
    except Exception as e:
        error_msg = f"Failed to save model to disk: {str(e)}"
        log_pending_error(error_msg)
        save_progress.update(message=f"Model saving failed: {str(e)}")
        raise
    
    # Log hyperparameters and metrics
    print("ℹ️ LOGGING STATUS: Logging results to MLflow")
    print("ℹ️ Best hyperparameters:")
    for param, value in best_config.items():
        print(f"  - {param}: {value}")
    
    # Check ZenML connection before logging to MLflow
    from src.lora_training_pipeline.utils.helpers import check_zenml_connection
    print("\nVerifying ZenML/MLflow connection before logging results...")
    connection_ok = check_zenml_connection(max_retries=2, retry_delay=3)
    
    if not connection_ok:
        log_pending_error("ZenML/MLflow connection failed during model training")
        print("⚠️ WARNING: ZenML/MLflow connection issues detected")
        print("ℹ️ Will attempt to log results but this operation may fail")
        print("ℹ️ Model will still be saved locally even if logging fails")
    
    # Start MLflow logging with error handling
    mlflow_success = False
    try:
        # Only try to log to MLflow if the tracker is available
        if mlflow_available:
            print("ℹ️ Logging results to MLflow...")

            # Get tracking URI with robust error handling
            try:
                tracking_uri = get_tracking_uri()
                print(f"Using MLflow tracking URI: {tracking_uri}")
                mlflow.set_tracking_uri(tracking_uri)
            except Exception as uri_error:
                print(f"⚠️ Warning: Error getting tracking URI: {uri_error}")
                # Check if environment variable is set as fallback
                import os
                if "MLFLOW_TRACKING_URI" in os.environ:
                    print(f"Using MLFLOW_TRACKING_URI from environment: {os.environ['MLFLOW_TRACKING_URI']}")
                else:
                    # Set a default tracking URI as last resort
                    default_uri = f"file://{os.path.join(os.getcwd(), '.zen', '.mlruns')}"
                    print(f"Setting default tracking URI: {default_uri}")
                    mlflow.set_tracking_uri(default_uri)

            # Create experiment with fallback to default
            try:
                experiment_name = "lora_finetuning_final"
                print(f"Setting experiment name: {experiment_name}")
                mlflow.set_experiment(experiment_name)
            except Exception as exp_error:
                print(f"⚠️ Warning: Error setting experiment: {exp_error}")
                # Fall back to default experiment
                print("Using default experiment")

            # Start run with detailed error tracking
            try:
                with mlflow.start_run() as run:
                    print(f"Started MLflow run with ID: {run.info.run_id}")

                    # Log parameters with error handling
                    try:
                        # Filter config to ensure all values are serializable
                        filtered_config = {}
                        for key, value in best_config.items():
                            try:
                                # Convert to basic Python types if needed
                                if hasattr(value, 'item'):  # numpy or torch values
                                    filtered_config[key] = value.item()
                                else:
                                    filtered_config[key] = value
                            except Exception as serialize_err:
                                print(f"[DEBUG] Parameter serialization error for {key}: {type(serialize_err).__name__}: {serialize_err}")
                                # If a value can't be serialized, convert to string
                                filtered_config[key] = str(value)

                        print(f"Logging {len(filtered_config)} parameters to MLflow")
                        mlflow.log_params(filtered_config)
                    except Exception as param_error:
                        print(f"⚠️ Warning: Error logging parameters: {param_error}")

                    # Log metrics with error handling
                    try:
                        print(f"Logging metric best_val_loss: {best_result.metrics['val_loss']}")
                        mlflow.log_metric("best_val_loss", float(best_result.metrics["val_loss"]))
                    except Exception as metric_error:
                        print(f"⚠️ Warning: Error logging metrics: {metric_error}")

                    # Log model with error handling
                    try:
                        model_name = f"{BASE_MODEL_NAME}_lora_tuned"
                        print(f"Logging PyTorch model as {model_name}")
                        mlflow.pytorch.log_model(
                            best_model.model,
                            "model",
                            registered_model_name=model_name,
                        )
                    except Exception as model_error:
                        print(f"⚠️ Warning: Error logging model: {model_error}")

                    # Log artifacts with error handling
                    try:
                        print(f"Logging artifact from path: {best_model_path}")
                        mlflow.log_artifact(best_model_path)
                    except Exception as artifact_error:
                        print(f"⚠️ Warning: Error logging artifact: {artifact_error}")

                print("✅ Successfully completed MLflow logging")
                mlflow_success = True
            except Exception as run_error:
                print(f"⚠️ Warning: Error with MLflow run: {run_error}")
                log_pending_error(f"MLflow run error: {run_error}")
        else:
            print("ℹ️ MLflow tracker not available - skipping MLflow logging")
            print("ℹ️ Model has been saved locally and will still work")
            # Still consider this a success since we're handling the absence gracefully
            mlflow_success = True
    except Exception as e:
        error_msg = f"Failed to log results to MLflow: {str(e)}"
        log_pending_error(error_msg)
        print(f"❌ ERROR: {error_msg}")
        print("ℹ️ This error has been logged to pending_errors.txt")
        print("ℹ️ This is non-critical - model has been saved locally and will still work")
    
    if mlflow_success:
        save_progress.complete("Model saved and logged successfully")
    else:
        save_progress.complete("Model saved locally but MLflow logging failed")
    
    # Complete the overall training process
    training_progress.complete("Model training pipeline completed successfully")
    print("✅ MODEL STATUS: Model successfully trained and saved")
    print("="*80 + "\n")
    
    # Return the tuple with annotated types for modern ZenML
    return best_config, best_checkpoint_path, best_model_path