# LoRA_Training_Pipeline/src/training/lora_finetune.py

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import transformers
# import pytorch_lightning as pl  # Removed
import lightning as L  # Changed to lightning
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Dict
from src.lora_training_pipeline.config import BASE_MODEL_NAME  # Import from config
from pathlib import Path
from datasets import Dataset
# Using HuggingFace's train_test_split method instead of scikit-learn's
# from sklearn.model_selection import train_test_split
import GPUtil
import os
import pandas as pd
from zenml import step # ZenML imports
from typing import Annotated, Tuple

# --- Configuration ---
OUTPUT_DIR = "./output"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

llama_lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
llama_target_layers = [f"model.layers.{i}" for i in range(6)]


def get_device(run_local: bool) -> str:
    """
    Determines the device (CPU or GPU) to use for training.

    Args:
        run_local: Whether to run locally or on a remote cluster.

    Returns:
        "cuda" if a suitable GPU is available (and run_local is True, or if run_local
        is False and a GPU is available), otherwise "cpu".
    """
    if run_local:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for gpu in gpus:
                    if "4080" in gpu.name:
                        print(f"Using local NVIDIA RTX 4080 GPU: {gpu.name}")
                        return "cuda"
            print("Local NVIDIA RTX 4080 not found or no GPUs available. Falling back to CPU.")
            return "cpu"
        except Exception as e:
            print(f"Error detecting GPU: {e}. Falling back to CPU.")
            return "cpu"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"

def prepare_data(clean_data: pd.DataFrame):
    """
    Splits the cleaned data into training, validation, and test sets.

    Args:
        clean_data: A Pandas DataFrame containing the cleaned text data.

    Returns:
        A tuple containing the training, validation, and test datasets
        (as Hugging Face `Dataset` objects).
    """
    dataset = Dataset.from_pandas(clean_data)  # Convert DataFrame to Hugging Face Dataset
    train_dataset, temp_dataset = train_test_split(dataset, test_size=0.1, random_state=42)  # 90/10 split
    validation_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=42) # Split remaining data
    return train_dataset, validation_dataset, test_dataset

class LoraFinetuneModule(L.LightningModule): # Changed to L.LightningModule
    """
    PyTorch Lightning module for fine-tuning a language model with LoRA.

    This class encapsulates the model, optimizer, training/validation/test logic,
    and data loading.
    """
    def __init__(self, model_name: str, lora_config: LoraConfig, config: Dict):
        """
        Initializes the LoraFinetuneModule.

        Args:
            model_name: The name of the pre-trained model (e.g., "meta-llama/Llama-3.2-1B-Instruct").
            lora_config: The LoRA configuration object.
            config: A dictionary containing hyperparameters (e.g., learning rate, batch size).
        """
        super().__init__()
        self.model_name = model_name
        self.lora_config = lora_config
        self.config = config
        # Check if Hugging Face token is available
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        use_auth = False
        
        if hf_token:
            print("✅ Using HUGGING_FACE_HUB_TOKEN for model authentication")
            use_auth = True
        else:
            print("⚠️ WARNING: Using a gated model without HUGGING_FACE_HUB_TOKEN environment variable")
            print("You may encounter authentication issues when accessing the model.")
            print("Consider setting the HUGGING_FACE_HUB_TOKEN environment variable.")
            
        # Load tokenizer with token if available
        tokenizer_args = {"token": hf_token} if use_auth else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with token if available
        model_args = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if use_auth:
            model_args["token"] = hf_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_args
        )
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.lora_config)


    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            labels:  Labels (for training).

        Returns:
            Model outputs.
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        """
        Logic for a single training step.

        Args:
            batch: A batch of data.
            batch_idx: The index of the batch.

        Returns:
            The training loss.
        """
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Logic for a single validation step.

        Args:
            batch: A batch of data.
            batch_idx: The index of the batch.

        Returns:
            The validation loss.
        """
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Logic for a single test step.

        Args:
            batch: A batch of data.
            batch_idx: The index of the batch.

        Returns:
            The test loss.
        """
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = outputs.loss
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            A dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = AdamW(self.parameters(), lr=self.config.get("lr", 1e-4))  # Use AdamW optimizer
        scheduler = get_linear_schedule_with_warmup(  # Use a linear schedule with warmup
            optimizer,
            num_warmup_steps=self.config.get("warmup_steps", 5),
            num_training_steps=self.trainer.max_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def collate_fn(self, batch):
        """
        Collates a batch of samples into a format suitable for the model.  Handles padding.

        Args:
            batch: A list of samples.

        Returns:
            A dictionary containing the padded input_ids, attention_mask, and labels.
        """
        input_ids = [item["input_ids"].squeeze() for item in batch]  # Extract input IDs
        attention_mask = [item["attention_mask"].squeeze() for item in batch]  # Extract attention masks
        labels = [item["input_ids"].squeeze().clone() for item in batch]  # Labels are the same as input IDs

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def train_dataloader(self):
        """
        Creates the training data loader.

        Returns:
            A PyTorch DataLoader for the training set.
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.get("per_device_train_batch_size", 8),
            shuffle=True,
            collate_fn=self.collate_fn,
            )

    def val_dataloader(self):
        """
        Creates the validation data loader.

        Returns:
            A PyTorch DataLoader for the validation set.
        """
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.config.get("per_device_eval_batch_size", 8),
            collate_fn=self.collate_fn,
            )
            
    def test_dataloader(self):
        """
        Creates the test data loader.

        Returns:
            A PyTorch DataLoader for the test set.
        """
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.get("per_device_eval_batch_size", 8),
            collate_fn = self.collate_fn,
            )


    def setup(self, stage=None):
        """
        Setup function called by PyTorch Lightning.  Used here to assign datasets.

        Args:
            stage:  Either 'fit' (for training/validation) or 'test'.
        """
        # Assign the datasets within setup, so they are available for the dataloaders.
        if stage == 'fit' or stage is None:
            self.train_dataset = self.trainer.datamodule.train_dataset
            self.validation_dataset = self.trainer.datamodule.validation_dataset
        if stage == 'test' or stage is None:
            self.test_dataset = self.trainer.datamodule.test_dataset

class HfDataModule(L.LightningDataModule): # Changed to L.LightningDataModule
    """
    PyTorch Lightning DataModule for handling the Hugging Face dataset.

    This class handles loading, preprocessing, and splitting the data.
    """
    def __init__(self, model_name: str, clean_data: pd.DataFrame):
        """
        Initializes the HfDataModule.

        Args:
            model_name: The name of the pre-trained model.
            clean_data:  The cleaned data (as a Pandas DataFrame).
        """
        super().__init__()
        self.model_name = model_name
        self.clean_data = clean_data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None


    def prepare_data(self):
        """
        Downloads and prepares the data.  This is called only once, on the main process.
        """
        # Convert the entire cleaned pandas DataFrame into a single unified Hugging Face Dataset
        # All data points are processed as a single dataset, not individually
        dataset = Dataset.from_pandas(self.clean_data)
        print(f"Created unified dataset from {len(self.clean_data)} cleaned data points")
        
        # Use HuggingFace's train_test_split method instead of scikit-learn's
        # First split dataset into 90% train, 10% temp
        splits = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = splits['train']
        temp_dataset = splits['test']
        
        # Split the temp dataset into validation and test (50% each of the 10%)
        temp_splits = temp_dataset.train_test_split(test_size=0.5, seed=42)
        validation_dataset = temp_splits['train']
        test_dataset = temp_splits['test']
        
        # Store the datasets
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        
        # Print some info about the splits for debugging
        print(f"Dataset splits - Train: {len(train_dataset)}, Validation: {len(validation_dataset)}, Test: {len(test_dataset)}")

    def setup(self, stage=None):
        """
        Sets up the datasets for the different stages (fit/test).  This is called on every process.

        Args:
            stage: Either 'fit' (for training/validation) or 'test'.
        """
        # Tokenize the datasets.  This adds the 'input_ids', 'attention_mask', etc., fields.
        if stage == "fit" or stage is None:
            self.train_dataset = self.train_dataset.map(lambda examples: self.tokenizer(examples["text"], truncation=True, max_length=512), batched=True)
            self.validation_dataset = self.validation_dataset.map(lambda examples: self.tokenizer(examples["text"], truncation=True, max_length=512), batched=True)

        if stage == "test" or stage is None:
            self.test_dataset = self.test_dataset.map(lambda examples: self.tokenizer(examples["text"], truncation=True, max_length=512), batched=True)

    def train_dataloader(self):
        """Creates the training data loader."""
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=8, shuffle=True)

    def val_dataloader(self):
        """Creates the validation data loader."""
        return torch.utils.data.DataLoader(self.validation_dataset, batch_size=8)

    def test_dataloader(self):
        """Creates the test data loader."""
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=8)