"""
Fine-tuning Module for Gaelic Football Match Analyser

This module handles:
1. Fine-tuning pre-trained LLMs for football/Gaelic football analysis
2. Memory-efficient training with LoRA/QLoRA
3. Evaluation of fine-tuned models
4. Saving and loading fine-tuned models
"""

import logging
import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time
import datetime
import shutil

logger = logging.getLogger(__name__)

class FineTuner:
    """
    Fine-tune LLM models for football/Gaelic football analysis.
    """
    
    def __init__(self, 
                 model,
                 output_dir: str = "models/fine_tuned",
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 learning_rate: float = 2e-4,
                 batch_size: int = 8,
                 epochs: int = 3,
                 max_steps: Optional[int] = None,
                 use_wandb: bool = False,
                 evaluation_strategy: str = "steps",
                 eval_steps: int = 200,
                 save_strategy: str = "steps",
                 save_steps: int = 500,
                 max_seq_length: int = 512):
        """
        Initialize the fine-tuning module.
        
        Args:
            model: Pre-trained model to fine-tune (from ModelSelector)
            output_dir: Directory to save fine-tuned model
            lora_rank: Rank for LoRA adapters
            lora_alpha: Alpha for LoRA adapters
            lora_dropout: Dropout probability for LoRA adapters
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            epochs: Number of training epochs
            max_steps: Maximum number of training steps (overrides epochs if provided)
            use_wandb: Whether to use Weights & Biases for tracking
            evaluation_strategy: When to evaluate during training ('steps' or 'epoch')
            eval_steps: Number of steps between evaluations when using 'steps' strategy
            save_strategy: When to save checkpoints during training ('steps' or 'epoch')
            save_steps: Number of steps between saving when using 'steps' strategy
            max_seq_length: Maximum sequence length for training
        """
        self.model = model
        self.output_dir = output_dir
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_steps = max_steps
        self.use_wandb = use_wandb
        self.evaluation_strategy = evaluation_strategy
        self.eval_steps = eval_steps
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        
        # Import libraries here to avoid loading them if not needed
        self.peft = None
        self.transformers = None
        self.datasets = None
        self.trainer = None
        
    def _import_libraries(self):
        """Import necessary libraries for fine-tuning."""
        try:
            logger.info("Importing fine-tuning libraries")
            import transformers
            import peft
            import datasets
            
            self.transformers = transformers
            self.peft = peft
            self.datasets = datasets
            
            # Set up transformers logging
            self.transformers.logging.set_verbosity_info()
            
        except ImportError as e:
            logger.error(f"Failed to import fine-tuning libraries: {e}")
            logger.error("Please install with: pip install transformers peft datasets")
            raise
            
        # Try to import wandb if requested
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                logger.info("Using Weights & Biases for experiment tracking")
            except ImportError:
                logger.warning("wandb not available. Install with: pip install wandb")
                self.use_wandb = False
    
    def _prepare_dataset(self, data: Dict[str, Any]) -> Any:
        """
        Prepare dataset for fine-tuning.
        
        Args:
            data: Transformed data containing training examples
            
        Returns:
            Dataset object ready for training
        """
        # Extract training examples from the data
        if "training_examples" not in data:
            raise ValueError("Data does not contain training examples")
        
        training_examples = data["training_examples"]
        logger.info(f"Preparing dataset with {len(training_examples)} examples")
        
        # Convert to the format expected by the Hugging Face datasets library
        formatted_examples = []
        for example in training_examples:
            formatted_examples.append({
                "text": f"### Instruction: {example['input']}\n\n### Response: {example['output']}"
            })
        
        # Create dataset
        dataset = self.datasets.Dataset.from_list(formatted_examples)
        
        # Split into training and validation sets (90/10 split)
        splits = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = splits["train"]
        eval_dataset = splits["test"]
        
        logger.info(f"Dataset split: {len(train_dataset)} training examples, {len(eval_dataset)} validation examples")
        
        return {
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset
        }
    
    def _prepare_trainer(self, tokenizer, train_dataset, eval_dataset):
        """
        Prepare the trainer for fine-tuning.
        
        Args:
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            
        Returns:
            Configured trainer
        """
        # Define data collator
        data_collator = self.transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We're not doing masked language modeling
        )
        
        # Define training arguments
        training_args = self.transformers.TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            max_steps=self.max_steps,
            weight_decay=0.01,
            evaluation_strategy=self.evaluation_strategy,
            eval_steps=self.eval_steps if self.evaluation_strategy == "steps" else None,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps if self.save_strategy == "steps" else None,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=10,
            report_to="wandb" if self.use_wandb else "none",
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            optim="adamw_torch"
        )
        
        # Define trainer
        trainer = self.transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        return trainer
    
    def _apply_lora(self, device):
        """
        Apply LoRA adapters to the model for memory-efficient fine-tuning.
        
        Args:
            device: Device to use for fine-tuning
            
        Returns:
            Model with LoRA adapters
        """
        logger.info(f"Applying LoRA adapters with rank={self.lora_rank}, alpha={self.lora_alpha}")
        
        # Configure LoRA
        lora_config = self.peft.LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention modules
        )
        
        # Apply LoRA to model
        model = self.peft.get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self._print_trainable_parameters(model)
        
        return model
    
    def _print_trainable_parameters(self, model):
        """
        Print the number of trainable parameters in the model.
        
        Args:
            model: Model to analyze
        """
        trainable_params = 0
        all_params = 0
        
        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / all_params:.2%} of {all_params:,} total)")
    
    def fine_tune(self, data: Dict[str, Any]) -> Any:
        """
        Fine-tune the model on football/Gaelic football data.
        
        Args:
            data: Transformed data containing training examples
            
        Returns:
            Fine-tuned model
        """
        self._import_libraries()
        
        logger.info("Starting fine-tuning process")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get the tokenizer from the model's attributes
        if hasattr(self.model, "tokenizer"):
            tokenizer = self.model.tokenizer
        else:
            # If tokenizer not attached to model, try to get it from the model's config
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path)
            except Exception as e:
                logger.error(f"Failed to get tokenizer: {e}")
                raise ValueError("Tokenizer not found. Please provide a model with an attached tokenizer.")
        
        # Prepare dataset
        datasets = self._prepare_dataset(data)
        train_dataset = datasets["train_dataset"]
        eval_dataset = datasets["eval_dataset"]
        
        # Get device
        device = self.model.device if hasattr(self.model, "device") else "cuda" if torch.cuda.is_available() else "cpu"
        
        # Apply LoRA for memory-efficient fine-tuning
        lora_model = self._apply_lora(device)
        
        # Save config for reference
        config_path = os.path.join(self.output_dir, "fine_tuning_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "max_steps": self.max_steps,
                "max_seq_length": self.max_seq_length,
                "date": datetime.datetime.now().isoformat()
            }, f, indent=2)
        
        # Initialize wandb if using it
        if self.use_wandb:
            self.wandb.init(
                project="gaelic-football-llm",
                name=f"fine-tune-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}",
                config={
                    "lora_rank": self.lora_rank,
                    "lora_alpha": self.lora_alpha,
                    "lora_dropout": self.lora_dropout,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "max_steps": self.max_steps,
                }
            )
        
        # Prepare trainer
        trainer = self._prepare_trainer(tokenizer, train_dataset, eval_dataset)
        
        # Run fine-tuning
        logger.info("Starting training")
        start_time = time.time()
        trainer.train()
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds")
        
        # Save the fine-tuned model
        logger.info(f"Saving fine-tuned model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        
        # Save adapter separately for easier loading
        self.peft.save_pretrained(lora_model, self.output_dir)
        
        # Run evaluation
        logger.info("Evaluating fine-tuned model")
        eval_results = trainer.evaluate()
        
        # Save evaluation results
        eval_path = os.path.join(self.output_dir, "eval_results.json")
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"Evaluation results: {eval_results}")
        
        # Close wandb if using it
        if self.use_wandb:
            self.wandb.finish()
        
        return lora_model
    
    def merge_and_save(self, model_path: Optional[str] = None) -> str:
        """
        Merge LoRA weights with base model and save the result.
        
        Args:
            model_path: Path to save the merged model (defaults to output_dir/merged)
            
        Returns:
            Path to the saved merged model
        """
        if self.peft is None:
            self._import_libraries()
        
        if model_path is None:
            model_path = os.path.join(self.output_dir, "merged")
        
        os.makedirs(model_path, exist_ok=True)
        
        logger.info(f"Merging LoRA weights and saving to {model_path}")
        
        # Load the fine-tuned model with adapters
        adapter_path = self.output_dir
        
        # Merge weights
        try:
            merged_model = self.peft.PeftModel.from_pretrained(self.model, adapter_path)
            merged_model = merged_model.merge_and_unload()
            
            # Save the merged model
            merged_model.save_pretrained(model_path)
            
            # Save tokenizer if available
            if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
                self.model.tokenizer.save_pretrained(model_path)
                
            logger.info(f"Successfully merged and saved model to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to merge and save model: {e}")
            raise
    
    def evaluate_custom(self, 
                       test_data: List[Dict[str, Any]], 
                       model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run custom evaluation on test data.
        
        Args:
            test_data: List of test examples
            model_path: Path to model to evaluate (defaults to fine-tuned model)
            
        Returns:
            Evaluation results
        """
        if self.transformers is None:
            self._import_libraries()
        
        # Load model
        if model_path:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            model = self.model
            # Get tokenizer
            if hasattr(model, "tokenizer"):
                tokenizer = model.tokenizer
            else:
                # If tokenizer not attached to model, try to get it from the model's config
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
                except Exception as e:
                    logger.error(f"Failed to get tokenizer: {e}")
                    raise ValueError("Tokenizer not found. Please provide a model with an attached tokenizer.")
        
        # Set model to evaluation mode
        model.eval()
        
        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        results = {
            "examples": [],
            "metrics": {}
        }
        
        # Evaluate each example
        for i, example in enumerate(test_data):
            logger.info(f"Evaluating example {i+1}/{len(test_data)}")
            
            input_text = f"### Instruction: {example['input']}\n\n### Response:"
            
            # Tokenize
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            
            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # Decode
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract just the response part
            response = output_text.split("### Response:")[-1].strip()
            
            # Calculate simple similarity metric (this would be more sophisticated in a real implementation)
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, response, example["output"]).ratio()
            
            # Save results
            results["examples"].append({
                "input": example["input"],
                "expected": example["output"],
                "generated": response,
                "similarity": similarity
            })
        
        # Calculate aggregate metrics
        similarities = [example["similarity"] for example in results["examples"]]
        results["metrics"]["avg_similarity"] = sum(similarities) / len(similarities)
        results["metrics"]["min_similarity"] = min(similarities)
        results["metrics"]["max_similarity"] = max(similarities)
        
        # Save results
        results_path = os.path.join(self.output_dir, "custom_eval_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Custom evaluation complete. Results saved to {results_path}")
        logger.info(f"Average similarity: {results['metrics']['avg_similarity']:.4f}")
        
        return results
