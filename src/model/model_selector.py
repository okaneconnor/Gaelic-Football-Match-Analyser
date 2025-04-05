"""
Model Selector Module for Gaelic Football Match Analyser

This module handles:
1. Selection of appropriate LLM model based on requirements
2. Loading and initialization of models
3. Comparison of model performance for sports analysis tasks
"""

import logging
import os
import json
import torch
from typing import Dict, List, Any, Optional, Tuple
import time
import psutil
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelSelector:
    """
    Select and load appropriate LLM models for football/Gaelic football analysis.
    """
    
    # Model configuration information
    MODEL_CONFIGS = {
        "llama-2-7b": {
            "huggingface_id": "meta-llama/Llama-2-7b-hf",
            "tokenizer": "meta-llama/Llama-2-7b-hf",
            "description": "Meta's powerful open-source LLM with 7B parameters",
            "memory_req": "14GB",
            "strengths": ["Robust reasoning", "Good context understanding", "Strong overall performance"],
            "local_paths": ["models/llama-2-7b"]
        },
        "mistral-7b": {
            "huggingface_id": "mistralai/Mistral-7B-v0.1",
            "tokenizer": "mistralai/Mistral-7B-v0.1",
            "description": "High-performance model with strong reasoning capabilities",
            "memory_req": "14GB",
            "strengths": ["Excellent reasoning", "Good at detailed tasks", "Efficient architecture"],
            "local_paths": ["models/mistral-7b"]
        },
        "phi-2": {
            "huggingface_id": "microsoft/phi-2",
            "tokenizer": "microsoft/phi-2",
            "description": "Microsoft's compact but capable model",
            "memory_req": "5GB",
            "strengths": ["Compact size", "Good performance/size ratio", "Lower resource requirements"],
            "local_paths": ["models/phi-2"]
        },
        "gpt-j-6b": {
            "huggingface_id": "EleutherAI/gpt-j-6b",
            "tokenizer": "EleutherAI/gpt-j-6b",
            "description": "EleutherAI's generation-focused model",
            "memory_req": "12GB",
            "strengths": ["Good text generation", "Widely tested", "Well-documented"],
            "local_paths": ["models/gpt-j-6b"]
        }
    }
    
    def __init__(self, 
                 model_name: str = "mistral-7b",
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 quantization: str = "4bit",
                 benchmark: bool = False):
        """
        Initialize the model selector.
        
        Args:
            model_name: Name of the model to use
            model_path: Optional path to a custom model
            device: Device to load the model on ('cpu', 'cuda', 'mps', or None for auto-detect)
            quantization: Quantization level ('4bit', '8bit', 'none')
            benchmark: Whether to run benchmarks on model loading
        """
        self.model_name = model_name
        self.model_path = model_path
        self.quantization = quantization
        self.benchmark = benchmark
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Validate model selection
        if model_name not in self.MODEL_CONFIGS and not model_path:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.MODEL_CONFIGS.keys())}")
        
        # Set up import of libraries here to avoid loading them if not needed
        self.transformers = None
        self.model = None
        self.tokenizer = None
        
    def _import_libraries(self):
        """Import necessary libraries for model loading."""
        try:
            logger.info("Importing transformers library")
            import transformers
            self.transformers = transformers
        except ImportError:
            logger.error("Failed to import transformers. Please install with: pip install transformers")
            raise
            
        # Set up transformers for best performance
        if self.device == "cuda":
            try:
                # Try to import optimizations for faster inference
                import bitsandbytes
                logger.info("Using bitsandbytes for quantization")
            except ImportError:
                logger.warning("bitsandbytes not available. Install with: pip install bitsandbytes")
            
            try:
                import accelerate
                logger.info("Using accelerate for optimized model loading")
            except ImportError:
                logger.warning("accelerate not available. Install with: pip install accelerate")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get information about available models.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        for name, config in self.MODEL_CONFIGS.items():
            # Check if model is available locally
            is_local = any(os.path.exists(path) for path in config["local_paths"])
            
            models.append({
                "name": name,
                "description": config["description"],
                "memory_req": config["memory_req"],
                "strengths": config["strengths"],
                "available_locally": is_local
            })
        
        return models
    
    def benchmark_models(self, models_to_test: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run benchmarks on specified models.
        
        Args:
            models_to_test: List of model names to benchmark, or None for all
            
        Returns:
            Dict of benchmark results by model name
        """
        if models_to_test is None:
            models_to_test = list(self.MODEL_CONFIGS.keys())
        
        results = {}
        
        # Check system resources
        mem = psutil.virtual_memory()
        available_ram = mem.available / (1024 ** 3)  # GB
        logger.info(f"Available system RAM: {available_ram:.1f}GB")
        
        for model_name in models_to_test:
            logger.info(f"Benchmarking {model_name}...")
            
            # Skip models that would likely run out of memory
            if self.device == "cpu":
                req_mem = float(self.MODEL_CONFIGS[model_name]["memory_req"].replace("GB", ""))
                if req_mem > available_ram and self.quantization == "none":
                    logger.warning(f"Skipping {model_name} as it requires {req_mem}GB RAM but only {available_ram:.1f}GB available")
                    continue
            
            try:
                # Record loading time
                start_time = time.time()
                
                # Temporarily set this as the current model
                prev_model = self.model_name
                self.model_name = model_name
                
                # Load model
                self._load_model()
                
                load_time = time.time() - start_time
                
                # Test inference speed
                inference_time = self._benchmark_inference()
                
                # Record results
                results[model_name] = {
                    "load_time_seconds": load_time,
                    "inference_time_seconds": inference_time,
                    "device": self.device,
                    "quantization": self.quantization,
                    "success": True
                }
                
                # Clean up
                self.model = None
                self.tokenizer = None
                
                # Set back the original model
                self.model_name = prev_model
                
                # Force garbage collection
                import gc
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Benchmark failed for {model_name}: {e}")
                results[model_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def _benchmark_inference(self) -> float:
        """
        Run inference benchmark on loaded model.
        
        Returns:
            Average inference time in seconds
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be loaded before benchmarking")
        
        # Prepare sample input
        sample_texts = [
            "Analyze this football play: The midfielder kicks a long ball to the forward.",
            "What tactics should be used when defending against a counter-attack?",
            "Describe the movement patterns in a successful scoring play."
        ]
        
        total_time = 0
        for text in sample_texts:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Inference
            start_time = time.time()
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=50)
            total_time += time.time() - start_time
        
        return total_time / len(sample_texts)
    
    def get_model(self):
        """
        Load and return the selected model.
        
        Returns:
            Loaded model and tokenizer
        """
        if self.model is None:
            self._load_model()
        
        return self.model
    
    def get_tokenizer(self):
        """
        Load and return the selected tokenizer.
        
        Returns:
            Loaded tokenizer
        """
        if self.tokenizer is None:
            self._load_model()
        
        return self.tokenizer
    
    def _load_model(self):
        """Load the selected model and tokenizer."""
        self._import_libraries()
        
        logger.info(f"Loading model: {self.model_name}")
        
        start_time = time.time()
        
        # Determine model path/ID
        if self.model_path:
            # Use custom model path if provided
            model_id_or_path = self.model_path
            logger.info(f"Using custom model from: {model_id_or_path}")
        else:
            # Check if model is available locally
            config = self.MODEL_CONFIGS[self.model_name]
            local_model_path = None
            for path in config["local_paths"]:
                if os.path.exists(path):
                    local_model_path = path
                    break
            
            if local_model_path:
                model_id_or_path = local_model_path
                logger.info(f"Using locally available model from: {model_id_or_path}")
            else:
                model_id_or_path = config["huggingface_id"]
                logger.info(f"Downloading model from HuggingFace: {model_id_or_path}")
        
        # Configure quantization
        if self.quantization == "4bit" and self.device == "cuda":
            load_in_4bit = True
            load_in_8bit = False
        elif self.quantization == "8bit" and self.device == "cuda":
            load_in_4bit = False
            load_in_8bit = True
        else:
            load_in_4bit = False
            load_in_8bit = False
        
        # Load tokenizer first
        try:
            tokenizer_id = self.MODEL_CONFIGS.get(self.model_name, {}).get("tokenizer")
            if not tokenizer_id and os.path.exists(os.path.join(model_id_or_path, "tokenizer_config.json")):
                tokenizer_id = model_id_or_path
            
            if tokenizer_id:
                self.tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                    tokenizer_id,
                    trust_remote_code=True
                )
            else:
                self.tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                    model_id_or_path,
                    trust_remote_code=True
                )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
            
        # Load model
        try:
            if load_in_4bit or load_in_8bit:
                self.model = self.transformers.AutoModelForCausalLM.from_pretrained(
                    model_id_or_path,
                    device_map="auto",
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    trust_remote_code=True
                )
            else:
                self.model = self.transformers.AutoModelForCausalLM.from_pretrained(
                    model_id_or_path,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                # Move model to device if not using device_map
                if self.device != "cuda":
                    self.model = self.model.to(self.device)
                    
            self.model.eval()  # Set to evaluation mode
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Log model info
        model_size = sum(p.numel() for p in self.model.parameters()) / 1e9
        logger.info(f"Model size: {model_size:.2f}B parameters")
        
        return self.model, self.tokenizer
    
    def compare_models(self, sample_texts: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models on specific sports analysis tasks.
        
        Args:
            sample_texts: List of sample texts to test on
            
        Returns:
            Dict of results by model
        """
        models_to_test = list(self.MODEL_CONFIGS.keys())
        results = {}
        
        for model_name in models_to_test:
            # Skip if model would be too large for current system
            if self.device == "cpu":
                mem = psutil.virtual_memory()
                available_ram = mem.available / (1024 ** 3)  # GB
                req_mem = float(self.MODEL_CONFIGS[model_name]["memory_req"].replace("GB", ""))
                if req_mem > available_ram and self.quantization == "none":
                    logger.warning(f"Skipping {model_name} as it requires {req_mem}GB RAM")
                    continue
            
            logger.info(f"Testing {model_name}...")
            
            try:
                # Temporarily set this as the current model
                prev_model = self.model_name
                self.model_name = model_name
                
                # Load model
                self._load_model()
                
                # Run generation
                outputs = []
                generation_times = []
                
                for text in sample_texts:
                    inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                    
                    start_time = time.time()
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs, 
                            max_new_tokens=100,
                            temperature=0.7,
                            top_p=0.9
                        )
                    generation_time = time.time() - start_time
                    
                    output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    outputs.append(output_text)
                    generation_times.append(generation_time)
                
                # Store results
                results[model_name] = {
                    "outputs": outputs,
                    "avg_generation_time": sum(generation_times) / len(generation_times),
                    "device": self.device,
                    "success": True
                }
                
                # Clean up
                self.model = None
                self.tokenizer = None
                
                # Set back the original model
                self.model_name = prev_model
                
                # Force garbage collection
                import gc
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Comparison failed for {model_name}: {e}")
                results[model_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
