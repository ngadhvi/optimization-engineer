import torch
import time
import gc
import psutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from torch.nn import CrossEntropyLoss

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    model_name: str
    dataset_name: str = "tatsu-lab/alpaca"
    num_samples: int = 20
    max_new_tokens: int = 100
    quantization_type: str = "none"
    use_torch_compile: bool = False
    calculate_perplexity: bool = False
    device: Optional[str] = None
    seed: int = 42

@dataclass 
class BenchmarkResult:
    """Single benchmark result."""
    prompt_id: int
    prompt: str
    generated_text: str
    input_tokens: int
    output_tokens: int
    total_time_seconds: float
    tokens_per_second: float
    first_token_latency_seconds: float
    peak_memory_mb: float
    perplexity: Optional[float] = None

class MemoryTracker:
    """Handles memory tracking across different devices."""
    
    def __init__(self, device: str):
        self.device = device
        
    def reset_stats(self):
        """Reset memory tracking."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            return psutil.Process().memory_info().rss / (1024 * 1024)
    
    def synchronize(self):
        """Synchronize device operations."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.device == "mps" and hasattr(torch.backends, 'mps'):
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
    
    def clear_cache(self):
        """Clear memory cache."""
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

class PerplexityCalculator:
    """Handles perplexity calculation."""
    
    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer  
        self.device = device
        
    def calculate(self, text: str) -> float:
        """Calculate perplexity of text."""
        try:
            encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_ids = encodings.input_ids
            
            if input_ids.size(1) <= 1:
                return float('inf')
                
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=input_ids.clone())
                
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    return torch.exp(outputs.loss).item()
                
                # Fallback manual calculation
                logits = outputs.logits[:, :-1, :].contiguous()
                labels = input_ids[:, 1:].contiguous()
                
                loss_fn = CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                return torch.exp(loss).item()
                
        except Exception as e:
            print(f"Perplexity calculation failed: {e}")
            return None

class InferenceRunner:
    """Handles model inference with timing and memory tracking."""
    
    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.memory_tracker = MemoryTracker(device)
        
    def run_single_inference(self, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        """Run inference on a single prompt."""
        # Tokenize input
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        input_token_count = input_ids.shape[1]
        
        # Reset memory tracking
        self.memory_tracker.reset_stats()
        initial_memory = self.memory_tracker.get_peak_memory_mb()
        
        # Generation parameters
        gen_params = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Time first token
        self.memory_tracker.synchronize()
        first_token_start = time.time()
        
        with torch.no_grad():
            first_output = self.model.generate(input_ids, max_new_tokens=1, **{k: v for k, v in gen_params.items() if k != 'max_new_tokens'})
            
        self.memory_tracker.synchronize()
        first_token_latency = time.time() - first_token_start
        
        # Full generation
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(input_ids, **gen_params)
            
        self.memory_tracker.synchronize()
        total_time = time.time() - start_time
        
        # Calculate metrics
        output_ids = outputs[0][input_token_count:]
        generated_token_count = len(output_ids)
        tokens_per_second = generated_token_count / total_time if total_time > 0 else 0
        
        # Get memory usage
        peak_memory_mb = self.memory_tracker.get_peak_memory_mb()
        if self.device != "cuda":
            peak_memory_mb = peak_memory_mb - initial_memory
            
        # Decode output
        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Clear memory
        self.memory_tracker.clear_cache()
        
        return {
            "input_tokens": input_token_count,
            "output_tokens": generated_token_count,
            "total_time_seconds": total_time,
            "tokens_per_second": tokens_per_second,
            "first_token_latency_seconds": first_token_latency,
            "peak_memory_mb": peak_memory_mb,
            "generated_text": generated_text
        }