import torch
import time
import json
import os
import numpy as np
from typing import Dict, List, Any
from dataclasses import asdict

from models.quantization import ModelLoader, QuantizationType
from core.benchmark import BenchmarkConfig, BenchmarkResult, InferenceRunner, PerplexityCalculator
from core.data import DatasetLoader
from core.utils import get_device

class ModelBenchmarker:
    """Main benchmarking agent."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def load_model(self, config: BenchmarkConfig):
        """Load model based on configuration."""
        self.device = get_device(config.device)
        
        quant_type = QuantizationType(config.quantization_type)
        
        if quant_type == QuantizationType.NONE:
            self.model, self.tokenizer = ModelLoader.load_standard(config.model_name, self.device)
        else:
            # Try Transformers integration first, fallback to direct API
            try:
                self.model, self.tokenizer = ModelLoader.load_quantized_transformers(config.model_name, quant_type)
                self.device = str(next(self.model.parameters()).device)
            except Exception as e:
                print(f"Transformers integration failed, using direct API: {e}")
                self.model, self.tokenizer = ModelLoader.load_quantized_direct(config.model_name, quant_type, self.device)
        
        # Apply torch.compile if requested
        if config.use_torch_compile:
            print("Applying torch.compile...")
            self.model = torch.compile(self.model)
            
    def run_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run benchmark with given configuration."""
        if self.model is None:
            self.load_model(config)
            
        # Get sample prompts
        prompts, indices = DatasetLoader.get_sample_prompts(config.dataset_name, config.num_samples, config.seed)
        
        # Setup inference runner
        inference_runner = InferenceRunner(self.model, self.tokenizer, self.device)
        
        # Setup perplexity calculator if needed
        perplexity_calc = None
        if config.calculate_perplexity:
            perplexity_calc = PerplexityCalculator(self.model, self.tokenizer, self.device)
        
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}")
            
            # Run inference
            inference_result = inference_runner.run_single_inference(prompt, config.max_new_tokens)
            
            # Calculate perplexity if requested
            perplexity = None
            if perplexity_calc:
                perplexity = perplexity_calc.calculate(inference_result["generated_text"])
            
            # Create result
            result = BenchmarkResult(
                prompt_id=i,
                prompt=prompt,
                generated_text=inference_result["generated_text"],
                input_tokens=inference_result["input_tokens"],
                output_tokens=inference_result["output_tokens"],
                total_time_seconds=inference_result["total_time_seconds"],
                tokens_per_second=inference_result["tokens_per_second"],
                first_token_latency_seconds=inference_result["first_token_latency_seconds"],
                peak_memory_mb=inference_result["peak_memory_mb"],
                perplexity=perplexity
            )
            
            results.append(result)
        
        # Calculate summary
        summary = self._create_summary(config, results)
        
        return {
            "summary": summary,
            "samples": [asdict(result) for result in results]
        }
    
    def _create_summary(self, config: BenchmarkConfig, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Create benchmark summary."""
        avg_tokens_per_second = sum(r.tokens_per_second for r in results) / len(results)
        avg_first_token_latency = sum(r.first_token_latency_seconds for r in results) / len(results)
        max_memory_mb = max(r.peak_memory_mb for r in results)
        
        avg_perplexity = None
        if config.calculate_perplexity:
            valid_perplexities = [r.perplexity for r in results if r.perplexity is not None and not np.isinf(r.perplexity)]
            if valid_perplexities:
                avg_perplexity = sum(valid_perplexities) / len(valid_perplexities)
        
        optimization_desc = config.quantization_type
        if config.use_torch_compile:
            optimization_desc += " + torch.compile"
            
        return {
            "model_name": f"{config.model_name} ({optimization_desc})",
            "device": self.device,
            "num_samples": len(results),
            "avg_tokens_per_second": avg_tokens_per_second,
            "avg_first_token_latency_seconds": avg_first_token_latency,
            "max_memory_mb": max_memory_mb,
            "avg_perplexity": avg_perplexity,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "optimization_type": optimization_desc
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "benchmark_results") -> str:
        """Save benchmark results."""
        os.makedirs(output_dir, exist_ok=True)
        
        model_name = results["summary"]["model_name"].split('/')[-1].replace(' ', '_')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{model_name}_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        return output_file