"""Command-line interface."""

import argparse
from agent.benchmarker import ModelBenchmarker
from core.benchmark import BenchmarkConfig

def main():
    parser = argparse.ArgumentParser(description="Model Benchmark Agent CLI")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--dataset", default="tatsu-lab/alpaca", help="Dataset name")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--tokens", type=int, default=100, help="Max new tokens")
    parser.add_argument("--quantization", choices=["none", "int8", "int4", "int2", "float8"], 
                       default="none", help="Quantization type")
    parser.add_argument("--torch-compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--perplexity", action="store_true", help="Calculate perplexity")
    parser.add_argument("--device", help="Device to use")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        num_samples=args.samples,
        max_new_tokens=args.tokens,
        quantization_type=args.quantization,
        use_torch_compile=args.torch_compile,
        calculate_perplexity=args.perplexity,
        device=args.device
    )
    
    print(f"🔄 Benchmarking {args.model} with {args.quantization} quantization...")
    
    benchmarker = ModelBenchmarker()
    results = benchmarker.run_benchmark(config)
    output_file = benchmarker.save_results(results, args.output_dir)
    
    # Print summary
    summary = results["summary"]
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY: {summary['model_name']}")
    print(f"{'='*60}")
    print(f"Throughput: {summary['avg_tokens_per_second']:.2f} tokens/second")
    print(f"First Token Latency: {summary['avg_first_token_latency_seconds']:.4f} seconds")
    print(f"Memory Usage: {summary['max_memory_mb']:.2f} MB")
    print(f"Device: {summary['device']}")
    
    if summary.get('avg_perplexity'):
        print(f"Perplexity: {summary['avg_perplexity']:.4f}")
    
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()