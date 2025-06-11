# Model Optimization Agent

A modular, simplified model benchmarking agent using **optimum-quanto** for quantization with Gradio web interface and MCP server support.

## 🚀 Features

- **Optimum-Quanto Integration**: Modern quantization with int8, int4, int2, and float8 support
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Multiple Interfaces**: Gradio web UI, MCP server, and CLI
- **Torch Compile Support**: Optional PyTorch 2.0 compilation for speed
- **Cross-Platform**: CUDA, MPS (Apple Silicon), and CPU support
- **Comprehensive Metrics**: Throughput, latency, memory, and perplexity

## 📁 Project Structure

```
model-benchmark-agent/
├── models/
│   └── quantization.py      # Quanto-based model loading
├── core/
│   ├── benchmark.py         # Core benchmarking logic  
│   ├── data.py             # Dataset utilities
│   └── utils.py            # Helper functions
├── agent/
│   └── benchmarker.py      # Main benchmarking agent
├── interfaces/
│   ├── gradio_app.py       # Web interface with MCP Server
├── cli.py                  # Command-line interface
├── main.py                 # Entry point
└── pyproject.toml          # Dependencies
```

## 🛠 Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## 🎯 Quick Start

### Web Interface
```bash
# Login to HuggingFace if using gated models
huggingface-cli login

uv run python main.py gradio
```
Access at `http://localhost:7860`
OR
Copy the generated MCP server URL to your Host's integration to access the tool through Claude Desktop.

### Command Line
```bash
# Basic benchmark with any model of your choice
uv run python main.py cli --model facebook/opt-iml-max-1.3b

# With quantization
uv run python main.py cli --model facebook/opt-iml-max-1.3b --quantization int8

# Full options
uv run python main.py cli \
  --model facebook/opt-iml-max-1.3b \
  --quantization int4 \
  --torch-compile \
  --samples 10 \
  --perplexity
```

### MCP Server
```bash
uv run python main.py mcp
```

## 🔧 Quantization Types

- **none**: Standard float16/float32
- **int8**: 8-bit integer quantization  
- **int4**: 4-bit integer quantization
- **int2**: 2-bit integer quantization  
- **float8**: 8-bit floating point

## 📊 Web Interface Features

1. **Single Benchmark**: Test individual model configurations
2. **Optimization Comparison**: Side-by-side comparison with charts
3. **History**: Track past benchmark results
4. **System Info**: Hardware capability detection

## 🔌 MCP Tools

- `benchmark_model`: Run single model benchmark
- `compare_optimizations`: Compare multiple quantization strategies  
- `get_system_info`: Get hardware information

## 📈 Example Usage

### Python API
```python
from agent.benchmarker import ModelBenchmarker
from core.benchmark import BenchmarkConfig

config = BenchmarkConfig(
    model_name="facebook/opt-iml-max-1.3b",
    quantization_type="int8",
    use_torch_compile=True,
    num_samples=10
)

benchmarker = ModelBenchmarker()
results = benchmarker.run_benchmark(config)
print(f"Throughput: {results['summary']['avg_tokens_per_second']:.2f} tok/s")
```

### Comparison Script
```python
from agent.benchmarker import ModelBenchmarker
from core.benchmark import BenchmarkConfig

optimizations = ["none", "int8", "int4"]
results = []

for opt in optimizations:
    config = BenchmarkConfig(
        model_name="facebook/opt-iml-max-1.3b",
        quantization_type=opt,
        num_samples=5
    )
    benchmarker = ModelBenchmarker()
    result = benchmarker.run_benchmark(config)
    results.append(result["summary"])

# Compare results
for r in results:
    print(f"{r['optimization_type']}: {r['avg_tokens_per_second']:.2f} tok/s")
```

## 🎛 Configuration

The `BenchmarkConfig` class handles all configuration:

```python
@dataclass
class BenchmarkConfig:
    model_name: str                    # HuggingFace model ID
    dataset_name: str = "tatsu-lab/alpaca"  # Dataset for prompts
    num_samples: int = 20              # Number of test samples
    max_new_tokens: int = 100          # Max tokens to generate
    quantization_type: str = "none"    # Quantization strategy
    use_torch_compile: bool = False    # Enable torch.compile
    calculate_perplexity: bool = False # Quality metric
    device: Optional[str] = None       # Target device
    seed: int = 42                     # Random seed
```
