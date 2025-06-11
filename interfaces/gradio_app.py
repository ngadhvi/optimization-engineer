import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple

from agent.benchmarker import ModelBenchmarker
from core.benchmark import BenchmarkConfig
from core.utils import get_system_info

class GradioApp:
    """Gradio web interface for model benchmarking."""
    
    def __init__(self):
        self.benchmarker = ModelBenchmarker()
        self.history = []
        
    def benchmark_single(
        self,
        model_name: str,
        dataset_name: str,
        num_samples: int,
        max_tokens: int,
        quantization: str,
        torch_compile: bool,
        perplexity: bool,
        device: str
    ) -> Tuple[str, str, str]:
        """Run single model benchmark."""
        try:
            config = BenchmarkConfig(
                model_name=model_name,
                dataset_name=dataset_name,
                num_samples=num_samples,
                max_new_tokens=max_tokens,
                quantization_type=quantization,
                use_torch_compile=torch_compile,
                calculate_perplexity=perplexity,
                device=device if device != "auto" else None
            )
            
            results = self.benchmarker.run_benchmark(config)
            self.history.append(results)
            
            # Format summary
            summary = results["summary"]
            summary_text = f"""## Benchmark Results

**Model**: {summary['model_name']}  
**Device**: {summary['device']}  
**Optimization**: {summary['optimization_type']}

### Performance Metrics
- **Throughput**: {summary['avg_tokens_per_second']:.2f} tokens/second
- **First Token Latency**: {summary['avg_first_token_latency_seconds']:.4f} seconds  
- **Peak Memory**: {summary['max_memory_mb']:.2f} MB
- **Samples**: {summary['num_samples']}
{f"- **Perplexity**: {summary['avg_perplexity']:.4f}" if summary.get('avg_perplexity') else ""}
            """
            
            # Sample results table
            samples_df = pd.DataFrame(results['samples'])
            if not samples_df.empty:
                display_cols = ['prompt_id', 'input_tokens', 'output_tokens', 'tokens_per_second', 'first_token_latency_seconds']
                samples_table = samples_df[display_cols].head(10).to_html(index=False)
            else:
                samples_table = "No sample data available"
            
            return summary_text, samples_table, "✅ Benchmark completed!"
            
        except Exception as e:
            return f"❌ Error: {str(e)}", "", f"❌ Failed: {str(e)}"
    
    def compare_optimizations(
        self,
        model_name: str,
        dataset_name: str,
        num_samples: int,
        optimizations: List[str]
    ) -> Tuple[str, go.Figure, str]:
        """Compare different quantization."""
        try:
            results = []
            
            for opt in optimizations:
                config = BenchmarkConfig(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    num_samples=num_samples,
                    quantization_type=opt,
                    calculate_perplexity=True
                )
                
                benchmarker = ModelBenchmarker()  # Fresh instance
                result = benchmarker.run_benchmark(config)
                results.append(result["summary"])
            
            # Create comparison
            df = pd.DataFrame(results)
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Throughput',
                x=df['optimization_type'],
                y=df['avg_tokens_per_second'],
                yaxis='y'
            ))
            
            fig.add_trace(go.Scatter(
                name='Memory (MB)',
                x=df['optimization_type'], 
                y=df['max_memory_mb'],
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f'Optimization Comparison: {model_name}',
                xaxis_title='Optimization',
                yaxis=dict(title='Throughput (tok/s)', side='left'),
                yaxis2=dict(title='Memory (MB)', side='right', overlaying='y')
            )
            
            # Summary text
            best_throughput = max(results, key=lambda x: x['avg_tokens_per_second'])
            best_memory = min(results, key=lambda x: x['max_memory_mb'])
            
            summary = f"""## Comparison Results

### Best Configurations
- **Highest Throughput**: {best_throughput['optimization_type']} ({best_throughput['avg_tokens_per_second']:.2f} tok/s)
- **Lowest Memory**: {best_memory['optimization_type']} ({best_memory['max_memory_mb']:.2f} MB)

### Results Table
| Optimization | Throughput | Memory | Perplexity |
|--------------|-----------|---------|-----------|
{chr(10).join([f"| {r['optimization_type']} | {r['avg_tokens_per_second']:.2f} | {r['max_memory_mb']:.2f} | {r.get('avg_perplexity', 'N/A')} |" for r in results])}
            """
            
            return summary, fig, "✅ Comparison completed!"
            
        except Exception as e:
            return f"❌ Error: {str(e)}", go.Figure(), f"❌ Failed: {str(e)}"
    
    def get_history(self) -> str:
        """Get benchmark history."""
        if not self.history:
            return "No benchmarks run yet."
        
        history_text = "# Benchmark History\n\n"
        for i, result in enumerate(self.history):
            summary = result["summary"]
            history_text += f"""## Run {i+1}
- **Model**: {summary['model_name']}
- **Time**: {summary['timestamp']}  
- **Throughput**: {summary['avg_tokens_per_second']:.2f} tok/s
- **Memory**: {summary['max_memory_mb']:.2f} MB

---
            """
        
        return history_text
    
    def create_interface(self):
        """Create Gradio interface."""
        with gr.Blocks(title="Model Benchmark Agent", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🚀 Model Benchmark Agent")
            gr.Markdown("Benchmark Hugging Face models with optimum-quanto quantization")
            
            with gr.Tabs():
                # Single Benchmark Tab
                with gr.TabItem("Single Benchmark"):
                    with gr.Row():
                        with gr.Column():
                            model_input = gr.Textbox("facebook/opt-iml-max-1.3b", label="Model Name")
                            dataset_input = gr.Textbox("tatsu-lab/alpaca", label="Dataset")
                            num_samples = gr.Slider(1, 100, 20, step=1, label="Samples")
                            max_tokens = gr.Slider(10, 512, 100, label="Max Tokens")
                            quantization = gr.Dropdown(
                                ["none", "int8", "int4", "int2", "float8"],
                                value="none",
                                label="Quantization"
                            )
                            torch_compile = gr.Checkbox(label="Use torch.compile")
                            perplexity = gr.Checkbox(label="Calculate Perplexity")
                            device = gr.Dropdown(["auto", "cuda", "cpu", "mps"], value="auto", label="Device")
                            
                            benchmark_btn = gr.Button("🚀 Run Benchmark", variant="primary")
                        
                        with gr.Column():
                            results_md = gr.Markdown()
                            samples_html = gr.HTML()
                            status_text = gr.Textbox(label="Status", interactive=False)
                    
                    benchmark_btn.click(
                        self.benchmark_single,
                        inputs=[model_input, dataset_input, num_samples, max_tokens, quantization, torch_compile, perplexity, device],
                        outputs=[results_md, samples_html, status_text]
                    )
                
                # Comparison Tab
                with gr.TabItem("Compare Optimizations"):
                    with gr.Row():
                        with gr.Column():
                            comp_model = gr.Textbox("facebook/opt-iml-max-1.3b", label="Model")
                            comp_dataset = gr.Textbox("tatsu-lab/alpaca", label="Dataset")
                            comp_samples = gr.Slider(1, 50, 10, step=1, label="Samples")
                            comp_opts = gr.CheckboxGroup(
                                ["none", "int8", "int4", "int2"],
                                value=["none", "int8"],
                                label="Optimizations to Compare"
                            )
                            
                            compare_btn = gr.Button("📊 Compare", variant="primary")
                        
                        with gr.Column():
                            comp_results = gr.Markdown()
                            comp_plot = gr.Plot()
                            comp_status = gr.Textbox(label="Status", interactive=False)
                    
                    compare_btn.click(
                        self.compare_optimizations,
                        inputs=[comp_model, comp_dataset, comp_samples, comp_opts],
                        outputs=[comp_results, comp_plot, comp_status]
                    )
                
                # History Tab
                with gr.TabItem("History"):
                    history_md = gr.Markdown()
                    refresh_btn = gr.Button("🔄 Refresh")
                    refresh_btn.click(self.get_history, outputs=[history_md])
                
                # System Info Tab
                with gr.TabItem("System Info"):
                    sys_info_md = gr.Markdown()
                    sys_info_btn = gr.Button("📋 Get System Info")
                    sys_info_btn.click(get_system_info, outputs=[sys_info_md])
        
        return app

def launch_app():
    """Launch the Gradio app."""
    app = GradioApp()
    interface = app.create_interface()
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860, mcp_server=True)