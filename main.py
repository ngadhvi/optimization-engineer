#!/usr/bin/env python3
"""Main entry point for the Model Benchmark Agent."""
import os

# Disable tokenizer parallelism to avoid forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import sys

def main():
    parser = argparse.ArgumentParser(description="Model Optimization Agent")
    parser.add_argument("mode", choices=["gradio", "cli"], 
                       help="Interface mode: gradio (web), mcp (server), or cli (command line)")
    
    args, remaining = parser.parse_known_args()
    
    if args.mode == "gradio":
        from interfaces.gradio_app import launch_app
        launch_app()
        
    # elif args.mode == "mcp":
    #     from model_optimization.mcp_server import MCPServer
    #     server = MCPServer()
    #     asyncio.run(server.run())
        
    elif args.mode == "cli":
        # Redirect to CLI module
        sys.argv = [sys.argv[0]] + remaining
        from cli import main as cli_main
        cli_main()

if __name__ == "__main__":
    main()