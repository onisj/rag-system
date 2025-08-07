"""
Setup Model Script - Model conversion and setup utilities

This script provides utilities for setting up and converting models
for the RAG system using OpenVINO GenAI.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.genai_model_converter import GenAIModelConverter
from rich.console import Console

console = Console()

def main():
    """Main function for model setup"""
    try:
        console.print("Setting up model for RAG system...", style="bold blue")
        
        # Initialize model converter
        converter = GenAIModelConverter("meta-llama/Llama-3.1-8B-Instruct")
        
        # Run conversion pipeline
        output_path = converter.run_conversion_pipeline(
            skip_download=False,
            skip_validation=False
        )
        
        console.print(f"Model setup completed: {output_path}", style="bold green")
        return 0
        
    except Exception as e:
        console.print(f"Model setup failed: {e}", style="red")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 