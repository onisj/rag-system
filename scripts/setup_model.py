"""
Setup Model Script - Model Conversion and Setup Utilities for RAG System

This script provides utilities to initialize and convert the Llama-3.1-8B-Instruct model
for the Retrieval-Augmented Generation (RAG) system using OpenVINO GenAI. It leverages
the GenAIModelConverter class to download, convert, and validate the model, producing
an OpenVINO IR format model with INT4 quantization optimized for inference.

Key Features:
    - Initializes the model converter with the specified HuggingFace model identifier.
    - Executes the conversion pipeline to download, convert, and validate the model.
    - Provides rich console output for user feedback and error handling.
    - Integrates with the RAG system by adding the project root to the Python path.

Usage:
    Run the script from the command line to set up the model:
        python setup_model.py
    The script outputs the path to the converted model directory on success or an error
    message on failure.

Author: Segun Oni
Version: 1.0.1
Last Modified: August 8, 2025
"""

import sys
from pathlib import Path

# Initialize rich console for enhanced, colored terminal output
from rich.console import Console
console = Console()

# Add project root to Python path for importing src modules
current_dir = Path(__file__).parent  # Get directory of this script
project_root = current_dir.parent    # Get parent directory (project root)
sys.path.insert(0, str(project_root))  # Prepend project root to sys.path

# Import GenAIModelConverter from src for model conversion
from src.genai_model_converter import GenAIModelConverter

def main() -> int:
    """
    Main function to execute the model setup pipeline for the RAG system.

    Initializes the GenAIModelConverter with the Llama-3.1-8B-Instruct model,
    runs the conversion pipeline (download, INT4 conversion, validation), and
    outputs the result to the console. Returns an exit code indicating success
    or failure.

    Returns:
        int: Exit code (0 for success, 1 for failure)

    Raises:
        Exception: If any step in the conversion pipeline fails (e.g., download,
                   conversion, or validation errors)
    """
    try:
        # Display setup start message
        console.print("Setting up model for RAG system...", style="bold blue")

        # Initialize converter for Llama-3.1-8B-Instruct model
        converter = GenAIModelConverter("meta-llama/Llama-3.1-8B-Instruct")

        # Run full conversion pipeline: download, convert to INT4, validate
        output_path = converter.run_conversion_pipeline(
            skip_download=False,  # Download model if not already cached
            skip_validation=False  # Perform validation on converted model
        )

        # Display success message with output path
        console.print(f"Model setup completed: {output_path}", style="bold green")
        return 0  # Success exit code

    except Exception as e:
        # Display error message and return failure exit code
        console.print(f"Model setup failed: {e}", style="red")
        return 1  # Failure exit code

if __name__ == "__main__":
    # Execute main function and exit with its return code
    sys.exit(main())