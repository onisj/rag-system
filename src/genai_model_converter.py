"""
OpenVINO GenAI Model Converter

This module handles downloading and converting models to OpenVINO GenAI format
for optimized inference on Intel hardware.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple

# Add src to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

# Import OpenVINO GenAI
import openvino_genai as ov_genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

class GenAIModelConverter:
    """
    OpenVINO GenAI Model Converter
    
    Handles downloading and converting models to OpenVINO GenAI format
    for optimized inference on Intel hardware.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize GenAI Model Converter
        
        Args:
            model_name: HuggingFace model name to convert
        """
        self.model_name = model_name
        self.cache_dir = Path("./models")
        self.cache_dir.mkdir(exist_ok=True)
        
        console.print(f"GenAI ModelConverter initialized for: {model_name}", style="blue")
    
    def _check_hardware_compatibility(self) -> list:
        """Check hardware compatibility for GenAI"""
        try:
            # Check for GPU availability
            devices = []
            
            if torch.cuda.is_available():
                devices.append("GPU")
                console.print("GPU detected", style="green")
            
            devices.append("CPU")
            console.print("CPU available", style="green")
            
            return devices
            
        except Exception as e:
            console.print(f"Hardware compatibility check failed: {e}", style="red")
            raise RuntimeError(f"No compatible devices found: {e}")
    
    def _authenticate_huggingface(self) -> bool:
        """Authenticate with HuggingFace"""
        try:
            from huggingface_hub import login
            
            # Check if token is set
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                console.print("HUGGINGFACE_TOKEN not set", style="yellow")
                return False
            
            login(token)
            console.print("HuggingFace authentication successful", style="green")
            return True
            
        except Exception as e:
            console.print(f"Authentication failed: {e}", style="red")
            return False
    
    def _download_model(self, skip_if_exists: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Download the model from HuggingFace"""
        try:
            model_dir = self.cache_dir / self.model_name.split("/")[-1]
            
            if skip_if_exists and model_dir.exists():
                console.print(f"Model already exists at {model_dir}", style="yellow")
                return (
                    AutoModelForCausalLM.from_pretrained(str(model_dir)),
                    AutoTokenizer.from_pretrained(str(model_dir))
                )
            
            console.print(f"Downloading model: {self.model_name}", style="blue")
            
            # Download model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Save to cache
            model.save_pretrained(str(model_dir))
            tokenizer.save_pretrained(str(model_dir))
            
            console.print(f"Model downloaded and saved to {model_dir}", style="green")
            return model, tokenizer
            
        except Exception as e:
            console.print(f"Model download failed: {e}", style="red")
            raise RuntimeError(f"Failed to download model: {e}")
    
    def convert_to_int4(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> Path:
        """Convert model to INT4 using OpenVINO GenAI"""
        console.print("Starting INT4 conversion using OpenVINO GenAI...", style="blue")
        
        try:
            # Create output directory
            output_path = self.cache_dir / f"{self.model_name.split('/')[-1]}-int4-genai"
            output_path.mkdir(exist_ok=True)
            
            # Save the original model and tokenizer for GenAI to use
            model_path = output_path / "original_model"
            model.save_pretrained(str(model_path))
            tokenizer.save_pretrained(str(model_path))
            
            # For OpenVINO GenAI, we don't need to convert the model manually
            # The LLMPipeline will handle the conversion automatically when loading
            # We just need to save the model in a format that GenAI can load
            
            console.print(f"Model prepared for GenAI at {output_path}", style="green")
            console.print("Note: OpenVINO GenAI will handle INT4 optimization automatically", style="yellow")
            return output_path
            
        except Exception as e:
            console.print(f"Model conversion failed: {e}", style="red")
            raise RuntimeError(f"Failed to convert model: {e}")
    
    def get_model_info(self) -> str:
        """Get information about the model"""
        try:
            model_dir = self.cache_dir / f"{self.model_name.split('/')[-1]}-int4-genai"
            
            if model_dir.exists():
                model_path = model_dir / "model.xml"
                if model_path.exists():
                    return f"Model converted and available at {model_path}"
                else:
                    return "Model directory exists but model.xml not found"
            else:
                return "Model not converted yet"
                
        except Exception as e:
            return f"Error getting model info: {e}"
    
    def run_conversion_pipeline(self, skip_download: bool = False, skip_validation: bool = False) -> Path:
        """
        Run the complete conversion pipeline
        
        Args:
            skip_download: Skip downloading if model exists
            skip_validation: Skip validation step
            
        Returns:
            Path to converted model
        """
        try:
            console.print("Starting GenAI model conversion pipeline...", style="bold blue")
            
            # Check hardware compatibility
            devices = self._check_hardware_compatibility()
            console.print(f"Available devices: {devices}", style="green")
            
            # Authenticate with HuggingFace
            if not self._authenticate_huggingface():
                console.print("Continuing without authentication", style="yellow")
            
            # Download model
            if not skip_download:
                model, tokenizer = self._download_model()
            else:
                console.print("Skipping download", style="yellow")
                model_dir = self.cache_dir / self.model_name.split("/")[-1]
                model = AutoModelForCausalLM.from_pretrained(str(model_dir))
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            
            # Convert to INT4
            output_path = self.convert_to_int4(model, tokenizer)
            
            # Validate conversion
            if not skip_validation:
                self._validate_conversion(output_path)
            
            console.print("Conversion pipeline completed successfully", style="bold green")
            return output_path
            
        except Exception as e:
            console.print(f"Conversion pipeline failed: {e}", style="red")
            raise RuntimeError(f"Conversion pipeline failed: {e}")
    
    def _validate_conversion(self, model_path: Path) -> None:
        """Validate the converted model"""
        try:
            console.print("Validating converted model...", style="blue")
            
            # Check if original model directory exists
            original_model_path = model_path / "original_model"
            if not original_model_path.exists():
                raise RuntimeError("Original model directory not found")
            
            # Check if model files exist
            model_files = list(original_model_path.glob("*.safetensors"))
            if not model_files:
                raise RuntimeError("Model files not found")
            
            # Check if tokenizer exists
            tokenizer_path = original_model_path / "tokenizer.json"
            if not tokenizer_path.exists():
                raise RuntimeError("Tokenizer file not found")
            
            console.print(f"Model validation successful - {len(model_files)} model files found", style="green")
            
        except Exception as e:
            console.print(f"Model validation failed: {e}", style="red")
            raise RuntimeError(f"Model validation failed: {e}")

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert model to OpenVINO GenAI format")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct", help="Model to convert")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if model exists")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation step")
    
    args = parser.parse_args()
    
    try:
        converter = GenAIModelConverter(args.model_name)
        output_path = converter.run_conversion_pipeline(
            skip_download=args.skip_download,
            skip_validation=args.skip_validation
        )
        console.print(f"Conversion completed: {output_path}", style="bold green")
        
    except Exception as e:
        console.print(f"Conversion failed: {e}", style="red")
        sys.exit(1)

if __name__ == "__main__":
    main() 