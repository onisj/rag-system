"""
OpenVINO GenAI Model Converter

This module provides functionality for downloading the Llama-3.1-8B-Instruct model from HuggingFace
and converting it to OpenVINO IR format with INT4 quantization for use with OpenVINO GenAI.

The converter handles the complete pipeline from model download through conversion to validation,
ensuring compatibility with OpenVINO GenAI's LLMPipeline for a RAG pipeline on Windows.

Key Features:
    - Downloads Llama-3.1-8B-Instruct model from HuggingFace
    - Converts to OpenVINO IR format (.xml and .bin files) using Optimum Intel
    - Applies INT4 quantization for optimal performance
    - Validates compatibility with OpenVINO GenAI
    - Provides comprehensive error handling and logging

Requirements:
    - HUGGINGFACE_TOKEN environment variable must be set for gated model access
    - OpenVINO GenAI and Optimum Intel packages must be installed
    - Sufficient disk space for model storage (approximately 16GB)

Usage:
    converter = GenAIModelConverter("meta-llama/Llama-3.1-8B-Instruct")
    output_path = converter.run_conversion_pipeline()

Author: Segun Oni
Version: 1.0.1
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple
import datetime
import subprocess

# Add project root to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Core dependencies
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

# OpenVINO imports
try:
    import openvino as ov
    import openvino_genai as ov_genai
except ImportError as e:
    raise ImportError(
        f"Required OpenVINO packages not found: {e}. "
        "Please install: pip install openvino openvino-genai optimum[openvino]"
    )

# Configure module-level logging with file output
logs_dir = Path("./logs")
logs_dir.mkdir(exist_ok=True)

# Create timestamped log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = logs_dir / f"genai_model_converter_{timestamp}.log"

# Configure logging to write to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize rich console for enhanced output formatting
console = Console()


class GenAIModelConverter:
    """
    OpenVINO GenAI Model Converter

    This class provides a pipeline for downloading and converting Llama-3.1-8B-Instruct
    to OpenVINO IR format with INT4 quantization, as per UL Benchmarks requirements.

    Attributes:
        model_name (str): HuggingFace model identifier
        cache_dir (Path): Directory for storing downloaded and converted models

    Example:
        converter = GenAIModelConverter("meta-llama/Llama-3.1-8B-Instruct")
        model_path = converter.run_conversion_pipeline()
        print(f"Model converted and saved to: {model_path}")
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the GenAI Model Converter.

        Args:
            model_name (str): HuggingFace model identifier. Defaults to
                             "meta-llama/Llama-3.1-8B-Instruct"

        Raises:
            ValueError: If model_name is empty or invalid
            RuntimeError: If cache directory cannot be created
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")

        self.model_name = model_name
        self.cache_dir = Path("./models")
        try:
            self.cache_dir.mkdir(exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create cache directory: {e}")

        logger.info(f"GenAI ModelConverter initialized for: {model_name}")
        console.print(f"GenAI ModelConverter initialized for: {model_name}", style="blue")

    def _check_hardware_compatibility(self) -> list:
        """
        Check available hardware for inference using OpenVINO and PyTorch.

        Returns:
            list: List of available device strings (e.g., ['GPU.0', 'GPU.1', 'CPU'])

        Raises:
            RuntimeError: If no compatible devices are found or detection fails
        """
        try:
            devices = []
            try:
                from openvino import Core
                core = Core()
                ov_devices = core.available_devices
                gpu_count = 0
                for device in ov_devices:
                    if device.startswith("GPU"):
                        device_name = core.get_property(device, "FULL_DEVICE_NAME")
                        device_id = f"GPU.{gpu_count}"
                        devices.append(device_id)
                        gpu_count += 1
                        logger.info(f"OpenVINO GPU detected: {device_name} ({device_id})")
                        console.print(f"OpenVINO GPU detected: {device_name} ({device_id})", style="green")
                    elif device == "CPU":
                        devices.append(device)
                        logger.info("CPU available for inference")
                        console.print("CPU available for inference", style="green")
            except ImportError as e:
                logger.warning(f"OpenVINO not installed: {e}. Falling back to PyTorch CUDA detection.")
                console.print(f"OpenVINO not installed: {e}. Falling back to PyTorch CUDA detection.", style="yellow")
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    device_id = f"GPU.{gpu_count}"
                    if device_id not in devices:
                        devices.append(device_id)
                        logger.info(f"PyTorch CUDA GPU detected: {gpu_name} ({device_id})")
                        console.print(f"PyTorch CUDA GPU detected: {gpu_name} ({device_id})", style="green")
            except ImportError as e:
                logger.warning(f"PyTorch not installed: {e}. Skipping CUDA detection.")
                console.print(f"PyTorch not installed: {e}. Skipping CUDA detection.", style="yellow")
            if not devices:
                raise RuntimeError("No compatible devices found for inference")
            return devices


    def _authenticate_huggingface(self) -> bool:
        """
        Authenticate with HuggingFace Hub using environment token.

        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            from huggingface_hub import login
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                logger.warning("HUGGINGFACE_TOKEN not set")
                console.print("HUGGINGFACE_TOKEN not set", style="yellow")
                return False

            login(token, add_to_git_credential=False)
            logger.info("HuggingFace authentication successful")
            console.print("HuggingFace authentication successful", style="green")
            return True

        except Exception as e:
            logger.error(f"HuggingFace authentication failed: {e}")
            console.print(f"HuggingFace authentication failed: {e}", style="red")
            return False

    def _download_model(self, skip_if_exists: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Download or load the HuggingFace model and tokenizer.
        
        Args:
            skip_if_exists (bool): If True, use local model if available
            
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Model and tokenizer
        """
        try:
            logger.info(f"Downloading model: {self.model_name}")
            console.print(f"Downloading model: {self.model_name}", style="blue")
            
            # Check if we have a local model path
            local_model_path = Path(f"./models/{self.model_name.split('/')[-1]}")
            if skip_if_exists and local_model_path.exists():
                logger.info(f"Using local model at: {local_model_path}")
                console.print(f"Using local model at: {local_model_path}", style="green")
                model = AutoModelForCausalLM.from_pretrained(
                    str(local_model_path),
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    str(local_model_path),
                    trust_remote_code=True
                )
                return model, tokenizer
            
            # Otherwise download from HuggingFace
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            logger.info("Model download completed successfully")
            console.print("Model download completed successfully", style="green")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            console.print(f"Failed to download model: {e}", style="red")
            raise RuntimeError(f"Model download failed: {e}")

    def convert_to_int4(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> Path:
        """
        Convert HuggingFace model to OpenVINO IR format with INT4 quantization using optimum-cli.
        
        This method uses optimum-cli directly on the local model path to avoid loading
        the entire model into memory, which is more memory-efficient for large models.
        
        Args:
            model (AutoModelForCausalLM): HuggingFace model to convert (not used in this approach)
            tokenizer (AutoTokenizer): Associated tokenizer (not used in this approach)
            
        Returns:
            Path: Path to the converted model directory
            
        Raises:
            RuntimeError: If conversion fails
        """
        logger.info("Starting INT4 conversion to OpenVINO IR format using optimum-cli")
        console.print("Starting INT4 conversion using optimum-cli...", style="blue")
        
        try:
            output_path = self.cache_dir / f"{self.model_name.split('/')[-1]}-int4-genai"
            
            # Clean up existing output directory
            if output_path.exists():
                import shutil
                shutil.rmtree(output_path)
            output_path.mkdir(exist_ok=True)
            
            # Use local model path for conversion
            local_model_path = self.cache_dir / self.model_name.split("/")[-1]
            if not local_model_path.exists():
                raise RuntimeError(f"Local model path {local_model_path} not found")
            
            logger.info(f"Using local model at: {local_model_path}")
            console.print(f"Using local model at: {local_model_path}", style="green")
            
            # Run optimum-cli to convert model directly from local path
            # This avoids loading the model into memory
            cmd = [
                "optimum-cli", "export", "openvino",
                "--model", str(local_model_path),
                "--task", "text-generation",
                "--weight-format", "int4",
                str(output_path)
            ]
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            console.print(f"Executing: {' '.join(cmd)}", style="blue")
            
            # Run the command with proper error handling
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("optimum-cli conversion completed")
            console.print("optimum-cli conversion completed", style="green")
            
            # Verify the conversion created the expected files
            model_xml = output_path / "openvino_model.xml"
            model_bin = output_path / "openvino_model.bin"
            
            if not model_xml.exists() or not model_bin.exists():
                raise RuntimeError("OpenVINO IR files not created by optimum-cli")
            
            logger.info("OpenVINO IR conversion completed successfully")
            console.print(f"Model saved to: {output_path}", style="green")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"optimum-cli conversion failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            console.print(f"optimum-cli conversion failed: {e}", style="red")
            raise RuntimeError(f"optimum-cli conversion failed: {e}")
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            console.print(f"Conversion failed: {e}", style="red")
            raise RuntimeError(f"Conversion failed: {e}")

    def _create_minimal_tokenizer_xml(self, tokenizer: AutoTokenizer, output_path: Path) -> None:
        """
        Create a minimal openvino_tokenizer.xml file for OpenVINO GenAI compatibility.

        Args:
            tokenizer (AutoTokenizer): HuggingFace tokenizer
            output_path (Path): Directory to save the tokenizer XML
        """
        try:
            logger.info("Creating minimal tokenizer XML")
            console.print("Creating minimal tokenizer XML...", style="blue")

            tokenizer_xml_content = """<?xml version="1.0"?>
                <net name="tokenizer" version="11">
                    <layers>
                        <layer id="0" name="input" type="Parameter" version="opset1">
                            <data shape="1" element_type="string"/>
                            <output>
                                <port id="0" precision="string">
                                    <dim>1</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="1" name="output" type="Result" version="opset1">
                            <input>
                                <port id="0" precision="i64">
                                    <dim>1</dim>
                                    <dim>128</dim>
                                </port>
                            </input>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
                    </edges>
                </net>"""

            tokenizer_xml_path = output_path / "openvino_tokenizer.xml"
            with open(tokenizer_xml_path, "w") as f:
                f.write(tokenizer_xml_content)

            logger.info("Minimal tokenizer XML created")
            console.print("Minimal tokenizer XML created", style="green")

        except Exception as e:
            logger.error(f"Failed to create tokenizer XML: {e}")
            console.print(f"Failed to create tokenizer XML: {e}", style="red")

    def _validate_genai_model(self, model_path: Path) -> bool:
        """
        Validate that the converted model is compatible with OpenVINO GenAI.

        Args:
            model_path (Path): Path to the converted model directory

        Returns:
            bool: True if model is valid, False otherwise
        """
        try:
            logger.info(f"Validating OpenVINO GenAI compatibility at {model_path}")
            console.print(f"Validating GenAI model at {model_path}...", style="blue")

            # Check OpenVINO IR format
            if not self._validate_openvino_ir_format(model_path):
                return False

            # Try loading with OpenVINO GenAI on Intel GPU (GPU.0)
            device = "GPU.0"  # Assumes Intel UHD Graphics is GPU.0
            logger.info(f"Testing model on device: {device}")
            console.print(f"Testing model on device: {device}", style="blue")
            pipeline = ov_genai.LLMPipeline(str(model_path), device)
            result = pipeline.generate("Hello, how are you?", max_length=10)

            logger.info(f"OpenVINO GenAI validation successful on {device}")
            console.print(f"OpenVINO GenAI validation successful on {device}", style="green")
            return True

        except Exception as e:
            logger.error(f"GenAI validation failed on {device}: {e}")
            console.print(f"GenAI validation failed on {device}: {e}", style="yellow")
            return False

    def get_model_info(self) -> str:
        """
        Get information about the current model conversion status.

        Returns:
            str: Human-readable status message
        """
        try:
            model_dir = self.cache_dir / f"{self.model_name.split('/')[-1]}-int4-genai"
            if not model_dir.exists():
                return "Model not converted yet"

            model_xml = model_dir / "openvino_model.xml"
            if model_xml.exists():
                return f"Model converted and available at {model_xml}"
            return "Model directory exists but no OpenVINO IR files found"

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return f"Error getting model info: {e}"

    def run_conversion_pipeline(self, skip_download: bool = False, skip_validation: bool = False) -> Path:
        """
        Execute the complete model conversion pipeline.
        
        Args:
            skip_download (bool): If True, skip download step and use local model
            skip_validation (bool): If True, skip validation step
            
        Returns:
            Path: Path to the converted model directory
            
        Raises:
            RuntimeError: If any step fails
        """
        try:
            logger.info("Starting OpenVINO GenAI model conversion pipeline")
            console.print("Starting GenAI model conversion pipeline...", style="bold blue")
            
            # Check hardware compatibility
            devices = self._check_hardware_compatibility()
            logger.info(f"Available devices: {devices}")
            console.print(f"Available devices: {devices}", style="green")
            
            # Authenticate with HuggingFace
            auth_success = self._authenticate_huggingface()
            if not auth_success:
                logger.warning("Continuing without HuggingFace authentication")
                console.print("Continuing without authentication", style="yellow")
            
            # Download or load model (only if not skipping download)
            if not skip_download:
                model, tokenizer = self._download_model(skip_if_exists=False)
            else:
                # Create dummy model and tokenizer for the conversion method
                # These won't actually be used since we're using optimum-cli directly
                model = None
                tokenizer = None
            
            # Convert to OpenVINO IR with INT4 quantization
            output_path = self.convert_to_int4(model, tokenizer)
            
            # Validate conversion
            if not skip_validation:
                logger.info("Validating converted model")
                if not self._validate_genai_model(output_path):
                    raise RuntimeError("Model validation failed")
            else:
                logger.info("Skipping validation step")
            
            logger.info("Conversion pipeline completed successfully")
            console.print("Conversion pipeline completed successfully", style="bold green")
            return output_path
            
        except Exception as e:
            logger.error(f"Conversion pipeline failed: {e}")
            console.print(f"Conversion pipeline failed: {e}", style="red")
            raise RuntimeError(f"Conversion pipeline failed: {e}")

    def _verify_input_names(self, model_path: Path) -> bool:
        """
        Verify that the converted model has correct input names.

        Args:
            model_path (Path): Path to the converted model directory

        Returns:
            bool: True if input names are correct, False otherwise
        """
        try:
            model_xml_path = model_path / "openvino_model.xml"
            if not model_xml_path.exists():
                logger.warning(f"Model XML not found at {model_xml_path}")
                return False

            core = ov.Core()
            model = core.read_model(str(model_xml_path))
            input_names = [input_node.get_any_name() for input_node in model.inputs]
            logger.info(f"Model input names: {input_names}")

            has_input_ids = any("input_ids" in name.lower() for name in input_names)
            has_attention_mask = any("attention_mask" in name.lower() for name in input_names)

            if has_input_ids and has_attention_mask:
                logger.info("Model has correct input names")
                console.print("Model input names verified", style="green")
                return True
            logger.warning(f"Model missing required input names: {input_names}")
            console.print(f"Input names verification failed: {input_names}", style="yellow")
            return False

        except Exception as e:
            logger.error(f"Error verifying input names: {e}")
            console.print(f"Error verifying input names: {e}", style="red")
            return False

    def _validate_openvino_ir_format(self, model_path: Path) -> bool:
        """
        Validate that the model directory contains valid OpenVINO IR format files.

        Args:
            model_path (Path): Path to the model directory

        Returns:
            bool: True if OpenVINO IR format is valid, False otherwise
        """
        try:
            model_xml = model_path / "openvino_model.xml"
            model_bin = model_path / "openvino_model.bin"

            if not model_xml.exists() or not model_bin.exists():
                logger.error("Missing OpenVINO IR files")
                console.print("Missing OpenVINO IR files", style="red")
                return False

            logger.info("OpenVINO IR format detected")
            console.print("OpenVINO IR format detected", style="green")
            return True

        except Exception as e:
            logger.error(f"OpenVINO IR format validation failed: {e}")
            console.print(f"OpenVINO IR format validation failed: {e}", style="red")
            return False


def main():
    """
    Main function for command-line usage of the model converter.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Llama-3.1-8B-Instruct to OpenVINO GenAI format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                python genai_model_converter.py
                python genai_model_converter.py --skip-download
                python genai_model_converter.py --skip-validation
                python genai_model_converter.py --verbose
            """
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading if model exists locally"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    try:
        converter = GenAIModelConverter(args.model_name)
        console.print("OpenVINO GenAI Model Conversion", style="bold blue")
        console.print(f"Model: {args.model_name}", style="blue")
        console.print(f"Skip download: {args.skip_download}", style="blue")
        console.print(f"Skip validation: {args.skip_validation}", style="blue")
        console.print("-" * 60, style="dim")

        output_path = converter.run_conversion_pipeline(
            skip_download=args.skip_download,
            skip_validation=args.skip_validation
        )

        console.print("-" * 60, style="dim")
        console.print("Conversion completed successfully!", style="bold green")
        console.print(f"Model location: {output_path}", style="green")
        console.print(f"Model info: {converter.get_model_info()}", style="green")
        logger.info(f"Conversion completed successfully: {output_path}")

    except KeyboardInterrupt:
        console.print("\nConversion interrupted by user", style="yellow")
        logger.info("Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"Conversion failed: {e}", style="red")
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()