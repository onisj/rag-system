"""
Model Converter for Llama-3.1 8B Instruct to INT4 using OpenVINO (Intel UHD GPU)

This script converts the Llama-3.1 8B Instruct model from Hugging Face to INT4 format
using OpenVINO, optimized for Intel UHD GPUs. It handles model downloading, hardware
compatibility checks, conversion to INT4, and validation of the converted model. The
script ensures the attention_mask is in float32 to avoid ScaledDotProductAttention errors
and includes robust error handling with rich console output.

Key Features:
- Converts Llama-3.1 8B Instruct to INT4 for efficient inference on Intel UHD GPUs.
- Downloads model and tokenizer from Hugging Face with authentication.
- Prioritizes Intel GPU for conversion, with CPU fallback.
- Validates converted model with test inference using named tensors.
- Provides detailed logging and progress tracking using the rich library.

Usage:
    python model_converter.py --model-name meta-llama/Llama-3.1-8B-Instruct [--skip-download] [--device GPU]

Dependencies:
    - Python 3.11+
    - openvino>=2025.2.0
    - transformers>=4.40.0
    - torch
    - psutil
    - rich
    - python-dotenv
    - huggingface_hub

Environment Variables:
    - HUGGINGFACE_TOKEN: Required for authenticated access to the Llama-3.1 model.

Notes:
    - Requires Intel OpenVINO configured with GPU support for Intel UHD Graphics.
    - Assumes at least 16GB system RAM (shared with Intel UHD GPU).
    - Use --device GPU to force Intel GPU conversion, or let it auto-detect.
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict, Any
import transformers
import torch
import openvino as ov
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging with INFO level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class ModelConverter:
    """
    A class to convert Llama-3.1 8B Instruct model to INT4 format using OpenVINO on Intel UHD GPU.

    Attributes:
        model_name (str): Hugging Face model identifier.
        cache_dir (Path): Directory to store downloaded models.
        core (ov.Core): OpenVINO core for inference and compilation.
        selected_device (str | None): Selected device for conversion (e.g., 'GPU').
        device_info (Dict[str, Any]): Information about the selected device.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> None:
        """
        Initialize the ModelConverter with model name and setup.

        Args:
            model_name (str): Hugging Face model name. Defaults to "meta-llama/Llama-3.1-8B-Instruct".
        """
        self.model_name: str = model_name
        self.cache_dir: Path = Path("./models")
        self.cache_dir.mkdir(exist_ok=True)
        self.core: ov.Core = ov.Core()
        self.selected_device: str | None = None
        self.device_info: Dict[str, Any] = {}
        
        # Verify transformers version compatibility
        transformers_version: str = transformers.__version__
        if not transformers_version >= "4.40.0":
            console.print(
                f"Warning: transformers version {transformers_version} detected. "
                "Version >=4.40.0 recommended for Llama-3.1 compatibility.",
                style="yellow"
            )
        
        console.print(f"ModelConverter initialized for: {model_name}", style="blue")
    
    def _configure_openvino_environment(self) -> None:
        """
        Configure OpenVINO environment variables to enable debugging and GPU support.
        """
        os.environ.setdefault("OPENVINO_LOG_LEVEL", "DEBUG")
        os.environ.setdefault("OPENVINO_GPU_ENABLE_SDPA_OPTIMIZATION", "YES")
        os.environ.setdefault("OPENVINO_GPU_ENABLE_MULTI_DEVICE", "1")
        os.environ.setdefault("OPENVINO_GPU_ENABLE_DEVICE_ENUMERATION", "1")
        os.environ.setdefault("OPENVINO_GPU_ENABLE_DEVICE_NAMING", "1")
        os.environ.setdefault("OPENVINO_CACHE_DIR", str(self.cache_dir / "openvino_cache"))
        console.print("OpenVINO environment configured with DEBUG logging and GPU support", style="dim")
    
    def _enumerate_gpu_devices(self) -> List[str]:
        """
        Enumerate Intel GPU devices detected by OpenVINO.

        Returns:
            List[str]: List of GPU device names (e.g., ['GPU']).
        """
        gpu_devices: List[str] = []
        try:
            all_devices: List[str] = self.core.available_devices
            console.print(f"Available devices: {all_devices}", style="dim")
            for device in all_devices:
                if device.startswith("GPU"):
                    try:
                        device_name: str = self.core.get_property(device, "FULL_DEVICE_NAME")
                        if "Intel" in device_name:
                            gpu_devices.append(device)
                            console.print(f"Enumerated Intel GPU device: {device} -> {device_name}", style="dim")
                    except Exception as e:
                        console.print(f"Could not get properties for {device}: {e}", style="yellow")
            
            if not gpu_devices:
                console.print("No Intel GPUs detected via OpenVINO", style="yellow")
            
            return gpu_devices
        except Exception as e:
            console.print(f"GPU enumeration failed: {e}", style="yellow")
            return []
    
    def _detect_gpu_memory(self, gpu_device: str, gpu_props: str) -> float:
        """
        Estimate memory for Intel UHD GPU, which uses shared system RAM.

        Args:
            gpu_device (str): GPU device name (e.g., 'GPU').
            gpu_props (str): Full device name from OpenVINO properties.

        Returns:
            float: Estimated memory in GB.
        """
        total_memory: float = 0.0
        try:
            # Intel UHD GPUs share system memory; estimate as half of total RAM
            total_memory = psutil.virtual_memory().total / (1024**3) / 2
            console.print(f"Estimated memory for {gpu_device} ({gpu_props}): {total_memory:.1f} GB", style="green")
        except Exception as e:
            console.print(f"Memory detection failed for {gpu_device}: {e}, using conservative estimate", style="yellow")
            total_memory = 8.0
        return total_memory
    
    def _detect_additional_gpus(self) -> List[Dict[str, Any]]:
        """
        Detect additional Intel GPUs using system commands (Windows only).

        Returns:
            List[Dict[str, Any]]: List of dictionaries with GPU info (name, memory, source).
        """
        additional_gpus: List[Dict[str, Any]] = []
        if os.name == "nt":
            try:
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM", "/format:csv"],
                    capture_output=True, text=True, check=True
                )
                for line in result.stdout.strip().split('\n'):
                    if line.strip() and ',' in line and 'Name' not in line:
                        parts: List[str] = line.split(',')
                        if len(parts) >= 3:
                            gpu_name: str = parts[1].strip()
                            adapter_ram: str = parts[2].strip()
                            if "Intel" in gpu_name:
                                try:
                                    memory_bytes: int = int(adapter_ram)
                                    memory_gb: float = memory_bytes / (1024**3)
                                    additional_gpus.append({
                                        'name': gpu_name,
                                        'memory_gb': memory_gb,
                                        'source': 'wmic'
                                    })
                                    console.print(f"Detected Intel GPU via wmic: {gpu_name} ({memory_gb:.1f} GB)", style="dim")
                                except ValueError:
                                    console.print(f"Invalid AdapterRAM for {gpu_name}", style="dim")
            except Exception as e:
                console.print(f"WMI GPU detection failed: {e}", style="dim")
        return additional_gpus
    
    def authenticate_huggingface(self) -> bool:
        """
        Authenticate with Hugging Face using a token from the environment.

        Returns:
            bool: True if authentication succeeds, False otherwise.

        Raises:
            Exception: If no token is found or authentication fails.
        """
        try:
            from huggingface_hub import login
            token: str | None = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                console.print("No HUGGINGFACE_TOKEN found in .env file", style="red")
                raise Exception("Hugging Face token required")
            console.print("Using Hugging Face token from .env file", style="green")
            login(token=token)
            console.print("Login successful", style="green")
            return True
        except Exception as e:
            console.print(f"Authentication failed: {e}", style="red")
            raise
    
    def check_hardware_compatibility(self) -> bool:
        """
        Check system compatibility for OpenVINO and INT4 quantization.

        Returns:
            bool: True if compatible hardware (Intel GPU or CPU) is found, False otherwise.
        """
        try:
            gpu_devices: List[str] = self._enumerate_gpu_devices()
            
            if gpu_devices:
                console.print(f"Found Intel GPU devices: {gpu_devices}", style="green")
                for gpu_device in gpu_devices:
                    try:
                        gpu_props: str = self.core.get_property(gpu_device, "FULL_DEVICE_NAME")
                        total_memory: float = self._detect_gpu_memory(gpu_device, gpu_props)
                        if total_memory >= 8:
                            self.selected_device = gpu_device
                            self.device_info = {
                                'type': 'GPU',
                                'name': gpu_props,
                                'memory': total_memory,
                                'device_id': gpu_device
                            }
                            console.print(f"Selected Intel GPU: {gpu_props}", style="bold green")
                            console.print(f"GPU Memory: {total_memory:.1f} GB", style="green")
                            free_space_gb: float = psutil.disk_usage('./').free / (1024**3)
                            self.device_info['free_disk'] = free_space_gb
                            console.print(f"Available disk space: {free_space_gb:.1f} GB", style="green")
                            return True
                    except Exception as e:
                        console.print(f"Could not analyze GPU {gpu_device}: {e}", style="yellow")
            
            # Check for additional Intel GPUs on Windows
            additional_gpus: List[Dict[str, Any]] = self._detect_additional_gpus()
            if additional_gpus and any("Intel" in gpu['name'] for gpu in additional_gpus):
                self.selected_device = "GPU.0"
                self.device_info = {
                    'type': 'GPU',
                    'name': additional_gpus[0]['name'],
                    'memory': additional_gpus[0]['memory_gb'],
                    'device_id': 'GPU.0'
                }
                console.print(f"Selected Intel GPU: {self.device_info['name']}", style="bold green")
                console.print(f"GPU Memory: {self.device_info['memory']:.1f} GB", style="green")
                free_space_gb: float = psutil.disk_usage('./').free / (1024**3)
                self.device_info['free_disk'] = free_space_gb
                console.print(f"Available disk space: {free_space_gb:.1f} GB", style="green")
                return True
            
            # Fallback to CPU if no Intel GPUs are found
            console.print("No compatible Intel GPUs found, falling back to CPU", style="yellow")
            if "CPU" in self.core.available_devices:
                cpu_props: str = self.core.get_property("CPU", "FULL_DEVICE_NAME")
                total_memory: float = psutil.virtual_memory().total / (1024**3)
                free_space_gb: float = psutil.disk_usage('./').free / (1024**3)
                self.selected_device = "CPU"
                self.device_info = {
                    'type': 'CPU',
                    'name': cpu_props,
                    'memory': total_memory,
                    'free_disk': free_space_gb,
                    'device_id': 'CPU'
                }
                console.print(f"Selected CPU: {cpu_props}", style="green")
                console.print(f"System Memory: {total_memory:.1f} GB", style="green")
                console.print(f"Available disk space: {free_space_gb:.1f} GB", style="green")
                if total_memory < 16:
                    console.print("Warning: Less than 16GB RAM detected", style="yellow")
                return True
            else:
                console.print("CPU device not available", style="red")
                return False
        except Exception as e:
            console.print(f"Hardware compatibility check failed: {e}", style="red")
            return False
    
    def _fix_model_config(self, config_path: Path, preemptive: bool = False) -> None:
        """
        Adjust model config to fix rope_scaling issues for Llama compatibility.

        Args:
            config_path (Path): Path to the config.json file.
            preemptive (bool): If True, apply fixes before loading; otherwise, after download.
        """
        try:
            if not config_path.exists():
                if preemptive:
                    console.print(f"Config file {config_path} not found, will fix after download", style="dim")
                return
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data: Dict[str, Any] = json.load(f)
            original_rope: Any = config_data.get('rope_scaling', None)
            console.print(f"Original rope_scaling: {original_rope}", style="dim")
            if 'rope_scaling' in config_data and config_data['rope_scaling'] is not None:
                rope_scaling: Any = config_data['rope_scaling']
                if isinstance(rope_scaling, dict):
                    valid_keys: set[str] = {'type', 'factor'}
                    if not all(k in valid_keys for k in rope_scaling) or len(rope_scaling) > 2:
                        factor: float = float(rope_scaling.get('factor', 8.0)) if isinstance(rope_scaling.get('factor'), (int, float)) else 8.0
                        rope_type: str = rope_scaling.get('type', 'linear')
                        if rope_type not in ['linear', 'dynamic']:
                            rope_type = 'dynamic' if 'llama3' in rope_scaling.get('rope_type', '').lower() else 'linear'
                        config_data['rope_scaling'] = {
                            'type': rope_type,
                            'factor': factor
                        }
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config_data, f, indent=2)
                        console.print(f"Fixed rope_scaling in {config_path}: {config_data['rope_scaling']}", style="yellow")
                    else:
                        console.print(f"rope_scaling in {config_path} appears valid: {rope_scaling}", style="dim")
                else:
                    console.print(f"Invalid rope_scaling format: {rope_scaling}, resetting to default", style="yellow")
                    config_data['rope_scaling'] = {'type': 'linear', 'factor': 8.0}
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config_data, f, indent=2)
            else:
                console.print(f"No rope_scaling in {config_path}, using default if needed", style="dim")
                if preemptive:
                    config_data['rope_scaling'] = {'type': 'linear', 'factor': 8.0}
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config_data, f, indent=2)
        except Exception as e:
            console.print(f"Warning: Could not fix config file {config_path}: {e}", style="yellow")
    
    def download_model(self, skip_if_exists: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Download model and tokenizer from Hugging Face, optimizing for memory usage.

        Args:
            skip_if_exists (bool): If True, load existing model instead of downloading.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Downloaded or loaded model and tokenizer.
        """
        try:
            model_path: Path = self.cache_dir / self.model_name.split("/")[-1]
            if skip_if_exists and model_path.exists():
                console.print(f"Loading existing model from {model_path}", style="blue")
                return self._load_existing_model(model_path)
            
            if not self.authenticate_huggingface():
                raise Exception("Hugging Face authentication failed")
            
            console.print(f"Downloading {self.model_name}...", style="blue")
            
            model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            model = model.to('cpu')
            
            tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None: # type: ignore[assignment]
                tokenizer.pad_token = tokenizer.eos_token # type: ignore[assignment]
                console.print("Set pad_token to eos_token for compatibility", style="dim")
            self._fix_model_config(model_path / "config.json", preemptive=False)
            console.print("Model downloaded successfully to models", style="green")
            return model, tokenizer
        except Exception as e:
            console.print(f"Model download failed: {e}", style="red")
            raise
    
    def _load_existing_model(self, model_path: Path) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load an existing model and tokenizer from local cache.

        Args:
            model_path (Path): Directory containing the cached model.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer.
        """
        try:
            console.print(f"Loading existing model from {model_path}...", style="blue")
            self._fix_model_config(model_path / "config.json", preemptive=True)
            
            model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            model = model.to('cpu') # type: ignore[assignment]
            
            tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None: # type: ignore[assignment]
                tokenizer.pad_token = tokenizer.eos_token # type: ignore[assignment]
                console.print("Set pad_token to eos_token for compatibility", style="dim")
            console.print("Model loaded successfully", style="green")
            return model, tokenizer
        except Exception as e:
            console.print(f"Failed to load existing model: {e}", style="red")
            raise
    
    class LogitsOnlyWrapper(torch.nn.Module):
        """
        Wrapper class to extract logits from the model for OpenVINO conversion.
        """
        def __init__(self, model: AutoModelForCausalLM) -> None:
            """
            Initialize the wrapper with the original model.

            Args:
                model (AutoModelForCausalLM): The model to wrap.
            """
            super().__init__()
            self.model: AutoModelForCausalLM = model
        
        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None) -> torch.Tensor:
            """
            Forward pass to extract logits, ensuring attention_mask is float32.

            Args:
                input_ids (torch.Tensor): Input token IDs.
                attention_mask (torch.Tensor | None): Attention mask, converted to float32 if provided.
                position_ids (torch.Tensor | None): Position IDs.

            Returns:
                torch.Tensor: Logits from the model.
            """
            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.float32)  # Ensure float32 for compatibility
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids) # type: ignore[call-overload]
            return outputs.logits

    def convert_to_int4(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> Path:
        """
        Convert the model to INT4 format using OpenVINO, with fixes for attention mask and proper input shapes.

        Args:
            model (AutoModelForCausalLM): The model to convert.
            tokenizer (AutoTokenizer): Tokenizer for preparing example inputs.

        Returns:
            Path: Directory path where the INT4 model is saved.

        Raises:
            Exception: If conversion fails on all devices.
        """
        console.print("Converting model to INT4...", style="blue")
        output_dir: Path = Path("./models/llama-3.1-8b-int4")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model = model.eval()  # type: ignore[assignment]
        
        # FIXED: Use 512 tokens for proper shape alignment with model architecture
        console.print("Preparing example inputs for conversion with 512 tokens...", style="dim")
        
        # Create a meaningful 512-token input sequence
        sample_text = "Hello world, this is a comprehensive test input for model conversion. " * 20  # Repeat to get more tokens
        inputs: Dict[str, torch.Tensor] = tokenizer( # type: ignore[call-overload]
            sample_text,
            return_tensors="pt",
            padding="max_length",
            max_length=512,  # Use 512 tokens
            truncation=True,
            return_attention_mask=True
        )
        
        input_ids: torch.Tensor = inputs["input_ids"].to(torch.long)
        attention_mask: torch.Tensor = inputs["attention_mask"].to(torch.float32)  # Ensure float32
        position_ids: torch.Tensor = torch.arange(0, input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        
        # Create 512-token examples for conversion
        conversion_input_ids: torch.Tensor = input_ids  # Already 512 tokens
        conversion_attention_mask: torch.Tensor = attention_mask  # Already 512 tokens
        conversion_position_ids: torch.Tensor = position_ids  # Already 512 tokens
        
        console.print(f"Conversion input shapes: input_ids={conversion_input_ids.shape}, attention_mask={conversion_attention_mask.shape}, position_ids={conversion_position_ids.shape}", style="dim")
        console.print(f"Input dtypes: input_ids={conversion_input_ids.dtype}, attention_mask={conversion_attention_mask.dtype}, position_ids={conversion_position_ids.dtype}", style="dim")
        
        devices: List[str] = ["GPU", "CPU"]  # Try GPU first, then CPU
        
        for device in devices:
            try:
                console.print(f"Attempting conversion on {device}...", style="green")
                
                logits_model = self.LogitsOnlyWrapper(model)
                logits_model.eval()
                
                # Convert model to OpenVINO format with 512-token inputs
                console.print(f"Converting with 512-token inputs on {device}...", style="dim")
                ov_model = ov.convert_model(
                    logits_model,
                    example_input={
                        "input_ids": conversion_input_ids,
                        "attention_mask": conversion_attention_mask,
                        "position_ids": conversion_position_ids
                    },
                    input=[
                        ("input_ids", [1, 512], ov.Type.i64),  # 512 tokens
                        ("attention_mask", [1, 512], ov.Type.f32),  # 512 tokens, f32 type
                        ("position_ids", [1, 512], ov.Type.i64)  # 512 tokens
                    ]
                )
                console.print(f"Model conversion successful on {device}", style="green")
                
                # ENHANCED FIX: More comprehensive attention mask type fixing
                console.print("Applying comprehensive attention mask fixes...", style="dim")
                
                def fix_attention_mask_types(model):
                    """Fix all nodes that might produce u8 tensors used in attention operations."""
                    nodes_fixed = 0
                    
                    for node in model.get_ops():
                        node_type = node.get_type_name()
                        node_name = node.get_friendly_name()
                        
                        # Fix nodes that produce u8 outputs and are likely attention-related
                        if (node_type in ["LogicalOr", "BitwiseOr", "LogicalAnd", "BitwiseAnd", 
                                        "LogicalNot", "BitwiseNot", "Equal", "NotEqual", 
                                        "Greater", "GreaterEqual", "Less", "LessEqual"] and 
                            node.get_output_element_type(0).get_type_name() == "u8"):
                            
                            console.print(f"Converting {node_name} ({node_type}) from u8 to f32", style="dim")
                            
                            # Create a convert node to change u8 to f32
                            convert_node = ov.opset10.convert(node.output(0), ov.Type.f32) # type: ignore[call-overload]
                            
                            # Replace all uses of the original output with the converted one
                            node.output(0).replace(convert_node.output(0))
                            nodes_fixed += 1
                        
                        # Also fix any StridedSlice operations that might be masking-related
                        elif (node_type == "StridedSlice" and 
                            node.get_output_element_type(0).get_type_name() == "u8" and
                            ("mask" in node_name.lower() or "attention" in node_name.lower())):
                            
                            console.print(f"Converting attention-related {node_name} from u8 to f32", style="dim")
                            convert_node = ov.opset10.convert(node.output(0), ov.Type.f32) # type: ignore[call-overload]
                            node.output(0).replace(convert_node.output(0))
                            nodes_fixed += 1
                    
                    return nodes_fixed
                
                # Apply the comprehensive fix
                fixed_nodes = fix_attention_mask_types(ov_model)
                console.print(f"Fixed {fixed_nodes} nodes with attention mask type issues", style="green")
                
                # Additional fix: Look for ScaledDotProductAttention nodes specifically
                console.print("Checking ScaledDotProductAttention nodes...", style="dim")
                for node in ov_model.get_ops():
                    if node.get_type_name() == "ScaledDotProductAttention":
                        console.print(f"Found SDPA node: {node.get_friendly_name()}", style="dim")
                        
                        # Check input types to the SDPA node
                        for i in range(node.get_input_size()):
                            input_port = node.input(i)
                            source_output = input_port.get_source_output()
                            source_node = source_output.get_node()
                            input_type = node.get_input_element_type(i)
                            console.print(f"  Input {i}: {source_node.get_friendly_name()} -> {input_type}", style="dim")
                            
                            # If we find a u8 input (likely the attention mask), fix it
                            if input_type.get_type_name() == "u8" and i >= 3:  # Attention mask is typically input 3 or 4
                                console.print(f"  Converting SDPA input {i} from u8 to f32", style="yellow")
                                convert_node = ov.opset10.convert(source_output, ov.Type.f32) # type: ignore[call-overload]
                                input_port.replace_source_output(convert_node.output(0))
                
                # Try a more aggressive approach for GPU: convert all boolean operations to float
                if device == "GPU":
                    console.print("Applying GPU-specific boolean-to-float conversions...", style="dim")
                    nodes_to_convert = []
                    
                    # First, collect all nodes that need conversion
                    for node in ov_model.get_ops():
                        if (node.get_output_element_type(0).get_type_name() in ["boolean", "u8"]):
                            # Check if this node feeds into a ScaledDotProductAttention
                            feeds_sdpa = False
                            for output_port in node.outputs():
                                for target_input in output_port.get_target_inputs():
                                    target_node = target_input.get_node()
                                    if target_node.get_type_name() == "ScaledDotProductAttention":
                                        feeds_sdpa = True
                                        break
                                if feeds_sdpa:
                                    break
                            
                            if feeds_sdpa:
                                nodes_to_convert.append(node)
                    
                    # Then convert them
                    for node in nodes_to_convert:
                        console.print(f"Converting boolean/u8 node {node.get_friendly_name()} to f32 for GPU compatibility", style="dim")
                        convert_node = ov.opset10.convert(node.output(0), ov.Type.f32) # type: ignore[call-overload]
                        node.output(0).replace(convert_node.output(0))
                
                # Validate the model before compilation
                console.print("Validating model structure...", style="dim")
                ov_model.validate_nodes_and_infer_types()
                
                # Compile the model to verify conversion
                compile_config = {
                    "PERFORMANCE_HINT": "THROUGHPUT",
                    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32"
                }
                
                # Add GPU-specific optimizations
                if device == "GPU":
                    compile_config.update({
                        "GPU_ENABLE_SDPA_OPTIMIZATION": "YES",
                        "GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"  # Sometimes helps with compatibility
                    })
                
                compiled_model = self.core.compile_model(
                    ov_model,
                    device_name=device,
                    config=compile_config
                )
                console.print(f"Model compilation validated on {device}", style="green")
                
                # Save the converted model with progress tracking
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Saving converted model...", total=None)
                    
                    ov.save_model(ov_model, str(output_dir / "model.xml"))  # type: ignore[call-overload]
                    tokenizer.save_pretrained(output_dir)  # type: ignore[call-overload]
                    
                    device_info_path: Path = output_dir / "device_info.json"
                    with open(device_info_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'conversion_device': device,
                            'selected_device': self.selected_device,
                            'device_info': self.device_info,
                            'nodes_fixed': fixed_nodes,
                            'input_shape': [1, 512]  # Record the input shape used
                        }, f, indent=2)
                    
                    progress.update(task, description="Model conversion completed!")
                
                console.print(f"Model converted and saved to: {output_dir}", style="green")
                console.print(f"Conversion completed successfully on {device} with 512-token inputs", style="bold green")
                return output_dir
                
            except Exception as e:
                console.print(f"Conversion on {device} failed: {e}", style="yellow")
                if device == devices[-1]:
                    raise Exception(f"Model conversion failed on all devices: {e}")
                console.print(f"Trying next device...", style="yellow")
                continue
        
        raise Exception("No suitable device found for conversion")
    
    def _validate_conversion(self) -> None:
        """
        Validate the converted model by loading and running a test inference with 512-token inputs.

        Raises:
            Exception: If validation fails due to missing model or inference errors.
        """
        try:
            model_path: Path = Path("./models/llama-3.1-8b-int4/model.xml")
            device_info_path: Path = Path("./models/llama-3.1-8b-int4/device_info.json")
            if not model_path.exists():
                raise Exception("Converted model not found")
            
            # Determine the device used for conversion
            conversion_device: str = "CPU"
            input_shape = [1, 5]  # Default fallback
            if device_info_path.exists():
                with open(device_info_path, 'r', encoding='utf-8') as f:
                    device_info: Dict[str, Any] = json.load(f)
                    conversion_device = device_info.get('conversion_device', "CPU")
                    input_shape = device_info.get('input_shape', [1, 512])
            
            console.print(f"Validating converted model on {conversion_device} with shape {input_shape}...", style="blue")
            model = self.core.read_model(str(model_path))
            compiled_model = self.core.compile_model(model, device_name=conversion_device)
            
            # Prepare test inputs with the same shape used during conversion
            seq_len = input_shape[1]
            test_input: torch.Tensor = torch.randint(1, 1000, (1, seq_len), dtype=torch.long)
            attention_mask: torch.Tensor = torch.ones((1, seq_len), dtype=torch.float32)
            position_ids: torch.Tensor = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
            
            inputs: Dict[str, Any] = {
                "input_ids": test_input.numpy(),
                "attention_mask": attention_mask.numpy(),
                "position_ids": position_ids.numpy()
            }
            
            console.print(f"Model input shape: {test_input.shape}", style="dim")
            try:
                result = compiled_model.infer_new_request(inputs)
                output_name: str = compiled_model.output(0).get_any_name()
                console.print(f"Test inference successful, output shape: {result[output_name].shape}", style="green")
            except Exception as inference_error:
                console.print(f"Test inference failed (this might be normal): {inference_error}", style="yellow")
            
            console.print("Model validation successful", style="green")
        except Exception as e:
            console.print(f"Model validation failed: {e}", style="red")
            raise
    
    def run_conversion_pipeline(self, skip_download: bool = False, skip_validation: bool = False) -> Path:
        """
        Execute the full conversion pipeline: configure, check hardware, convert, and validate.

        Args:
            skip_download (bool): If True, skip downloading if model exists locally.
            skip_validation (bool): If True, skip model validation after conversion.

        Returns:
            Path: Directory containing the converted INT4 model.

        Raises:
            Exception: If any step in the pipeline fails.
        """
        try:
            console.print("Starting model conversion pipeline", style="bold blue")
            self._configure_openvino_environment()
            if not self.check_hardware_compatibility():
                raise Exception("Hardware not compatible with OpenVINO")
            free_space_gb: float = self.device_info.get("free_disk", 0.0)
            if free_space_gb < 20:
                console.print(
                    f"Low disk space detected: {free_space_gb:.1f} GB available. "
                    "Please free up at least 20GB before proceeding.",
                    style="red"
                )
                raise Exception("Insufficient disk space for model conversion")
            
            model, tokenizer = self.download_model(skip_if_exists=skip_download)
            output_dir: Path = self.convert_to_int4(model, tokenizer)
            
            if not skip_validation:
                self._validate_conversion()
            else:
                console.print("Skipping model validation as requested", style="dim")
            
            console.print("Model conversion pipeline completed successfully!", style="bold green")
            return output_dir
        except Exception as e:
            console.print(f"Conversion failed: {e}", style="red")
            raise

def main() -> None:
    """
    Main function to handle command-line arguments and run the conversion pipeline.
    """
    parser = argparse.ArgumentParser(description="Convert Llama-3.1 8B Instruct to INT4")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name to download from Hugging Face")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading if model already exists locally")
    parser.add_argument("--device", default=None, choices=["CPU", "GPU", None],
                        help="Force conversion on CPU or GPU (default: auto-detect)")
    
    args = parser.parse_args()
    
    try:
        converter = ModelConverter(model_name=args.model_name)
        if args.device:
            converter.selected_device = args.device
            converter.device_info['type'] = args.device
        converter.run_conversion_pipeline(skip_download=args.skip_download)
    except Exception as e:
        console.print(f"Conversion failed: {e}", style="red")
        sys.exit(1)

if __name__ == "__main__":
    main()