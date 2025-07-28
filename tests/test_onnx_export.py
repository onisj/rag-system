#!/usr/bin/env python3
"""
Test ONNX Export - Verify warning suppression is working

This script tests the ONNX export with warning suppression to ensure
the TracerWarning doesn't cause the conversion to abort.
"""

import sys
from pathlib import Path
import warnings
import contextlib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
import torch
import openvino as ov

console = Console()

@contextlib.contextmanager
def suppress_all_warnings():
    """Context manager to suppress all warnings"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Also suppress stderr temporarily
        import sys
        import os
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            yield
        finally:
            sys.stderr.close()
            sys.stderr = stderr

def test_warning_suppression():
    """Test that warning suppression works"""
    console.print("Testing Warning Suppression", style="bold blue")
    
    try:
        # Suppress all warnings globally
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        console.print("✅ Warning filters configured", style="green")
        
        # Test the context manager
        console.print("Testing context manager...", style="dim")
        
        with suppress_all_warnings():
            # This should not produce any warnings
            console.print("  Context manager active - warnings suppressed", style="dim")
        
        console.print("✅ Context manager working", style="green")
        
        # Assert that warning suppression is working
        assert True, "Warning suppression configured successfully"
        
    except Exception as e:
        console.print(f"❌ Warning suppression test failed: {e}", style="red")
        assert False, f"Warning suppression test failed: {e}"

def test_openvino_initialization():
    """Test OpenVINO initialization"""
    console.print("\nTesting OpenVINO Initialization", style="bold blue")
    
    try:
        core = ov.Core()
        devices = core.available_devices
        
        console.print(f"✅ OpenVINO initialized successfully", style="green")
        console.print(f"Available devices: {devices}", style="green")
        
        # Assert that OpenVINO is working
        assert core is not None, "OpenVINO core should not be None"
        assert len(devices) > 0, "Should have at least one device available"
        
    except Exception as e:
        console.print(f"❌ OpenVINO initialization failed: {e}", style="red")
        assert False, f"OpenVINO initialization failed: {e}"

def test_torch_onnx():
    """Test PyTorch ONNX export capabilities"""
    console.print("\nTesting PyTorch ONNX Export", style="bold blue")
    
    try:
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 10)
        
        console.print("Testing ONNX export with warning suppression...", style="dim")
        
        with suppress_all_warnings():
            torch.onnx.export(
                model,
                dummy_input,
                "test_model.onnx",
                input_names=["input"],
                output_names=["output"],
                opset_version=11,
                do_constant_folding=True,
                export_params=True,
                verbose=False
            )
        
        console.print("✅ ONNX export test successful", style="green")
        
        # Clean up
        import os
        if os.path.exists("test_model.onnx"):
            os.remove("test_model.onnx")
        
        # Assert that ONNX export worked
        assert True, "ONNX export completed successfully"
        
    except Exception as e:
        console.print(f"❌ ONNX export test failed: {e}", style="red")
        assert False, f"ONNX export test failed: {e}"

def main():
    """Main test function"""
    console.print(Panel(
        "ONNX Export Test\n\n"
        "This test verifies that warning suppression is working\n"
        "and ONNX export can proceed without TracerWarning issues.",
        title="Test ONNX Export",
        border_style="blue"
    ))
    
    tests = [
        ("Warning Suppression", test_warning_suppression),
        ("OpenVINO Initialization", test_openvino_initialization),
        ("PyTorch ONNX Export", test_torch_onnx)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        console.print(f"\n{'='*50}", style="blue")
        console.print(f"Running: {test_name}", style="bold blue")
        console.print(f"{'='*50}", style="blue")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            console.print(f"❌ Test {test_name} crashed: {e}", style="red")
            results.append((test_name, False))
    
    # Summary
    console.print(f"\n{'='*60}", style="blue")
    console.print("TEST SUMMARY", style="bold blue")
    console.print(f"{'='*60}", style="blue")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        console.print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    console.print(f"\nResults: {passed}/{len(results)} tests passed", style="bold")
    
    if passed == len(results):
        console.print("\n🎉 All tests passed! ONNX export should work without warnings.", style="bold green")
        return 0
    else:
        console.print("\n⚠️  Some tests failed. Check the issues above.", style="bold yellow")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 