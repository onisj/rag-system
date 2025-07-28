"""
Test suite for Performance Monitor

This module tests the performance monitoring functionality:
- Performance metrics collection
- Memory monitoring
- Timing measurements
- Performance reporting
- Resource tracking
"""

import sys
import os
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import psutil

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from rich.console import Console
from performance_monitor import PerformanceMonitor

console = Console()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

class TestPerformanceMonitor:
    """Test cases for performance monitoring"""
    
    def test_initialization(self):
        """Test PerformanceMonitor initialization"""
        monitor = PerformanceMonitor()
        assert monitor is not None
        assert hasattr(monitor, 'performance_data')
        assert hasattr(monitor, 'monitoring')
        assert hasattr(monitor, 'inference_times')
    
    def test_start_monitoring(self):
        """Test starting performance monitoring"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        assert monitor.monitoring is True
        assert monitor.monitor_thread is not None
        assert monitor.monitor_thread.is_alive()
    
    def test_stop_monitoring(self):
        """Test stopping performance monitoring"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        time.sleep(0.1)  # Small delay to ensure measurable time
        
        monitor.stop_monitoring()
        
        assert monitor.monitoring is False
    
    def test_performance_data_structure(self):
        """Test performance data structure"""
        monitor = PerformanceMonitor()
        
        assert hasattr(monitor, 'performance_data')
        assert isinstance(monitor.performance_data, dict)
        assert 'cpu_usage' in monitor.performance_data
        assert 'memory_usage' in monitor.performance_data
        assert 'gpu_usage' in monitor.performance_data
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_performance_monitoring_mocked(self, mock_virtual_memory, mock_cpu_percent):
        """Test performance monitoring with mocked psutil"""
        mock_cpu_percent.return_value = 25.5
        mock_virtual_memory.return_value = Mock(percent=60.0)
        
        monitor = PerformanceMonitor()
        
        # Test that performance data is accessible
        assert monitor.performance_data['cpu_usage'] == 0  # Initial value
        assert monitor.performance_data['memory_usage'] == 0  # Initial value
    
    def test_inference_time_recording(self):
        """Test inference time recording"""
        monitor = PerformanceMonitor()
        
        # Record some inference times
        monitor.record_inference_time(0.1)
        monitor.record_inference_time(0.2)
        monitor.record_inference_time(0.15)
        
        assert len(monitor.inference_times) == 3
        assert monitor.inference_times == [0.1, 0.2, 0.15]
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        monitor = PerformanceMonitor()
        
        # Add some test data
        monitor.performance_data['cpu_usage'] = 25.5
        monitor.performance_data['memory_usage'] = 60.0
        monitor.record_inference_time(0.1)
        monitor.record_inference_time(0.2)
        
        summary = monitor.get_performance_summary()
        
        assert isinstance(summary, dict)
        assert 'cpu_usage_percent' in summary
        assert 'memory_usage_percent' in summary
        assert 'inference_latency_ms' in summary
        assert 'throughput_tokens_per_second' in summary
    
    def test_display_performance_table(self):
        """Test performance table display"""
        monitor = PerformanceMonitor()
        
        # Add some test data
        monitor.performance_data['cpu_usage'] = 25.5
        monitor.performance_data['memory_usage'] = 60.0
        monitor.record_inference_time(0.1)
        
        table = monitor.display_performance_table()
        
        assert table is not None
        # Rich table should have rows
        assert hasattr(table, 'rows')
    
    def test_live_monitor(self):
        """Test live monitoring functionality"""
        monitor = PerformanceMonitor()
        
        # Test that live monitor can be called (but don't run it for long)
        # This is a basic test to ensure the method exists and doesn't crash
        assert hasattr(monitor, 'live_monitor')
        assert callable(monitor.live_monitor)
    
    def test_create_performance_monitor(self):
        """Test factory function for creating performance monitor"""
        from performance_monitor import create_performance_monitor
        
        monitor = create_performance_monitor()
        
        assert isinstance(monitor, PerformanceMonitor)
        assert monitor is not None

def test_performance_monitor_integration():
    """Integration test for performance monitoring"""
    monitor = PerformanceMonitor()
    
    # Test complete workflow
    monitor.start_monitoring()
    time.sleep(0.1)
    monitor.stop_monitoring()
    
    # Verify monitoring state
    assert monitor.monitoring is False
    
    # Test performance data structure
    assert isinstance(monitor.performance_data, dict)
    assert 'cpu_usage' in monitor.performance_data
    assert 'memory_usage' in monitor.performance_data
    
    # Test inference time recording
    monitor.record_inference_time(0.1)
    assert len(monitor.inference_times) > 0
    
    # Test performance summary
    summary = monitor.get_performance_summary()
    assert isinstance(summary, dict)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 