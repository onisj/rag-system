"""
Pytest configuration file for the RAG system tests.

This file configures pytest to suppress known warnings from external libraries
that don't affect our test functionality, while preserving important warnings.
"""

import warnings
import pytest
import os

def pytest_configure(config):
    """Configure pytest to suppress specific warnings."""
    
    # Set environment variable to disable warnings only in test environment
    if os.getenv('ENVIRONMENT') != 'production':
        os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Suppress ALL SWIG-related deprecation warnings (more aggressive approach)
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="importlib._bootstrap"
    )
    
    # Suppress specific SWIG deprecation warnings from PyMuPDF/fitz
    warnings.filterwarnings(
        "ignore",
        message="builtin type SwigPyPacked has no __module__ attribute",
        category=DeprecationWarning
    )
    warnings.filterwarnings(
        "ignore", 
        message="builtin type SwigPyObject has no __module__ attribute",
        category=DeprecationWarning
    )
    warnings.filterwarnings(
        "ignore",
        message="builtin type swigvarlink has no __module__ attribute", 
        category=DeprecationWarning
    )
    
    # PyTorch flash attention warning (not critical for our use case)
    warnings.filterwarnings(
        "ignore",
        message="Torch was not compiled with flash attention",
        category=UserWarning
    )
    # Also suppress the specific flash attention warning from transformers
    warnings.filterwarnings(
        "ignore",
        message="1Torch was not compiled with flash attention",
        category=UserWarning
    )
    
    # Suppress distutils deprecation (will be removed in Python 3.12)
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="distutils"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    # Removed automatic unit marker addition to avoid warnings
    pass

def pytest_sessionstart(session):
    """Setup at the start of the test session to suppress warnings early."""
    # Suppress SWIG warnings at session start
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="importlib._bootstrap"
    )
    warnings.filterwarnings(
        "ignore",
        message="builtin type SwigPyPacked has no __module__ attribute",
        category=DeprecationWarning
    )
    warnings.filterwarnings(
        "ignore",
        message="builtin type SwigPyObject has no __module__ attribute", 
        category=DeprecationWarning
    )
    warnings.filterwarnings(
        "ignore",
        message="builtin type swigvarlink has no __module__ attribute",
        category=DeprecationWarning
    )

def pytest_runtest_setup(item):
    """Setup for each test to ensure warnings are suppressed."""
    # Only suppress specific warnings, not all warnings
    if os.getenv('ENVIRONMENT') != 'production':
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="distutils") 