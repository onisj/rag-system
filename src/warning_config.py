"""
Warning Configuration Module - Intelligent Warning Management for RAG System

This module provides intelligent warning handling that balances system stability
with debugging capabilities across different environments. It implements a
sophisticated warning management system that:

1. Suppresses known harmless warnings from external libraries (PyMuPDF, PyTorch, etc.)
2. Preserves important warnings that indicate real issues or potential problems
3. Logs warnings appropriately for monitoring and debugging purposes
4. Configures different warning levels based on environment (development, testing, production)

The module addresses common warning sources in the RAG system:
- PyMuPDF SWIG warnings (harmless deprecation warnings)
- PyTorch flash attention warnings (feature availability notifications)
- Distutils deprecation warnings (Python version compatibility)
- Custom application warnings (development and debugging)

Key Features:
- Environment-specific warning configurations
- Custom warning logging with severity classification
- Production-ready warning monitoring capabilities
- Development-friendly error detection
- Testing-optimized clean output

Usage:
    The module automatically initializes based on the ENVIRONMENT variable.
    For manual configuration, call configure_warnings() with the desired environment.

Author: Segun Oni
Version: 1.0.0
"""

import warnings
import logging
import os
from typing import Optional

# Configure logging for warnings - creates a dedicated logger for warning management
logger = logging.getLogger(__name__)

def configure_warnings(environment: str = "development") -> None:
    """
    Configure warning handling based on the specified environment.
    
    This function serves as the main entry point for warning configuration,
    routing to the appropriate environment-specific configuration function.
    
    Args:
        environment: Target environment - "development", "testing", or "production"
                    Determines the level of warning suppression and logging detail
    """
    
    # Route to appropriate configuration based on environment
    if environment == "production":
        configure_production_warnings()
    elif environment == "testing":
        configure_testing_warnings()
    else:  # development - default fallback
        configure_development_warnings()

def configure_production_warnings() -> None:
    """
    Configure warnings for production environment.
    
    Production configuration focuses on stability and monitoring while suppressing
    known harmless warnings that could clutter logs. It maintains visibility into
    important issues while reducing noise from external library warnings.
    """
    
    # In production, we want to see most warnings but suppress known harmless ones
    
    # Suppress SWIG warnings from PyMuPDF (external library issue)
    # These are deprecation warnings from SWIG-generated code that don't affect functionality
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
    
    # Suppress PyTorch flash attention warning (not critical for functionality)
    # This warning appears when PyTorch wasn't compiled with flash attention support
    warnings.filterwarnings(
        "ignore",
        message="Torch was not compiled with flash attention",
        category=UserWarning
    )
    # Also suppress the specific flash attention warning from transformers library
    warnings.filterwarnings(
        "ignore",
        message="1Torch was not compiled with flash attention",
        category=UserWarning
    )
    
    # Suppress distutils deprecation (will be removed in Python 3.12)
    # This is a Python standard library deprecation that affects many packages
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="distutils"
    )
    
    # Set up warning logging for production monitoring
    # This captures all warnings and routes them through the logging system
    logging.captureWarnings(True)
    logger.info("Production warning configuration applied")

def configure_testing_warnings() -> None:
    """
    Configure warnings for testing environment.
    
    Testing configuration maximizes clean output by suppressing most warnings
    while maintaining the ability to detect critical issues. This ensures that
    test results are not cluttered with expected warnings from external libraries.
    """
    
    # In testing, suppress more warnings for clean output
    # Set environment variable to ignore all warnings at Python level
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Apply the same specific suppressions as production
    # This ensures consistency in warning handling across environments
    configure_production_warnings()
    
    logger.info("Testing warning configuration applied")

def configure_development_warnings() -> None:
    """
    Configure warnings for development environment.
    
    Development configuration prioritizes early detection of issues by treating
    warnings as errors for the application code. This helps developers catch
    potential problems before they reach production while still suppressing
    known harmless external library warnings.
    """
    
    # In development, show most warnings but suppress known harmless ones
    # Start with production configuration as base
    configure_production_warnings()
    
    # Show warnings as errors in development to catch issues early
    # This applies only to the application code, not external libraries
    warnings.filterwarnings("error", category=DeprecationWarning, module="src")
    warnings.filterwarnings("error", category=UserWarning, module="src")
    
    logger.info("Development warning configuration applied")

def log_warning(warning: Warning, filename: str, lineno: int, line: Optional[str] = None) -> None:
    """
    Custom warning handler that logs warnings with appropriate severity levels.
    
    This function provides intelligent warning logging that categorizes warnings
    by type and severity, enabling better monitoring and debugging capabilities.
    
    Args:
        warning: The warning object containing the warning message and category
        filename: Source filename where the warning originated
        lineno: Line number in the source file where the warning occurred
        line: Source line content (optional, for additional context)
    """
    
    # Determine warning severity based on warning type
    # This helps in filtering and prioritizing warnings in monitoring systems
    if isinstance(warning, DeprecationWarning):
        level = logging.WARNING
        prefix = "DEPRECATION"
    elif isinstance(warning, UserWarning):
        level = logging.INFO
        prefix = "USER"
    else:
        level = logging.WARNING
        prefix = "WARNING"
    
    # Log the warning with structured information
    # Includes warning type, message, location, and severity level
    logger.log(level, f"{prefix}: {warning} in {filename}:{lineno}")
    
    # In production, also log to monitoring systems
    # This provides integration points for external monitoring services
    if os.getenv('ENVIRONMENT') == 'production':
        # Here you could send to monitoring services like Sentry, DataDog, etc.
        # Implementation would depend on the specific monitoring infrastructure
        pass

def initialize_warnings() -> None:
    """
    Initialize warning configuration based on environment variable.
    
    This function automatically configures the warning system based on the
    ENVIRONMENT environment variable. It provides a convenient way to set up
    appropriate warning handling without manual configuration.
    """
    # Get environment from environment variable with development as default
    environment = os.getenv('ENVIRONMENT', 'development')
    configure_warnings(environment)

# Set up custom warning handler
# This replaces the default warning handler with our custom logging function
warnings.showwarning = log_warning 