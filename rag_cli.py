"""
RAG CLI - Main entry point for the RAG system
Command: rag_cli --query "..." → retrieve k chunks → stream answer with references.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure warnings before importing other modules
from src.warning_config import initialize_warnings
initialize_warnings()

from src.cli import cli

if __name__ == "__main__":
    cli() 