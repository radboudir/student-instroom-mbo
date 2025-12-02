#!/usr/bin/env python
"""
Main entry point for MBO student enrollment predictions.
Usage: uv run main.py -y 2024 -w 4 -wf
"""

import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

# Import and run the individual MBO model
from scripts.models.individual_mbo import main as run_individual_mbo

if __name__ == "__main__":
    run_individual_mbo()
