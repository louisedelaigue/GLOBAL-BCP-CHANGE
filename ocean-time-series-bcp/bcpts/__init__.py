"""Shared functions for the ocean time-series analysis of the biological carbon pump."""
import sys
from pathlib import Path

# make config.py importable when scripts are run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
