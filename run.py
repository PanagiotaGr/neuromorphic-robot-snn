#!/usr/bin/env python3
"""
Simple launcher script for neuromorphic robot experiments.

Usage:
  python run.py                           # Run with default config
  python run.py --config configs/deep_snn.yaml  # Use specific config
  python run.py --train.epochs 30         # Override parameters
  python run.py --generate-dataset        # Only generate data
  python run.py --visualize              # Just create visualizations
"""

import sys
import subprocess


def main():
    # Forward all arguments to main_enhanced.py
    cmd = [sys.executable, "main_enhanced.py"] + sys.argv[1:]
    print("Running: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
