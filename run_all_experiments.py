#!/usr/bin/env python3
"""
One-command runner for ALL novel experiments.

This script runs the comprehensive experimental suite testing all
cutting-edge features and generates a detailed comparison report.

Usage:
    python run_all_experiments.py                    # Run everything
    python run_all_experiments.py --device cuda     # Use GPU
    python run_all_experiments.py --output results/ # Custom output
"""

import argparse
from pathlib import Path
import sys
import logging

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from experimental.comprehensive_experiments import main as run_comprehensive


def main():
    parser = argparse.ArgumentParser(
        description="Run ALL novel SNN experiments (comprehensive suite)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run everything, save to outputs/novel_experiments
  %(prog)s --device cuda           # Use GPU if available
  %(prog)s --output my_results/    # Custom output directory
  %(prog)s --verbose               # Debug logging
        """
    )

    parser.add_argument('--output', type=Path, default=Path('outputs/novel_experiments'),
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cuda, cpu')
    parser.add_argument('--verbose', action='store_true',
                       help='Debug logging')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Run comprehensive suite
    print("="*80)
    print("COMPREHENSIVE NOVEL FEATURES EXPERIMENT SUITE")
    print("="*80)
    print(f"Output directory: {args.output}")
    print(f"Device: {device}")
    print("="*80)

    # Import and run
    from experimental import run_all_experiments
    import torch

    device_torch = torch.device(device)
    results = run_all_experiments(
        logging.getLogger("comprehensive"),
        args.output,
        device_torch
    )

    print("\n" + "="*80)
    print("EXPERIMENT SUITE COMPLETED!")
    print("="*80)
    print(f"\nResults saved to: {args.output / 'comprehensive_results.json'}")
    print("\nNext steps:")
    print("  1. Review results in the JSON file")
    print("  2. Check individual experiment outputs")
    print("  3. Compare performance across features")
    print("="*80)


if __name__ == "__main__":
    main()
