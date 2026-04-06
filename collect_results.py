#!/usr/bin/env python3
"""
Quick Benchmark - Collect baseline results for all models

This script runs a FAST version of experiments to get preliminary results:
- Fewer epochs (3-5 instead of 18-30)
- Fewer tracks (3-5 instead of 30)
- Smaller batch size for speed
- Quick evaluation

Goal: Get numbers to put in README/paper within 10-30 minutes!
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from config_enhanced import Config, setup_logging
from dataset import generate_supervised_dataset
from models import ANNController, SNNController, DeepSNNController, rate_encode, multi_step_encode
from train import train_ann, train_snn, evaluate_ann, evaluate_snn
from simulator import ProceduralTrack, run_episode
from evaluate import evaluate_policy
from utils import compute_model_stats, compare_energy_efficiency, estimate_inference_time


def quick_config():
    """Create a quick config for fast benchmarking"""
    config = Config()

    # Override for speed
    config.data.n_train_tracks = 5  # Instead of 30
    config.data.n_test_tracks = 3   # Instead of 12
    config.data.samples_per_track = 100  # Instead of 450

    config.train.batch_size = 64
    config.train.ann_epochs = 3  # Very few epochs
    config.train.snn_epochs = 4
    config.train.lr = 0.001

    config.model.hidden_dim = 64  # Smaller model
    config.model.snn_steps = 15   # Fewer timesteps

    config.experiment.output_dir = "outputs/quick_benchmark"
    config.experiment.log_level = "INFO"
    config.experiment.save_plots = False
    config.experiment.save_animations = False

    return config


def run_quick_benchmark(config, logger):
    """Run quick benchmark and collect results"""
    logger.info("="*70)
    logger.info("QUICK BENCHMARK - Collecting Preliminary Results")
    logger.info("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Generate small dataset
    logger.info("\n[1/5] Generating small dataset...")
    train_seeds = list(range(100, 100 + config.data.n_train_tracks))
    test_seeds = list(range(1000, 1000 + config.data.n_test_tracks))

    x_train, y_train = generate_supervised_dataset(train_seeds, config.data.samples_per_track)
    x_test, y_test = generate_supervised_dataset(test_seeds, config.data.samples_per_track // 2)

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=config.train.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=config.train.batch_size,
        shuffle=False
    )

    logger.info(f"  Dataset: {len(x_train)} train, {len(x_test)} test samples")

    # Train ANN
    logger.info("\n[2/5] Training ANN...")
    ann_model = ANNController(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim
    ).to(device)

    start_time = time.time()
    ann_history = train_ann(
        ann_model, train_loader, test_loader,
        epochs=config.train.ann_epochs,
        lr=config.train.lr,
        device=device,
        logger=logger
    )
    ann_train_time = time.time() - start_time

    ann_test_loss, ann_test_acc = evaluate_ann(
        ann_model, test_loader,
        loss_fn=nn.CrossEntropyLoss(),
        device=device
    )

    logger.info(f"  ANN final test accuracy: {ann_test_acc*100:.2f}%")
    logger.info(f"  ANN training time: {ann_train_time:.1f}s")

    # Train SNN
    logger.info("\n[3/5] Training SNN...")
    snn_model = SNNController(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim,
        beta=config.model.beta
    ).to(device)

    start_time = time.time()
    snn_history = train_snn(
        snn_model, train_loader, test_loader,
        epochs=config.train.snn_epochs,
        lr=config.train.lr,
        num_steps=config.model.snn_steps,
        device=device,
        logger=logger
    )
    snn_train_time = time.time() - start_time

    snn_test_loss, snn_test_acc = evaluate_snn(
        snn_model, test_loader,
        num_steps=config.model.snn_steps,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        encoding_type='rate'
    )

    logger.info(f"  SNN final test accuracy: {snn_test_acc*100:.2f}%")
    logger.info(f"  SNN training time: {snn_train_time:.1f}s")

    # Model statistics
    logger.info("\n[4/5] Computing model statistics...")
    ann_stats = compute_model_stats(ann_model, timesteps=config.model.snn_steps)
    snn_stats = compute_model_stats(snn_model, timesteps=config.model.snn_steps)
    energy_comparison = compare_energy_efficiency(ann_stats, snn_stats)

    logger.info(f"  ANN parameters: {ann_stats['total_parameters']:,}")
    logger.info(f"  SNN parameters: {snn_stats['total_parameters']:,}")
    logger.info(f"  Energy: ANN={ann_stats['energy_estimate']['total_energy']:.2f}, SNN={snn_stats['energy_estimate']['total_energy']:.2f}")
    logger.info(f"  Energy savings: {energy_comparison['energy_savings_percent']:.1f}%")

    # Inference time
    logger.info("\n[5/5] Benchmarking inference time...")
    ann_time = estimate_inference_time(ann_model, (1, config.model.input_dim), device, num_runs=50)
    snn_time = estimate_inference_time(snn_model, (1, config.model.input_dim), device, num_runs=50)

    logger.info(f"  ANN inference: {ann_time['mean_ms']:.3f} ± {ann_time['std_ms']:.3f} ms")
    logger.info(f"  SNN inference: {snn_time['mean_ms']:.3f} ± {snn_time['std_ms']:.3f} ms")

    # Collect all results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_train_tracks': config.data.n_train_tracks,
            'n_test_tracks': config.data.n_test_tracks,
            'samples_per_track': config.data.samples_per_track,
            'ann_epochs': config.train.ann_epochs,
            'snn_epochs': config.train.snn_epochs,
            'model_hidden_dim': config.model.hidden_dim,
            'snn_steps': config.model.snn_steps,
        },
        'ann': {
            'test_accuracy': float(ann_test_acc),
            'test_loss': float(ann_test_loss),
            'train_time_seconds': ann_train_time,
            'parameters': int(ann_stats['total_parameters']),
            'inference_time_ms': ann_time['mean_ms'],
            'energy_estimate': ann_stats['energy_estimate']['total_energy'],
            'history': {
                'train_loss': [float(x) for x in ann_history['train_loss']],
                'test_loss': [float(x) for x in ann_history['test_loss']],
                'test_acc': [float(x) for x in ann_history['test_acc']],
            }
        },
        'snn': {
            'test_accuracy': float(snn_test_acc),
            'test_loss': float(snn_test_loss),
            'train_time_seconds': snn_train_time,
            'parameters': int(snn_stats['total_parameters']),
            'inference_time_ms': snn_time['mean_ms'],
            'energy_estimate': snn_stats['energy_estimate']['total_energy'],
            'spike_count': int(snn_stats['energy_estimate']['spike_count']),
            'sparsity': float(snn_stats['energy_estimate']['sparsity']),
            'history': {
                'train_loss': [float(x) for x in snn_history['train_loss']],
                'test_loss': [float(x) for x in snn_history['test_loss']],
                'test_acc': [float(x) for x in snn_history['test_acc']],
            }
        },
        'comparison': {
            'accuracy_difference': float(ann_test_acc - snn_test_acc),
            'train_time_ratio': float(snn_train_time / ann_train_time),
            'inference_time_ratio': float(snn_time['mean_ms'] / ann_time['mean_ms']),
            'energy_ratio': float(snn_stats['energy_estimate']['total_energy'] / ann_stats['energy_estimate']['total_energy']),
            'energy_savings_percent': float(energy_comparison['energy_savings_percent']),
            'parameter_ratio': float(snn_stats['total_parameters'] / ann_stats['total_parameters']),
        }
    }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quick benchmark for results collection")
    parser.add_argument('--output', type=Path, default=Path('outputs/quick_benchmark/results.json'),
                       help='Output JSON file')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger("quick_benchmark")

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Run benchmark
        config = quick_config()
        results = run_quick_benchmark(config, logger)

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("\n" + "="*70)
        logger.info("BENCHMARK COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nResults saved to: {args.output}")

        # Print summary
        logger.info("\n📊 SUMMARY:")
        logger.info(f"  ANN accuracy: {results['ann']['test_accuracy']*100:.2f}%")
        logger.info(f"  SNN accuracy: {results['snn']['test_accuracy']*100:.2f}%")
        logger.info(f"  ANN train time: {results['ann']['train_time_seconds']:.1f}s")
        logger.info(f"  SNN train time: {results['snn']['train_time_seconds']:.1f}s")
        logger.info(f"  Energy savings: {results['comparison']['energy_savings_percent']:.1f}%")
        logger.info(f"  Inference: ANN={results['ann']['inference_time_ms']:.2f}ms, SNN={results['snn']['inference_time_ms']:.2f}ms")

        logger.info("\n✅ Ready to include in README/paper!")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import logging
    main()
