#!/usr/bin/env python3
"""
Comprehensive benchmarking script for ANN and SNN models.

Features:
  - Performance comparison across multiple tracks
  - Energy efficiency analysis
  - Statistical significance testing
  - Inference time benchmarking
  - Robustness evaluation under perturbations
"""

import argparse
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config_enhanced import Config, parse_cli_args
from dataset import generate_supervised_dataset
from models import ANNController, SNNController, DeepSNNController
from utils import compute_model_stats

from models import ANNController, SNNController, DeepSNNController
from utils import compute_model_stats

from train import evaluate_ann, evaluate_snn
from simulator import ProceduralTrack, run_episode
from utils import (
    CheckpointManager, compute_all_statistics, print_statistics_summary,
    compare_energy_efficiency, estimate_inference_time
)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger("benchmark")


def load_models(config, device, checkpoint_dir: Path):
    """Load trained models from checkpoint directory"""
    ann_model = ANNController(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim
    ).to(device)

    snn_model = SNNController(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim,
        beta=config.model.beta,
        deep=config.model.hidden_dim >= 192
    ).to(device)

    # Load checkpoints
    ann_ckpt = checkpoint_dir / "ann_final.pt"
    snn_ckpt = checkpoint_dir / "snn_final.pt"

    if ann_ckpt.exists():
        ann_model.load_state_dict(torch.load(ann_ckpt, map_location=device, weights_only=False))
        print(f"Loaded ANN checkpoint from {ann_ckpt}")
    else:
        print(f"Warning: ANN checkpoint not found at {ann_ckpt}")

    if snn_ckpt.exists():
        snn_model.load_state_dict(torch.load(snn_ckpt, map_location=device, weights_only=False))
        print(f"Loaded SNN checkpoint from {snn_ckpt}")
    else:
        print(f"Warning: SNN checkpoint not found at {snn_ckpt}")

    return ann_model, snn_model


def benchmark_performance(ann_model, snn_model, train_loader, test_loader,
                         config, device, logger):
    """Benchmark supervised learning performance"""
    logger.info("\n" + "="*60)
    logger.info("Performance Benchmark (Supervised)")
    logger.info("="*60)

    loss_fn = nn.CrossEntropyLoss()

    ann_loss, ann_acc = evaluate_ann(ann_model, test_loader, loss_fn, device)
    snn_loss, snn_acc = evaluate_snn(
        snn_model, test_loader, config.model.snn_steps, loss_fn, device,
        encoding_type=config.encoding.encoding_type
    )

    logger.info(f"ANN Test Loss: {ann_loss:.4f}, Accuracy: {ann_acc*100:.2f}%")
    logger.info(f"SNN Test Loss: {snn_loss:.4f}, Accuracy: {snn_acc*100:.2f}%")

    return {
        'ann': {'loss': ann_loss, 'accuracy': ann_acc},
        'snn': {'loss': snn_loss, 'accuracy': snn_acc}
    }


def benchmark_closed_loop(ann_model, snn_model, config, device, n_tracks: int = 50):
    """Benchmark closed-loop control performance"""
    results = []

    test_seeds = list(range(3000, 3000 + n_tracks))

    # Evaluate both policies
    ann_policy = ANNControllerPolicy(ann_model, device)
    snn_policy = SNNControllerPolicy(
        snn_model,
        config.model.snn_steps,
        device,
        encoding_type=config.encoding.encoding_type
    )

    conditions = [
        ("clean", config.sim),
        ("noise", config.sim.__class__(**{**config.sim.__dict__, 'noise_std': 0.08})),
        ("delay", config.sim.__class__(**{**config.sim.__dict__, 'delay_steps': 2})),
    ]

    for cond_name, sim_cfg in conditions:
        from evaluate import evaluate_policy
        rows = []
        rows.extend(evaluate_policy("ANN", ann_policy, test_seeds, sim_cfg, config.data))
        rows.extend(evaluate_policy("SNN", snn_policy, test_seeds, sim_cfg, config.data))
        results.append((cond_name, rows))

    return results


def benchmark_energy_efficiency(ann_model, snn_model, config, device):
    """Compute energy efficiency statistics"""
    ann_stats = compute_model_stats(ann_model, timesteps=config.model.snn_steps)
    snn_stats = compute_model_stats(snn_model, timesteps=config.model.snn_steps)
    comparison = compare_energy_efficiency(ann_stats, snn_stats)

    return ann_stats, snn_stats, comparison


def benchmark_inference_time(ann_model, snn_model, config, device, input_dim: int, num_runs: int = 100):
    """Benchmark inference times"""
    batch_size = 1

    ann_time = estimate_inference_time(
        ann_model,
        (batch_size, input_dim),
        device,
        num_runs=num_runs
    )

    # For SNN, we run the full temporal simulation
    with torch.no_grad():
        dummy_input = torch.randn(batch_size, input_dim, device=device)
        # Warmup
        for _ in range(10):
            spk_probs = torch.sigmoid(dummy_input).unsqueeze(0).repeat(config.model.snn_steps, 1, 1)
            spk_in = torch.bernoulli(spk_probs)
            _ = snn_model(spk_in.to(device))

        # Benchmark
        import time
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            spk_probs = torch.sigmoid(dummy_input).unsqueeze(0).repeat(config.model.snn_steps, 1, 1)
            spk_in = torch.bernoulli(spk_probs)
            _ = snn_model(spk_in.to(device))
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.perf_counter()
            times.append((end - start) * 1000)

    snn_time = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'fps': 1000.0 / np.mean(times) if np.mean(times) > 0 else 0
    }

    return ann_time, snn_time


class ANNControllerPolicy:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def act(self, sensor_values):
        self.model.eval()
        x = torch.tensor([sensor_values], dtype=torch.float32, device=self.device)
        logits = self.model(x)
        action = int(logits.argmax(dim=1).item())
        scores = logits.squeeze(0).detach().cpu().numpy()
        activity = float(np.mean(np.abs(scores)))
        return action, scores, activity


class SNNControllerPolicy:
    def __init__(self, model, num_steps, device, encoding_type="rate", encoding_kwargs=None):
        from models import multi_step_encode
        self.model = model
        self.num_steps = num_steps
        self.device = device
        self.encoding_type = encoding_type
        self.encoding_kwargs = encoding_kwargs or {}
        self.multi_step_encode = multi_step_encode

    @torch.no_grad()
    def act(self, sensor_values):
        self.model.eval()
        x = torch.tensor([sensor_values], dtype=torch.float32, device=self.device)
        spk_in = self.multi_step_encode(x, self.num_steps, self.encoding_type, **self.encoding_kwargs).to(self.device)
        spk_out, _ = self.model(spk_in)
        logits = spk_out.sum(dim=0)
        action = int(logits.argmax(dim=1).item())
        scores = logits.squeeze(0).detach().cpu().numpy()
        spike_count = float(spk_out.sum().item())
        return action, scores, spike_count


def main():
    parser = argparse.ArgumentParser(description="Comprehensive benchmarking for SNN vs ANN")
    parser.add_argument('--config', type=Path, help='Path to config YAML')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('outputs/checkpoints'),
                       help='Directory containing trained checkpoints')
    parser.add_argument('--n-tracks', type=int, default=50,
                       help='Number of tracks for closed-loop evaluation')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of runs for timing benchmark')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    args = parser.parse_args()

    logger = setup_logging(verbose=args.verbose)

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load models
    ann_model, snn_model = load_models(config, device, args.checkpoint_dir)

    # Benchmark 1: Supervised performance
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK 1: Supervised Learning Performance")
    logger.info("="*60)

    # Generate test dataset
    test_seeds = list(range(5000, 5000 + config.data.n_test_tracks))
    x_test, y_test = generate_supervised_dataset(test_seeds, config.data.samples_per_track // 2)
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=config.train.batch_size,
        shuffle=False
    )

    perf_results = benchmark_performance(ann_model, snn_model, None, test_loader, config, device, logger)

    # Benchmark 2: Energy efficiency
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK 2: Energy Efficiency")
    logger.info("="*60)

    ann_stats, snn_stats, energy_comparison = benchmark_energy_efficiency(ann_model, snn_model, config, device)

    logger.info(f"\nANN Statistics:")
    logger.info(f"  Parameters: {ann_stats['total_parameters']:,}")
    logger.info(f"  Energy per inference: {ann_stats['energy_estimate']['total_energy']:.2f} (relative units)")
    logger.info(f"  MACs: {ann_stats['estimated_macs']:,}")

    logger.info(f"\nSNN Statistics:")
    logger.info(f"  Parameters: {snn_stats['total_parameters']:,}")
    logger.info(f"  Energy per inference: {snn_stats['energy_estimate']['total_energy']:.2f} (relative units)")
    logger.info(f"  Spike count: {snn_stats['energy_estimate']['spike_count']:.1f}")
    logger.info(f"  Sparsity: {snn_stats['energy_estimate']['sparsity']*100:.1f}%")

    logger.info(f"\nEnergy Comparison:")
    logger.info(f"  SNN/ANN ratio: {energy_comparison['energy_ratio_snn_to_ann']:.2f}x")
    logger.info(f"  Energy savings: {energy_comparison['energy_savings_percent']:.1f}%")
    logger.info(f"  More efficient: {'SNN' if energy_comparison['snn_more_efficient'] else 'ANN'}")

    # Benchmark 3: Inference time
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK 3: Inference Time")
    logger.info("="*60)

    ann_time, snn_time = benchmark_inference_time(ann_model, snn_model, config, device, config.model.input_dim, args.num_runs)

    logger.info(f"\nANN inference time: {ann_time['mean_ms']:.3f} ± {ann_time['std_ms']:.3f} ms (n={args.num_runs})")
    logger.info(f"  FPS: {ann_time['fps']:.1f}")

    logger.info(f"\nSNN inference time: {snn_time['mean_ms']:.3f} ± {snn_time['std_ms']:.3f} ms (n={args.num_runs})")
    logger.info(f"  FPS: {snn_time['fps']:.1f}")

    # Benchmark 4: Closed-loop control
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK 4: Closed-Loop Control Performance")
    logger.info("="*60)

    cl_results = benchmark_closed_loop(ann_model, snn_model, config, device, args.n_tracks)

    for cond_name, rows in cl_results:
        ann_rows = [r for r in rows if r['policy'] == 'ANN']
        snn_rows = [r for r in rows if r['policy'] == 'SNN']

        ann_success = np.mean([r['success'] for r in ann_rows])
        snn_success = np.mean([r['success'] for r in snn_rows])
        ann_error = np.mean([r['mean_lateral_error'] for r in ann_rows])
        snn_error = np.mean([r['mean_lateral_error'] for r in snn_rows])

        logger.info(f"\n{cond_name.upper()} condition:")
        logger.info(f"  ANN: success={ann_success*100:.1f}%, mean_error={ann_error:.4f}")
        logger.info(f"  SNN: success={snn_success*100:.1f}%, mean_error={snn_error:.4f}")

    # Save results
    output_dir = Path(config.experiment.output_dir) / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    benchmark_results = {
        'supervised_performance': perf_results,
        'energy_efficiency': {
            'ann': ann_stats['energy_estimate'],
            'snn': snn_stats['energy_estimate'],
            'comparison': energy_comparison
        },
        'inference_time': {'ann': ann_time, 'snn': snn_time},
        'closed_loop': {cond: rows for cond, rows in cl_results}
    }

    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    logger.info(f"\nBenchmark results saved to {output_dir / 'benchmark_results.json'}")
    logger.info("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()