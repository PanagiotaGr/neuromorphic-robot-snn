#!/usr/bin/env python3
"""
Neuromorphic Robot Control: SNN vs ANN Comparison (Enhanced Edition)

This script implements a comprehensive comparison between Spiking Neural Networks
and Artificial Neural Networks for closed-loop robotic control tasks.

Features:
  - Flexible configuration via YAML or CLI arguments
  - Multiple spike encoding methods (rate, latency, population)
  - Deep SNN architectures with multiple LIF layers
  - Advanced training features: checkpointing, early stopping, LR scheduling
  - Experiment tracking with Weights & Biases integration
  - Comprehensive logging
  - Modular and extensible design
"""

import random
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Try to import wandb conditionally
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Local imports
from config_enhanced import Config, setup_logging, parse_cli_args
from dataset import generate_supervised_dataset
from models import (
    ANNController, SNNController, DeepSNNController,
    count_spikes, get_spike_activity
)
from train import train_ann, train_snn, evaluate_ann, evaluate_snn
from simulator import ProceduralTrack, run_episode
from evaluate import evaluate_policy, save_eval_csv
from plots import plot_training_curves, plot_episode, plot_bar_summary, make_animation
from utils import CheckpointManager, EarlyStopping, get_scheduler


class ExperimentRunner:
    """Main experiment orchestrator"""

    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds
        random.seed(config.experiment.seed)
        np.random.seed(config.experiment.seed)
        torch.manual_seed(config.experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.experiment.seed)

        # Initialize wandb if requested
        self.wandb_run = None
        if config.experiment.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()

        # Create output directory structure
        self._setup_directories()

        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Configuration: {config}")

    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        if not WANDB_AVAILABLE:
            self.logger.warning("wandb not available, skipping experiment tracking")
            return

        # Generate run name if not provided
        run_name = self.config.experiment.experiment_name
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"snb_robot_{timestamp}"

        self.wandb_run = wandb.init(
            project=self.config.experiment.wandb_project,
            entity=self.config.experiment.wandb_entity,
            name=run_name,
            config={
                'data': self.config.data.__dict__,
                'model': self.config.model.__dict__,
                'train': self.config.train.__dict__,
                'sim': self.config.sim.__dict__,
                'encoding': self.config.encoding.__dict__,
                'experiment': self.config.experiment.__dict__,
            }
        )
        self.logger.info(f"WandB run initialized: {wandb.run.url}")

    def _setup_directories(self):
        """Create output directory structure"""
        base_dir = Path(self.config.experiment.output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = base_dir / self.config.train.checkpoint_dir
        self.plots_dir = base_dir / "plots"
        self.animations_dir = base_dir / "animations"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.animations_dir.mkdir(parents=True, exist_ok=True)

        # Save config to output directory
        config_path = base_dir / "config_used.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump({
                'data': self.config.data.__dict__,
                'model': self.config.model.__dict__,
                'train': self.config.train.__dict__,
                'sim': self.config.sim.__dict__,
                'encoding': self.config.encoding.__dict__,
                'experiment': self.config.experiment.__dict__,
            }, f, default_flow_style=False)

    def generate_datasets(self) -> Tuple[DataLoader, DataLoader]:
        """Generate training and test datasets"""
        self.logger.info("Generating datasets...")

        train_seeds = list(range(100, 100 + self.config.data.n_train_tracks))
        test_seeds = list(range(1000, 1000 + self.config.data.n_test_tracks))

        # Check for caching
        cache_dir = Path(self.config.data.dataset_cache_dir)
        train_cache = cache_dir / f"train_{len(train_seeds)}tracks_{self.config.data.samples_per_track}samples.pt"
        test_cache = cache_dir / f"test_{len(test_seeds)}tracks_{self.config.data.samples_per_track // 2}samples.pt"

        if self.config.data.cache_dataset and train_cache.exists() and test_cache.exists():
            self.logger.info("Loading cached datasets...")
            x_train = torch.load(train_cache)
            y_train = torch.load(train_cache.with_suffix('.pt.labels'))
            x_test = torch.load(test_cache)
            y_test = torch.load(test_cache.with_suffix('.pt.labels'))
        else:
            x_train, y_train = generate_supervised_dataset(train_seeds, self.config.data.samples_per_track)
            x_test, y_test = generate_supervised_dataset(test_seeds, self.config.data.samples_per_track // 2)

            if self.config.data.cache_dataset:
                cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save(x_train, train_cache)
                torch.save(y_train, train_cache.with_suffix('.pt.labels'))
                torch.save(x_test, test_cache)
                torch.save(y_test, test_cache.with_suffix('.pt.labels'))

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 2-4 for parallel loading
            pin_memory=self.device.type == 'cuda'
        )
        test_loader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == 'cuda'
        )

        self.logger.info(f"Dataset: {len(x_train)} train samples, {len(x_test)} test samples")
        return train_loader, test_loader

    def create_models(self) -> Tuple[nn.Module, nn.Module]:
        """Create ANN and SNN models"""
        self.logger.info("Creating models...")

        # ANN Model
        ann_model = ANNController(
            input_dim=self.config.model.input_dim,
            hidden_dim=self.config.model.hidden_dim,
            output_dim=self.config.model.output_dim,
            dropout=self.config.model.dropout
        ).to(self.device)

        # SNN Model - choose architecture based on config
        # Use deep architecture if explicitly requested or if hidden_dim is large
        if getattr(self.config.model, 'deep', False) or (self.config.model.hidden_dim >= 192 and self.config.model.use_batch_norm):
            if hasattr(SNNController, 'deep'):
                snn_model = SNNController(
                    input_dim=self.config.model.input_dim,
                    hidden_dim=self.config.model.hidden_dim,
                    output_dim=self.config.model.output_dim,
                    beta=self.config.model.beta,
                    dropout=self.config.model.dropout,
                    use_batch_norm=self.config.model.use_batch_norm,
                    deep=True
                ).to(self.device)
            else:
                snn_model = DeepSNNController(
                    input_dim=self.config.model.input_dim,
                    hidden_dim=self.config.model.hidden_dim,
                    output_dim=self.config.model.output_dim,
                    beta=self.config.model.beta,
                    dropout=self.config.model.dropout,
                    use_batch_norm=self.config.model.use_batch_norm
                ).to(self.device)
            self.logger.info("Using DeepSNN architecture (3 LIF layers)")
        else:
            snn_model = SNNController(
                input_dim=self.config.model.input_dim,
                hidden_dim=self.config.model.hidden_dim,
                output_dim=self.config.model.output_dim,
                beta=self.config.model.beta,
                dropout=self.config.model.dropout,
                use_batch_norm=self.config.model.use_batch_norm,
                deep=False
            ).to(self.device)
            self.logger.info("Using standard SNN architecture (2 LIF layers)")

        # Log model sizes
        ann_params = sum(p.numel() for p in ann_model.parameters())
        snn_params = sum(p.numel() for p in snn_model.parameters())
        self.logger.info(f"ANN parameters: {ann_params:,}")
        self.logger.info(f"SNN parameters: {snn_params:,}")

        return ann_model, snn_model

    def train_models(self, train_loader: DataLoader, test_loader: DataLoader) -> Tuple[dict, dict]:
        """Train both ANN and SNN models"""
        self.logger.info("Starting training...")

        ann_model, snn_model = self.create_models()

        # Setup training utilities
        checkpoint_dir = self.checkpoint_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_manager_ann = CheckpointManager(
            checkpoint_dir / "ann",
            save_best_only=self.config.train.save_best_only,
            metric_name="test_acc",
            mode="max"
        ) if self.config.train.checkpoint_dir else None
        checkpoint_manager_snn = CheckpointManager(
            checkpoint_dir / "snn",
            save_best_only=self.config.train.save_best_only,
            metric_name="test_acc",
            mode="max"
        ) if self.config.train.checkpoint_dir else None

        early_stopping_ann = EarlyStopping(
            patience=self.config.train.early_stopping_patience,
            mode="min",
            verbose=True
        ) if self.config.train.early_stopping_patience > 0 else None
        early_stopping_snn = EarlyStopping(
            patience=self.config.train.early_stopping_patience,
            mode="min",
            verbose=True
        ) if self.config.train.early_stopping_patience > 0 else None

        lr_scheduler_ann = get_scheduler(
            torch.optim.Adam(ann_model.parameters(), lr=self.config.train.lr),
            self.config.train.lr_scheduler,
            patience=self.config.train.lr_patience,
            step_size=self.config.train.lr_step_size,
            gamma=self.config.train.lr_gamma
        )
        lr_scheduler_snn = get_scheduler(
            torch.optim.Adam(snn_model.parameters(), lr=self.config.train.lr),
            self.config.train.lr_scheduler,
            patience=self.config.train.lr_patience,
            step_size=self.config.train.lr_step_size,
            gamma=self.config.train.lr_gamma
        )

        # Train ANN
        self.logger.info("\n" + "="*60)
        self.logger.info("Training ANN...")
        self.logger.info("="*60)
        ann_history = train_ann(
            model=ann_model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=self.config.train.ann_epochs,
            lr=self.config.train.lr,
            device=self.device,
            logger=self.logger,
            checkpoint_manager=checkpoint_manager_ann,
            early_stopping=early_stopping_ann,
            lr_scheduler=lr_scheduler_ann,
            gradient_clip=self.config.train.gradient_clip if self.config.train.gradient_clip > 0 else None
        )

        # Train SNN
        self.logger.info("\n" + "="*60)
        self.logger.info("Training SNN...")
        self.logger.info("="*60)

        # Determine number of timesteps (encoding.snn_steps takes precedence)
        num_snn_steps = self.config.encoding.snn_steps if hasattr(self.config.encoding, 'snn_steps') and self.config.encoding.snn_steps else self.config.model.snn_steps

        encoding_kwargs = {
            'threshold': self.config.encoding.threshold,
            'population_num_neurons': self.config.encoding.population_num_neurons,
        }
        snn_history = train_snn(
            model=snn_model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=self.config.train.snn_epochs,
            lr=self.config.train.lr,
            num_steps=num_snn_steps,
            device=self.device,
            encoding_type=self.config.encoding.encoding_type,
            encoding_kwargs=encoding_kwargs,
            logger=self.logger,
            checkpoint_manager=checkpoint_manager_snn,
            early_stopping=early_stopping_snn,
            lr_scheduler=lr_scheduler_snn,
            gradient_clip=self.config.train.gradient_clip if self.config.train.gradient_clip > 0 else None
        )

        # Plot training curves
        if self.config.experiment.save_plots:
            plot_training_curves(
                ann_history, snn_history
            )

        return ann_history, snn_history

    def evaluate_models(self, ann_model: nn.Module, snn_model: nn.Module,
                        test_track_seeds: List[int]) -> List[dict]:
        """Evaluate both models under different conditions"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Evaluating models...")
        self.logger.info("="*60)

        ann_policy = ANNControllerPolicy(ann_model, self.device)

        # Determine num_steps for inference
        num_steps = self.config.encoding.snn_steps if hasattr(self.config.encoding, 'snn_steps') and self.config.encoding.snn_steps else self.config.model.snn_steps

        snn_policy = SNNControllerPolicy(
            snn_model,
            num_steps,
            self.device,
            encoding_type=self.config.encoding.encoding_type
        )

        conditions = [
            ("clean", self.config.sim),
            ("noise", self.config.sim.__class__(
                **{**self.config.sim.__dict__, 'noise_std': 0.08}
            )),
            ("delay", self.config.sim.__class__(
                **{**self.config.sim.__dict__, 'delay_steps': 2}
            )),
            ("dropout", self.config.sim.__class__(
                **{**self.config.sim.__dict__, 'sensor_dropout_prob': 0.18}
            )),
            ("dead_sensor", self.config.sim.__class__(
                **{**self.config.sim.__dict__, 'dead_sensor_index': 4}
            )),
        ]

        eval_rows = []
        for cond_name, sim_cfg in conditions:
            self.logger.info(f"\nEvaluating condition: {cond_name}")
            eval_rows.extend(evaluate_policy("ANN", ann_policy, test_track_seeds, sim_cfg, self.config.data))
            eval_rows.extend(evaluate_policy("SNN", snn_policy, test_track_seeds, sim_cfg, self.config.data))

        # Save results
        save_eval_csv(eval_rows, self.plots_dir / "evaluation_metrics.csv")

        # Plot summary
        if self.config.experiment.save_plots:
            plot_bar_summary(eval_rows, save_path=self.plots_dir / "summary_bar.png")

        return eval_rows

    def create_visualizations(self, ann_model: nn.Module, snn_model: nn.Module,
                              test_track_seeds: List[int]):
        """Create trajectory plots and animations"""
        if not self.config.experiment.save_plots and not self.config.experiment.save_animations:
            return

        self.logger.info("\n" + "="*60)
        self.logger.info("Creating visualizations...")
        self.logger.info("="*60)

        ann_policy = ANNControllerPolicy(ann_model, self.device)

        # Determine num_steps for inference
        num_steps = self.config.encoding.snn_steps if hasattr(self.config.encoding, 'snn_steps') and self.config.encoding.snn_steps else self.config.model.snn_steps

        snn_policy = SNNControllerPolicy(
            snn_model,
            num_steps,
            self.device,
            encoding_type=self.config.encoding.encoding_type
        )

        demo_track = ProceduralTrack(seed=2026, y_max=self.config.data.track_y_max)

        # Clean conditions
        sim_cfg_clean = self.config.sim
        _, ann_demo_hist, ann_demo_metrics = run_episode(demo_track, ann_policy, sim_cfg_clean, record=True)
        _, snn_demo_hist, snn_demo_metrics = run_episode(demo_track, snn_policy, sim_cfg_clean, record=True)

        if self.config.experiment.save_plots:
            plot_episode(
                demo_track, ann_demo_hist, snn_demo_hist,
                title_suffix="clean",
                save_path=self.plots_dir / "episode_comparison_clean.png"
            )

        # Noisy conditions
        sim_cfg_noise = self.config.sim.__class__(**{**self.config.sim.__dict__, 'noise_std': 0.08})
        _, ann_noise_hist, _ = run_episode(demo_track, ann_policy, sim_cfg_noise, record=True)
        _, snn_noise_hist, _ = run_episode(demo_track, snn_policy, sim_cfg_noise, record=True)

        if self.config.experiment.save_plots:
            plot_episode(
                demo_track, ann_noise_hist, snn_noise_hist,
                title_suffix="noise",
                save_path=self.plots_dir / "episode_comparison_noise.png"
            )

        # Animations
        if self.config.experiment.save_animations:
            self.logger.info("Creating animations (this may take a moment)...")
            make_animation(
                demo_track, snn_demo_hist,
                self.animations_dir / "snn_demo_clean.gif",
                "SNN Demo - Clean",
                dpi=100
            )
            make_animation(
                demo_track, snn_noise_hist,
                self.animations_dir / "snn_demo_noise.gif",
                "SNN Demo - Noisy Sensors",
                dpi=100
            )

        self.logger.info(f"Visualizations saved to {self.plots_dir} and {self.animations_dir}")

    def run(self, args: argparse.Namespace):
        """Main experiment runner"""
        try:
            # Generate datasets
            if args.generate_dataset or args.train or args.eval:
                train_loader, test_loader = self.generate_datasets()
                test_track_seeds = list(range(2000, 2000 + self.config.data.n_test_tracks))

            # Train models
            ann_model = snn_model = None
            ann_history = snn_history = None
            if args.train:
                ann_history, snn_history = self.train_models(train_loader, test_loader)

                # Save final models
                torch.save(ann_model.state_dict() if ann_model else None,
                          self.checkpoint_dir / "ann_final.pt")
                torch.save(snn_model.state_dict() if snn_model else None,
                          self.checkpoint_dir / "snn_final.pt")

            # Load models if not trained in this run
            if args.eval and not args.train:
                ann_model, snn_model = self.create_models()
                checkpoint_path = self.checkpoint_dir / "ann_final.pt"
                if checkpoint_path.exists():
                    ann_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=False))
                    self.logger.info("Loaded ANN checkpoint")
                checkpoint_path = self.checkpoint_dir / "snn_final.pt"
                if checkpoint_path.exists():
                    snn_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=False))
                    self.logger.info("Loaded SNN checkpoint")

            # Evaluate
            if args.eval and ann_model and snn_model:
                self.evaluate_models(ann_model, snn_model, test_track_seeds)

            # Visualize
            if args.visualize and ann_model and snn_model:
                self.create_visualizations(ann_model, snn_model, test_track_seeds)

            # Finalize
            if self.wandb_run:
                self.wandb_run.finish()

            self.logger.info("\n" + "="*60)
            self.logger.info("Experiment completed successfully!")
            self.logger.info(f"Outputs saved to: {self.config.experiment.output_dir}")
            self.logger.info("="*60)

        except KeyboardInterrupt:
            self.logger.info("\nInterrupted by user")
            if self.wandb_run:
                self.wandb_run.finish(exit_code=1)
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}", exc_info=True)
            raise


class ANNControllerPolicy:
    """Wrapper for ANN model to match policy interface"""
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def act(self, sensor_values: np.ndarray):
        self.model.eval()
        x = torch.tensor([sensor_values], dtype=torch.float32, device=self.device)
        logits = self.model(x)
        action = int(logits.argmax(dim=1).item())
        scores = logits.squeeze(0).detach().cpu().numpy()
        activity = float(np.mean(np.abs(scores)))
        return action, scores, activity


class SNNControllerPolicy:
    """Wrapper for SNN model to match policy interface"""
    def __init__(self, model: nn.Module, num_steps: int, device: torch.device,
                 encoding_type: str = "rate", encoding_kwargs: Optional[dict] = None):
        self.model = model
        self.num_steps = num_steps
        self.device = device
        self.encoding_type = encoding_type
        self.encoding_kwargs = encoding_kwargs or {}

    @torch.no_grad()
    def act(self, sensor_values: np.ndarray):
        from models import multi_step_encode
        self.model.eval()
        x = torch.tensor([sensor_values], dtype=torch.float32, device=self.device)
        spk_in = multi_step_encode(x, self.num_steps, self.encoding_type, **self.encoding_kwargs).to(self.device)
        spk_out, _ = self.model(spk_in)
        logits = spk_out.sum(dim=0)
        action = int(logits.argmax(dim=1).item())
        scores = logits.squeeze(0).detach().cpu().numpy()
        spike_count = float(spk_out.sum().item())
        return action, scores, spike_count


def main():
    """Main entry point with CLI"""
    config, args = parse_cli_args()

    # Setup logging
    log_file = Path(config.experiment.output_dir) / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.experiment.log_level,
        log_file=log_file
    )

    try:
        runner = ExperimentRunner(config, logger)
        runner.run(args)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Ensure wandb is closed
        if 'runner' in locals() and runner.wandb_run:
            runner.wandb_run.finish()


if __name__ == "__main__":
    main()
