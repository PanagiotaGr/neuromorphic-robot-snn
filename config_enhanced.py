import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    """Data generation and processing configuration"""
    num_sensors: int = 9
    n_train_tracks: int = 30
    n_test_tracks: int = 12
    samples_per_track: int = 450
    lookahead: float = 0.85
    sensor_arc_deg: float = 90.0
    track_y_max: float = 40.0
    line_sigma: float = 0.28
    cache_dataset: bool = True  # New: cache generated datasets
    dataset_cache_dir: str = "cache/datasets"  # New: cache directory


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    input_dim: int = 9  # Should match DATA_CFG.num_sensors
    hidden_dim: int = 96
    output_dim: int = 3  # 3 actions: left, forward, right
    beta: float = 0.92  # LIF decay parameter
    snn_steps: int = 25  # Number of timesteps for SNN processing
    dropout: float = 0.0  # Dropout for regularization
    use_batch_norm: bool = False  # Batch normalization flag
    deep: bool = False  # Use deep SNN architecture (3 layers)


@dataclass
class TrainConfig:
    """Training configuration"""
    batch_size: int = 96
    ann_epochs: int = 18
    snn_epochs: int = 18
    lr: float = 1e-3
    lr_scheduler: str = "none"  # none, step, cosine, plateau
    lr_patience: int = 5  # For ReduceLROnPlateau
    lr_step_size: int = 10  # For StepLR
    lr_gamma: float = 0.5  # LR decay factor
    weight_decay: float = 0.0
    gradient_clip: float = 1.0  # Gradient clipping norm
    checkpoint_dir: str = "outputs/checkpoints"  # New: checkpoint directory
    save_best_only: bool = True  # Save only best model
    early_stopping_patience: int = 10  # Early stopping patience
    mixed_precision: bool = False  # Use mixed precision training
    seed: int = 13


@dataclass
class SimConfig:
    """Simulation/environment configuration"""
    speed: float = 0.16
    turn_rate: float = 0.11
    max_steps: int = 340
    failure_distance: float = 1.55
    delay_steps: int = 0
    noise_std: float = 0.0
    sensor_dropout_prob: float = 0.0
    dead_sensor_index: int = -1
    actuator_noise: float = 0.01  # New: actuator/motor noise


@dataclass
class EncodingConfig:
    """Spike encoding configuration"""
    encoding_type: str = "rate"  # rate, latency, population
    population_num_neurons: int = 5  # For population coding
    latency_time_window: int = 25  # For latency coding (will also be used as snn_steps if specified)
    threshold: float = 0.5  # Rate coding threshold
    snn_steps: int = 25  # Number of timesteps (primary location, can override model.snn_steps)


@dataclass
class ExperimentConfig:
    """Experiment tracking and output configuration"""
    output_dir: str = "outputs"
    experiment_name: Optional[str] = None
    use_wandb: bool = False  # Weights & Biases tracking
    wandb_project: str = "neuromorphic-robot-snn"
    wandb_entity: Optional[str] = None
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    save_plots: bool = True
    save_animations: bool = True
    save_trajectories: bool = False  # Save full trajectory data
    seed: int = 13  # Global random seed


@dataclass
class Config:
    """Master configuration combining all sub-configs"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Extract sub-configs
        data_cfg = DataConfig(**config_dict.get('data', {}))
        model_cfg = ModelConfig(**config_dict.get('model', {}))
        train_cfg = TrainConfig(**config_dict.get('train', {}))
        sim_cfg = SimConfig(**config_dict.get('sim', {}))
        encoding_cfg = EncodingConfig(**config_dict.get('encoding', {}))
        exp_cfg = ExperimentConfig(**config_dict.get('experiment', {}))

        return cls(
            data=data_cfg,
            model=model_cfg,
            train=train_cfg,
            sim=sim_cfg,
            encoding=encoding_cfg,
            experiment=exp_cfg
        )

    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'train': self.train.__dict__,
            'sim': self.sim.__dict__,
            'encoding': self.encoding.__dict__,
            'experiment': self.experiment.__dict__,
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary (for CLI args override)"""
        # Update nested configs
        data_cfg = DataConfig(**{**config_dict.get('data', {})})
        model_cfg = ModelConfig(**{**config_dict.get('model', {})})
        train_cfg = TrainConfig(**{**config_dict.get('train', {})})
        sim_cfg = SimConfig(**{**config_dict.get('sim', {})})
        encoding_cfg = EncodingConfig(**{**config_dict.get('encoding', {})})
        exp_cfg = ExperimentConfig(**{**config_dict.get('experiment', {})})

        return cls(
            data=data_cfg,
            model=model_cfg,
            train=train_cfg,
            sim=sim_cfg,
            encoding=encoding_cfg,
            experiment=exp_cfg
        )


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging with both console and file output"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logger = logging.getLogger("neuromorphic_snn")
    logger.setLevel(numeric_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def parse_cli_args() -> tuple[Config, argparse.Namespace]:
    """
    Parse command line arguments and return configuration.
    Supports: default config, YAML config file, or CLI overrides
    """
    parser = argparse.ArgumentParser(
        description="Neuromorphic Robot Control: SNN vs ANN Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run with default config
  %(prog)s --config configs/exp1.yaml  # Load config from YAML
  %(prog)s --model.hidden_dim 128      # Override specific parameter
  %(prog)s --train.epochs 30 --train.lr 0.001  # Multiple overrides
  %(prog)s --generate-dataset           # Only generate dataset
  %(prog)s --train --eval              # Train and evaluate
        """
    )

    # Config file
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to YAML configuration file'
    )

    # Mode selection
    mode_group = parser.add_argument_group('operation modes')
    mode_group.add_argument(
        '--generate-dataset',
        action='store_true',
        help='Only generate datasets without training'
    )
    mode_group.add_argument(
        '--train',
        action='store_true',
        help='Train models (default if no other mode specified)'
    )
    mode_group.add_argument(
        '--eval',
        action='store_true',
        help='Evaluate models (requires trained models or training)'
    )
    mode_group.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualizations and animations'
    )
    mode_group.add_argument(
        '--benchmark',
        action='store_true',
        help='Run comprehensive benchmarks'
    )

    # Data configuration
    data_group = parser.add_argument_group('data configuration')
    data_group.add_argument(
        '--data.num_sensors',
        type=int,
        help='Number of sensors'
    )
    data_group.add_argument(
        '--data.n_train_tracks',
        type=int,
        help='Number of training tracks'
    )
    data_group.add_argument(
        '--data.n_test_tracks',
        type=int,
        help='Number of test tracks'
    )
    data_group.add_argument(
        '--data.samples_per_track',
        type=int,
        help='Samples per track'
    )

    # Model configuration
    model_group = parser.add_argument_group('model configuration')
    model_group.add_argument(
        '--model.hidden_dim',
        type=int,
        help='Hidden dimension'
    )
    model_group.add_argument(
        '--model.beta',
        type=float,
        help='LIF decay parameter (beta)'
    )
    model_group.add_argument(
        '--model.snn_steps',
        type=int,
        help='Number of SNN timesteps'
    )
    model_group.add_argument(
        '--model.dropout',
        type=float,
        help='Dropout rate'
    )

    # Training configuration
    train_group = parser.add_argument_group('training configuration')
    train_group.add_argument(
        '--train.batch_size',
        type=int,
        help='Batch size'
    )
    train_group.add_argument(
        '--train.ann_epochs',
        type=int,
        help='ANN training epochs'
    )
    train_group.add_argument(
        '--train.snn_epochs',
        type=int,
        help='SNN training epochs'
    )
    train_group.add_argument(
        '--train.lr',
        type=float,
        help='Learning rate'
    )
    train_group.add_argument(
        '--train.lr_scheduler',
        type=str,
        choices=['none', 'step', 'cosine', 'plateau'],
        help='Learning rate scheduler'
    )
    train_group.add_argument(
        '--train.gradient_clip',
        type=float,
        help='Gradient clipping norm'
    )
    train_group.add_argument(
        '--train.mixed_precision',
        action='store_true',
        help='Enable mixed precision training'
    )
    train_group.add_argument(
        '--train.no_early_stopping',
        action='store_true',
        help='Disable early stopping'
    )

    # Encoding configuration
    encoding_group = parser.add_argument_group('encoding configuration')
    encoding_group.add_argument(
        '--encoding.encoding_type',
        type=str,
        choices=['rate', 'latency', 'population'],
        help='Spike encoding type'
    )
    encoding_group.add_argument(
        '--encoding.population_num_neurons',
        type=int,
        help='Neurons per population'
    )
    encoding_group.add_argument(
        '--encoding.threshold',
        type=float,
        help='Rate coding threshold'
    )

    # Experiment configuration
    exp_group = parser.add_argument_group('experiment configuration')
    exp_group.add_argument(
        '--experiment.output_dir',
        type=str,
        help='Output directory'
    )
    exp_group.add_argument(
        '--experiment.experiment_name',
        type=str,
        help='Experiment name (auto-generated if not provided)'
    )
    exp_group.add_argument(
        '--experiment.use_wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    exp_group.add_argument(
        '--experiment.wandb_entity',
        type=str,
        help='WandB entity/username'
    )
    exp_group.add_argument(
        '--experiment.log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    exp_group.add_argument(
        '--experiment.seed',
        type=int,
        help='Random seed'
    )

    # Parse arguments
    args = parser.parse_args()

    # Determine default mode
    if not any([args.generate_dataset, args.train, args.eval, args.visualize, args.benchmark]):
        args.train = True
        args.eval = True
        args.visualize = True

    # Load base configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Apply CLI overrides
    config_dict = {}

    # Helper to safely override nested configs
    def override_from_args(prefix: str, config_obj: object):
        result = {}
        for key, value in vars(args).items():
            if key.startswith(prefix + '.') and value is not None:
                nested_key = key.split('.', 1)[1]
                result[nested_key] = value
        return result

    # Override each config section
    for section_name in ['data', 'model', 'train', 'sim', 'encoding', 'experiment']:
        overrides = override_from_args(section_name, getattr(config, section_name))
        if overrides:
            # Update the dataclass
            section_obj = getattr(config, section_name)
            for key, value in overrides.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

    # Handle special cases
    if getattr(args, 'train.no_early_stopping', False):
        config.train.early_stopping_patience = 0

    # Sync dimensionality
    config.model.input_dim = config.data.num_sensors

    return config, args
