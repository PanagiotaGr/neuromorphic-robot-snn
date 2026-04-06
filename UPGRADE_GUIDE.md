# 🚀 Upgrade Guide: Enhanced Neuromorphic Robot SNN

This guide explains how to use the enhanced version of the project.

## New Features

### 1. Flexible Configuration System
- **YAML configs**: Store and reuse configurations in `configs/`
- **CLI overrides**: Override any parameter via command line
- Examples:
  ```bash
  python main_enhanced.py                          # Use defaults
  python main_enhanced.py --config configs/exp1.yaml  # Load config
  python main_enhanced.py --model.hidden_dim 128   # Override single param
  python main_enhanced.py --train.epochs 30 --train.lr 0.001  # Multiple overrides
  ```

### 2. Enhanced Spike Encodings
- **Rate coding** (original): probability-based spike generation
- **Latency coding**: strong inputs fire earlier (new!)
- **Population coding**: distributed representation across neuron populations (new!)
- Switch with `--encoding.encoding_type latency` or `population`

### 3. Deep SNN Architectures
- **Standard 2-layer SNN**: original architecture
- **Deep SNN**: 3 LIF layers with batch normalization and dropout
- **Dual-path SNN**: fast and slow integration pathways (inspired by biology)
- Enable with `--model.use_batch_norm true --model.dropout 0.1`

### 4. Advanced Training Features
- **Checkpointing**: Save best models automatically, resume training
- **Early stopping**: Prevent overfitting, configured with `--train.early_stopping_patience`
- **Learning rate schedulers**: step, cosine annealing, reduce-on-plateau
- **Gradient clipping**: `--train.gradient_clip 1.0` for stable training
- **Mixed precision**: `--train.mixed_precision` for faster training (GPU only)

### 5. Experiment Tracking
- **Weights & Biases integration**: automatic logging with `--experiment.use_wandb`
- **Comprehensive logging**: console + file logs in `outputs/`
- **Reproducibility**: Config saved with every experiment

### 6. Energy Efficiency Analysis
- Automatic estimation of energy consumption for SNN vs ANN
- Spike counting and sparsity metrics
- Compare theoretical energy efficiency
- Run with: `python benchmark.py` (new!)

### 7. Statistical Significance Testing
- Paired t-tests, Wilcoxon tests, McNemar's test
- Bootstrap confidence intervals
- Built into benchmark script

### 8. Modular Architecture
- New `utils/` package with:
  - `checkpoint.py`: model save/load
  - `early_stopping.py`: early stopping logic
  - `scheduler.py`: LR schedulers
  - `energy.py`: energy estimation
  - `statistics.py`: statistical tests

## Quick Start

### 1. Install Additional Dependencies
```bash
pip install -r requirements_enhanced.txt
```

### 2. Run a Quick Experiment
```bash
# Basic run with config file
python main_enhanced.py --config configs/default.yaml

# Try latency encoding with deep SNN
python main_enhanced.py --config configs/default.yaml --encoding.encoding_type latency --model.hidden_dim 192 --model.use_batch_norm true

# Custom experiment with checkpointing and early stopping
python main_enhanced.py \
  --train.ann_epochs 25 \
  --train.snn_epochs 30 \
  --train.early_stopping_patience 5 \
  --train.lr_scheduler cosine \
  --experiment.experiment_name "my_experiment_001" \
  --experiment.use_wandb
```

### 3. Run Comprehensive Benchmark
```bash
python benchmark.py --config configs/default.yaml --n-tracks 100 --num-runs 1000
```

## File Structure

```
neuromorphic-robot-snn/
├── config_enhanced.py       # Enhanced configuration with CLI
├── main_enhanced.py         # Main script with all features
├── benchmark.py            # Comprehensive benchmarking
├── configs/
│   ├── default.yaml       # Default configuration
│   ├── deep_snn.yaml      # Deep SNN configuration
│   └── latency_encoding.yaml  # Latency coding config
├── utils/
│   ├── checkpoint.py      # Checkpoint management
│   ├── early_stopping.py  # Early stopping
│   ├── scheduler.py       # LR schedulers
│   ├── energy.py          # Energy estimation
│   ├── statistics.py      # Statistical tests
│   └── __init__.py
├── outputs/               # All outputs (generated)
│   ├── checkpoints/
│   ├── plots/
│   ├── animations/
│   └── benchmarks/
└── README.md             # Original README
```

## Migration from Original Code

The original files are preserved. New/updated files:

- ✅ `models.py` - **Enhanced**: Added multiple encodings & architectures
- ✅ `train.py` - **Enhanced**: Checkpointing, early stopping, LR scheduling
- ✅ `main.py` - Keep as original reference
- ✅ `config_enhanced.py` - New: Configuration system
- ✅ `main_enhanced.py` - New: Main script (replace main.py)
- ✅ `utils/` - New: Utilities (importable)
- ✅ `benchmark.py` - New: Benchmarking script

### To use enhanced version: simply run `main_enhanced.py` instead of `main.py`.

## Customization Examples

### Example 1: Deep SNN with Dropout
Create `configs/deep_snn.yaml`:
```yaml
model:
  hidden_dim: 128
  dropout: 0.1
  use_batch_norm: true
  deep: true  # Uses DeepSNNController

train:
  epochs: 30
  lr_scheduler: "cosine"
  early_stopping_patience: 10
```

Run: `python main_enhanced.py --config configs/deep_snn.yaml`

### Example 2: Population Coding + Dual Path
```bash
python main_enhanced.py \
  --encoding.encoding_type population \
  --encoding.population_num_neurons 10 \
  --model.deep true \
  --model.hidden_dim 160
```

### Example 3: Run Multiple Experiments
```bash
# Experiment 1: Baseline
python main_enhanced.py --experiment.experiment_name "baseline" --config configs/baseline.yaml

# Experiment 2: Latency coding
python main_enhanced.py --experiment.experiment_name "latency" --config configs/baseline.yaml --encoding.encoding_type latency

# Experiment 3: Deep architecture
python main_enhanced.py --experiment.experiment_name "deep" --config configs/baseline.yaml --model.hidden_dim 192 --model.use_batch_norm true
```

## Tips

1. **Start simple**: Begin with `--config configs/default.yaml` to ensure setup works
2. **Use WandB**: Enable `--experiment.use_wandb` to track experiments automatically
3. **Checkpoints**: Checkpoints are saved in `outputs/checkpoints/YYYYMMDD_HHMMSS/`
4. **Best model**: The checkpoint manager saves the best model based on validation accuracy
5. **Resume training**: Reload the best checkpoint and continue training (not yet implemented, but checkpoints are there!)
6. **Energy estimation**: `utils/energy.py` provides theoretical estimates; actual measurements would require hardware

## Performance Notes

- **Mixed precision** can speed up training ~2x on modern GPUs (RTX 20xx+)
- **Batch size**: Adjust based on GPU memory (start with 96, can increase)
- **SNN timesteps**: More steps = more accurate temporal dynamics but slower
  - For real-time: 10-20 steps
  - For research: 25-50 steps
- **Dataset caching**: Enabled by default, speeds up repeated runs

## Troubleshooting

**Import errors**: Make sure you've installed all dependencies from `requirements_enhanced.txt`

**CUDA out of memory**: Reduce batch size: `--train.batch_size 32`

**Slow SNN training**: SNNs are inherently slower than ANNs due to temporal unfolding. Reduce `--model.snn_steps` for faster training.

**WandB not working**: Either install wandb (`pip install wandb`) or don't use `--experiment.use_wandb`

**No checkpoint found**: Make sure you've trained models first, or specify correct `--checkpoint-dir` in benchmark

## Next Steps

1. Run baseline experiment: `python main_enhanced.py`
2. Try different encodings: `--encoding.encoding_type latency`
3. Enable WandB tracking: `--experiment.use_wandb`
4. Run benchmark: `python benchmark.py`
5. Customize configs in `configs/`
6. Extend with custom architectures or encodings!

Enjoy the enhanced neuromorphic robot SNN project! 🎉
