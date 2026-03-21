# Neuromorphic Path-Following Robot

A clean multi-file project for comparing an ANN controller and an SNN controller on a 2D path-following robot simulator.

## Features
- Procedural tracks
- 9 front sensors
- ANN baseline
- SNN with LIF neurons
- Noise, delay, dropout, and dead-sensor conditions
- Metrics, plots, GIF animations, saved model weights

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch snntorch numpy matplotlib pillow
python main.py
```

## Files
- `config.py`: central configuration
- `simulator.py`: track, robot, sensing, episode rollout
- `dataset.py`: supervised dataset generation and teacher policy
- `models.py`: ANN and SNN models and policy wrappers
- `train.py`: training loops
- `evaluate.py`: evaluation and CSV saving
- `plots.py`: plots and animation
- `main.py`: full pipeline entry point

## Outputs
Saved in `outputs/`.
