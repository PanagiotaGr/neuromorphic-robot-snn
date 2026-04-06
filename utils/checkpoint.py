"""Utility functions for model checkpointing"""
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import json


def save_checkpoint(state: Dict[str, Any], checkpoint_dir: Path, filename: str = "checkpoint.pt"):
    """Save model checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    return filepath


def load_checkpoint(filepath: Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    """Load model checkpoint"""
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def get_best_checkpoint(checkpoint_dir: Path, metric_name: str = "val_loss", mode: str = "min"):
    """
    Get the best checkpoint from directory based on metric.

    Args:
        checkpoint_dir: Directory containing checkpoints
        metric_name: Metric to compare (e.g., "val_loss", "val_acc")
        mode: "min" for lower is better, "max" for higher is better

    Returns:
        Path to best checkpoint, or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("best_*.pt")) + list(checkpoint_dir.glob("checkpoint_*.pt"))

    if not checkpoints:
        return None

    best_checkpoint = None
    best_metric = None

    for ckpt_path in checkpoints:
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            metric = checkpoint.get('metrics', {}).get(metric_name)

            if metric is None:
                continue

            if best_metric is None:
                best_metric = metric
                best_checkpoint = ckpt_path
            else:
                if (mode == "min" and metric < best_metric) or (mode == "max" and metric > best_metric):
                    best_metric = metric
                    best_checkpoint = ckpt_path
        except Exception as e:
            print(f"Warning: Could not load checkpoint {ckpt_path}: {e}")
            continue

    return best_checkpoint


def save_model_only(model: torch.nn.Module, filepath: Path):
    """Save just the model state dict"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_model_only(model: torch.nn.Module, filepath: Path):
    """Load just the model state dict"""
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    state_dict = torch.load(filepath, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)


class CheckpointManager:
    """
    Manage checkpoints during training with best model tracking.
    """

    def __init__(self, checkpoint_dir: Path, save_best_only: bool = True, metric_name: str = "val_loss", mode: str = "min"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.mode = mode

        self.best_metric = None
        self.best_checkpoint = None
        self.history = []

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, metrics: Dict[str, float], extra_state: Optional[Dict[str, Any]] = None):
        """
        Save checkpoint if it's the best so far (or always if save_best_only=False).

        Returns:
            True if checkpoint was saved
        """
        metric_value = metrics.get(self.metric_name)

        if metric_value is None and self.save_best_only:
            print(f"Warning: Metric '{self.metric_name}' not found in metrics. Saving checkpoint anyway.")

        should_save = False
        filename = None

        if not self.save_best_only:
            # Save every checkpoint
            filename = f"checkpoint_epoch_{epoch:04d}.pt"
            should_save = True
        else:
            # Save only if metric improved
            if self.best_metric is None:
                should_save = True
                filename = f"best_{self.metric_name}_{metric_value:.4f}_epoch_{epoch}.pt"
            else:
                improved = (self.mode == "min" and metric_value < self.best_metric) or \
                          (self.mode == "max" and metric_value > self.best_metric)
                if improved:
                    should_save = True
                    # Delete previous best
                    if self.best_checkpoint and self.best_checkpoint.exists():
                        self.best_checkpoint.unlink()
                    filename = f"best_{self.metric_name}_{metric_value:.4f}_epoch_{epoch}.pt"

        if should_save and filename:
            checkpoint_path = self.checkpoint_dir / filename

            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }
            if extra_state:
                state.update(extra_state)

            torch.save(state, checkpoint_path)

            if should_save:
                self.best_metric = metric_value
                self.best_checkpoint = checkpoint_path
                print(f"  → Saved checkpoint: {filename}")

            self.history.append({
                'epoch': epoch,
                'checkpoint': filename,
                'metrics': metrics,
            })

        return should_save

    def get_best_model_path(self) -> Optional[Path]:
        """Get path to best checkpoint"""
        return self.best_checkpoint
