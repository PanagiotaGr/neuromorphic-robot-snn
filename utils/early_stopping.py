"""Early stopping implementation"""
import numpy as np
from typing import Optional


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Args:
        patience: Number of epochs to wait after metric stops improving
        min_delta: Minimum change in metric to qualify as improvement
        mode: "min" for loss metrics, "max" for accuracy metrics
        verbose: Print messages when early stopping triggers
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 mode: str = "min", verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_metric = None
        self.early_stop = False

        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")

    def step(self, metric: float) -> bool:
        """
        Call after each epoch to check if training should stop.

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_metric is None:
            self.best_metric = metric
            return False

        if self.mode == "min":
            if metric < self.best_metric - self.min_delta:
                # Metric improved
                self.best_metric = metric
                self.counter = 0
            else:
                # Metric did not improve
                self.counter += 1
        else:  # mode == "max"
            if metric > self.best_metric + self.min_delta:
                # Metric improved
                self.best_metric = metric
                self.counter = 0
            else:
                # Metric did not improve
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"EarlyStopping: Triggered after {self.counter} epochs without improvement")
            return True

        return False

    def get_best_metric(self) -> Optional[float]:
        return self.best_metric

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
