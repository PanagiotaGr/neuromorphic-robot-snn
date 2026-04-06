"""Learning rate schedulers"""
import torch
from torch.optim.lr_scheduler import _LRScheduler


def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str, **kwargs):
    """
    Get learning rate scheduler by name.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ("none", "step", "cosine", "plateau")
        **kwargs: Scheduler-specific parameters

    Returns:
        Learning rate scheduler or None
    """
    if scheduler_type == "none":
        return None
    elif scheduler_type == "step":
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "cosine":
        T_max = kwargs.get('T_max', 50)
        eta_min = kwargs.get('eta_min', 1e-6)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type == "plateau":
        patience = kwargs.get('patience', 5)
        factor = kwargs.get('factor', 0.5)
        threshold = kwargs.get('threshold', 1e-4)
        verbose = kwargs.get('verbose', True)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience,
            threshold=threshold, verbose=verbose
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
