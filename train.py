import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from models import rate_encode, multi_step_encode
from utils import CheckpointManager, EarlyStopping, get_scheduler


def train_ann(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    logger=None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    early_stopping: Optional[EarlyStopping] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    gradient_clip: Optional[float] = None
) -> dict:
    """
    Train ANN model with advanced features.

    Returns:
        Training history dictionary
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "learning_rate": []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Forward pass
            logits = model(xb)
            loss = loss_fn(logits, yb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            # Metrics
            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluation
        test_loss, test_acc = evaluate_ann(model, test_loader, loss_fn, device)

        # Track metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])

        # Logging
        log_msg = (f"[ANN] Epoch {epoch+1:02d}/{epochs} | "
                   f"train_loss={train_loss:.4f} (acc={train_acc*100:.2f}%) | "
                   f"test_loss={test_loss:.4f} (acc={test_acc*100:.2f}%) | "
                   f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)

        # Save checkpoint
        if checkpoint_manager:
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": optimizer.param_groups[0]['lr']
            }
            checkpoint_manager.save(model, optimizer, epoch + 1, metrics)

        # Learning rate scheduler step
        if lr_scheduler:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(test_loss)
            else:
                lr_scheduler.step()

        # Early stopping check
        if early_stopping and early_stopping.step(test_loss):
            if logger:
                logger.info(f"EarlyStopping: Stopping training at epoch {epoch+1}")
            else:
                print(f"EarlyStopping: Stopping training at epoch {epoch+1}")
            break

    return history


def train_snn(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    num_steps: int,
    device: torch.device,
    encoding_type: str = "rate",
    encoding_kwargs: Optional[dict] = None,
    logger=None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    early_stopping: Optional[EarlyStopping] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    gradient_clip: Optional[float] = None
) -> dict:
    """
    Train SNN model with advanced features and flexible encoding.

    Returns:
        Training history dictionary
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    if encoding_kwargs is None:
        encoding_kwargs = {}

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "learning_rate": []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Spike encoding
            spk_in = multi_step_encode(xb, num_steps, encoding_type, **encoding_kwargs).to(device)

            # Forward pass
            spk_out, _ = model(spk_in)
            logits = spk_out.sum(dim=0)
            loss = loss_fn(logits, yb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            # Metrics
            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluation
        test_loss, test_acc = evaluate_snn(model, test_loader, num_steps, loss_fn, device, encoding_type, encoding_kwargs)

        # Track metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])

        # Logging
        log_msg = (f"[SNN] Epoch {epoch+1:02d}/{epochs} | "
                   f"train_loss={train_loss:.4f} (acc={train_acc*100:.2f}%) | "
                   f"test_loss={test_loss:.4f} (acc={test_acc*100:.2f}%) | "
                   f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)

        # Save checkpoint
        if checkpoint_manager:
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": optimizer.param_groups[0]['lr']
            }
            checkpoint_manager.save(model, optimizer, epoch + 1, metrics)

        # Learning rate scheduler step
        if lr_scheduler:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(test_loss)
            else:
                lr_scheduler.step()

        # Early stopping check
        if early_stopping and early_stopping.step(test_loss):
            if logger:
                logger.info(f"EarlyStopping: Stopping training at epoch {epoch+1}")
            else:
                print(f"EarlyStopping: Stopping training at epoch {epoch+1}")
            break

    return history


@torch.no_grad()
def evaluate_ann(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate ANN model"""
    model.eval()
    total = correct = 0
    total_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        preds = logits.argmax(dim=1)
        total += yb.size(0)
        correct += (preds == yb).sum().item()
        total_loss += loss.item() * yb.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_snn(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    num_steps: int,
    loss_fn: nn.Module,
    device: torch.device,
    encoding_type: str = "rate",
    encoding_kwargs: Optional[dict] = None
) -> Tuple[float, float]:
    """Evaluate SNN model with specified encoding"""
    if encoding_kwargs is None:
        encoding_kwargs = {}

    model.eval()
    total = correct = 0
    total_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        spk_in = multi_step_encode(xb, num_steps, encoding_type, **encoding_kwargs).to(device)
        spk_out, _ = model(spk_in)
        logits = spk_out.sum(dim=0)
        loss = loss_fn(logits, yb)
        preds = logits.argmax(dim=1)
        total += yb.size(0)
        correct += (preds == yb).sum().item()
        total_loss += loss.item() * yb.size(0)

    return total_loss / total, correct / total
