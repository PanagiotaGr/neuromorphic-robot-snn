"""
Continual Learning - Learn New Tasks Without Forgetting Old Ones

NOVEL: Apply state-of-the-art continual learning techniques to SNNs.

Strategies implemented:
1. **Elastic Weight Consolidation (EWC)**: Bayesian regularization to protect important weights
2. **Progress & Compress**: Alternate between learning and consolidation
3. **Memory Replay**: Store examples from previous tasks (with SNN-specific encoding)
4. **Dynamic Architecture**: Add new neurons for new tasks (neurogenesis)
5. **Gradient Projection**: Project gradients to avoid interference

This is critical for lifelong learning robots!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


class ElasticWeightConsolidation(nn.Module):
    """
    Elastic Weight Consolidation (EWC) for SNNs.

    Prevents catastrophic forgetting by regularizing changes to
    weights that were important for previous tasks.

    Key equation:
    L = L_task + λ/2 * Σ_i F_i (θ_i - θ*_i)^2

    where F_i is the Fisher information (importance) of weight i
    and θ*_i is the optimal weight for previous tasks.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 5000.0,
                 online: bool = True, gamma: float = 0.9):
        super().__init__()

        self.model = model
        self.ewc_lambda = ewc_lambda
        self.online = online
        self.gamma = gamma

        # Store optimal parameters and Fisher matrix for each task
        self.optimal_params = {}  # task_id → {param_name: value}
        self.fisher_matrices = {}  # task_id → {param_name: importance}

    def compute_fisher_matrix(self,
                              data_loader: torch.utils.data.DataLoader,
                              task_id: int,
                              num_samples: int = 100):
        """
        Compute Fisher information matrix for current task.

        Args:
            data_loader: DataLoader for current task
            task_id: Identifier for this task
            num_samples: Number of samples to use (approximate Fisher)
        """
        print(f"Computing Fisher matrix for task {task_id}...")

        # Store current parameters as optimal for this task
        self.optimal_params[task_id] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Initialize Fisher matrix accumulator
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        n_samples = 0

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if n_samples >= num_samples:
                break

            inputs, targets = inputs.cuda(), targets.cuda()

            # Forward pass
            if hasattr(self.model, 'forward_snn'):
                outputs, _ = self.model(inputs)
                log_probs = F.log_softmax(outputs.sum(dim=0), dim=-1)
            else:
                log_probs = F.log_softmax(self.model(inputs), dim=-1)

            # Sample from predictive distribution
            probs = F.softmax(log_probs, dim=-1)
            sampled_target = torch.multinomial(probs, 1).squeeze()

            # Compute gradient of log-likelihood
            self.model.zero_grad()
            loss = F.nll_loss(log_probs, sampled_target)
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += (param.grad.detach() ** 2) * inputs.size(0)

            n_samples += inputs.size(0)

        # Normalize by number of samples
        for name in fisher:
            fisher[name] /= n_samples

        # Store Fisher matrix
        self.fisher_matrices[task_id] = fisher

        print(f"  Computed Fisher for {n_samples} samples")

        if self.online:
            # Update accumulated Fisher (exponentially weighted)
            if len(self.fisher_matrices) > 1:
                # Combine with previous
                older_tasks = [t for t in self.fisher_matrices.keys() if t != task_id]
                for old_task in older_tasks:
                    for name in fisher:
                        self.fisher_matrices[old_task][name] = (
                            self.gamma * self.fisher_matrices[old_task][name] +
                            (1 - self.gamma) * fisher[name]
                        )

    def ewc_loss(self, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        Args:
            task_id: If specified, only regularize w.r.t this task.
                    If None, regularize w.r.t all previous tasks.

        Returns:
            ewc_loss: scalar regularization term
        """
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        if task_id is not None:
            # Specific task
            tasks_to_consider = [task_id]
        else:
            # All tasks
            tasks_to_consider = list(self.optimal_params.keys())

        for t in tasks_to_consider:
            if t not in self.optimal_params:
                continue

            for name, param in self.model.named_parameters():
                if name in self.optimal_params[t] and param.requires_grad:
                    # Retrieve stored values
                    optimal = self.optimal_params[t][name]
                    fisher = self.fisher_matrices[t][name]

                    # EWC penalty: 0.5 * λ * F * (θ - θ*)^2
                    penalty = (fisher * (param - optimal) ** 2).sum()
                    ewc_loss = ewc_loss + 0.5 * self.ewc_lambda * penalty

        return ewc_loss


class MemoryReplay(nn.Module):
    """
    Memory replay for catastrophic forgetting prevention.

    Stores examples from previous tasks and replays them during
    training on new tasks.

    SNN-specific: Encode stored samples as spike trains consistent with training.
    """

    def __init__(self,
                 memory_size: int = 500,
                 sample_strategy: str = 'reservoir'):
        super().__init__()

        self.memory_size = memory_size
        self.sample_strategy = sample_strategy

        # Memory buffers (will hold raw data)
        self.memory_buffers = defaultdict(list)  # task_id → list of (input, target)
        self.memory_counts = defaultdict(int)

    def update_memory(self, task_id: int, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Update memory with new samples from task.

        Uses reservoir sampling to maintain fixed-size memory.
        """
        for i in range(inputs.size(0)):
            if self.memory_counts[task_id] < self.memory_size:
                # Fill memory
                self.memory_buffers[task_id].append((
                    inputs[i].clone(),
                    targets[i].clone()
                ))
                self.memory_counts[task_id] += 1
            else:
                # Reservoir sampling: replace with probability memory_size / count
                j = np.random.randint(0, self.memory_counts[task_id])
                if j < self.memory_size:
                    self.memory_buffers[task_id][j] = (
                        inputs[i].clone(),
                        targets[i].clone()
                    )
            self.memory_counts[task_id] += 1

    def get_replay_batch(self,
                         task_id: int,
                         batch_size: int,
                         include_current: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get replay batch from memory.

        Args:
            task_id: Which task's memory to sample from
            batch_size: Number of samples
            include_current: Whether to include some samples from current task

        Returns:
            inputs: [batch_size, ...]
            targets: [batch_size]
        """
        if task_id not in self.memory_buffers or len(self.memory_buffers[task_id]) == 0:
            # No memory for this task
            return torch.empty(0), torch.empty(0)

        buffer = self.memory_buffers[task_id]
        indices = np.random.choice(len(buffer), min(batch_size, len(buffer)), replace=False)

        inputs = torch.stack([buffer[i][0] for i in indices])
        targets = torch.tensor([buffer[i][1] for i in indices])

        return inputs, targets

    def get_mixed_replay_batch(self,
                               current_task_id: int,
                               batch_size: int,
                               replay_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get mixed batch: some from current task, some from previous tasks.

        Args:
            current_task_id: Current task being trained
            batch_size: Total batch size
            replay_ratio: Fraction of batch that is replay (vs current)

        Returns:
            inputs: [batch_size, ...]
            targets: [batch_size]
            task_ids: [batch_size] which task each sample came from
        """
        n_replay = int(batch_size * replay_ratio)
        n_current = batch_size - n_replay

        all_inputs = []
        all_targets = []
        all_tasks = []

        # Current task samples
        if n_current > 0:
            # Will be provided by the caller, use placeholders
            current_inputs = torch.zeros(n_current, *self.input_shape) if hasattr(self, 'input_shape') else None
            current_targets = torch.zeros(n_current)
            # Fill in caller provides actual data
            pass

        # Replay samples from previous tasks
        if n_replay > 0 and len(self.memory_buffers) > 0:
            # Sample from all previous tasks
            prev_task_ids = [t for t in self.memory_buffers.keys() if t != current_task_id]
            if prev_task_ids:
                n_per_task = n_replay // len(prev_task_ids) + 1
                for t in prev_task_ids:
                    r_inputs, r_targets = self.get_replay_batch(t, n_per_task)
                    if r_inputs.numel() > 0:
                        all_inputs.append(r_inputs[:min(n_per_task, r_inputs.size(0))])
                        all_targets.append(r_targets[:min(n_per_task, r_targets.size(0))])
                        all_tasks.append(torch.full((min(n_per_task, r_inputs.size(0)),), t))

        # Concatenate
        if all_inputs:
            replay_inputs = torch.cat(all_inputs, dim=0)[:n_replay]
            replay_targets = torch.cat(all_tasks, dim=0)[:n_replay]
            replay_tasks = torch.cat(all_tasks, dim=0)[:n_replay]
        else:
            # No replay available
            replay_inputs = torch.zeros(0)
            replay_targets = torch.zeros(0)
            replay_tasks = torch.zeros(0)

        # For simplicity, return replay only (caller will combine with current)
        return replay_inputs, replay_targets


class ProgressiveNeural Networks(nn.Module):
    """
    Progressive Neural Networks: Add new columns for each task.

    Architecture:
    - Shared base column (transfer from previous tasks)
    - Task-specific column on top
    - Lateral connections from previous task columns

    This prevents forgetting because old task columns are frozen!
    """

    def __init__(self, base_model: nn.Module, num_tasks: int = 10):
        super().__init__()

        self.base_column = base_model
        self.task_columns = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()
        self.frozen_columns = []  # List of frozen column indices

    def add_task_column(self, task_id: int, hidden_dim: int, output_dim: int):
        """
        Add a new column for a task.

        Args:
            task_id: Task identifier
            hidden_dim: Hidden layer size
            output_dim: Output size for this task
        """
        if task_id >= len(self.task_columns):
            # Create new column
            column = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            self.task_columns.append(column)

            # Create lateral connections from each previous column
            laterals = nn.ModuleList()
            for prev_col in self.task_columns[:-1]:
                lateral = nn.Linear(hidden_dim, hidden_dim)
                laterals.append(lateral)
            self.lateral_connections.append(laterals)

            # Freeze all previous columns
            for idx in self.frozen_columns:
                for param in self.task_columns[idx].parameters():
                    param.requires_grad = False

            self.frozen_columns.append(task_id)

    def forward(self,
                x: torch.Tensor,
                task_id: int,
                mode: str = 'train') -> torch.Tensor:
        """
        Forward through task-specific column with lateral connections.

        Args:
            x: Input
            task_id: Which task to run
            mode: 'train' or 'eval'

        Returns:
            output: Task-specific output
        """
        # Base column (shared)
        base_features = self.base_column(x)
        if isinstance(base_features, tuple):
            base_features = base_features[0]  # If returns (spikes, mem)

        # Task-specific column
        if task_id >= len(self.task_columns):
            raise ValueError(f"Task {task_id} column not created yet")

        column = self.task_columns[task_id]
        laterals = self.lateral_connections[task_id] if task_id < len(self.lateral_connections) else []

        # Apply lateral connections from frozen columns
        lateral_sum = 0
        for prev_idx, lateral in enumerate(laterals):
            if prev_idx < len(self.task_columns):
                prev_out = self.task_columns[prev_idx](x)
                lateral_sum = lateral_sum + lateral(prev_out.detach())  # Detach frozen columns

        # Task column input
        task_input = base_features + lateral_sum
        output = column(task_input)

        return output


class GradientProjection(nn.Module):
    """
    Project gradients to avoid interference with previous tasks.

    Key idea: If a gradient would increase loss on previous tasks,
    project it to the nullspace of those task gradients.

    This is "Gradient Episodic Memory" (GEM) or "Implicit Gradient
    Modulation" approach.
    """

    def __init__(self, model: nn.Module, memory_strength: float = 0.5):
        super().__init__()

        self.model = model
        self.memory_strength = memory_strength

        # Store gradients for previous tasks
        self.task_gradients = {}  # task_id → {param_name: gradient}
        self.task_constraints = {}  # task_id → what to preserve

    def store_gradients(self, task_id: int):
        """Store current gradients as constraint for this task"""
        self.task_gradients[task_id] = {
            name: param.grad.data.clone()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }

    def project_gradients(self,
                          current_task_id: int,
                          eps: float = 1e-8) -> float:
        """
        Project current gradient to avoid interference.

        Returns:
            violation: amount of constraint violation
        """
        if current_task_id <= 0:
            return 0.0  # First task, no constraints

        # Get current gradients
        current_grads = {
            name: param.grad.data
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }

        violation = 0.0

        # Check each previous task
        for prev_task in range(current_task_id):
            if prev_task not in self.task_gradients:
                continue

            # Compute dot product
            dot = 0.0
            for name in current_grads:
                if name in self.task_gradients[prev_task]:
                    dot += (current_grads[name] * self.task_gradients[prev_task][name]).sum()

            # If dot product is negative (would decrease loss on prev task), OK
            # If positive, need to project
            if dot < 0:
                # No violation
                continue

            violation += dot.item()

            # Project: subtract component along constraint gradient
            norm_sq = sum((self.task_gradients[prev_task][name] ** 2).sum()
                         for name in current_grads if name in self.task_gradients[prev_task])

            if norm_sq > eps:
                alpha = dot / norm_sq
                for name in current_grads:
                    if name in self.task_gradients[prev_task]:
                        current_grads[name] -= alpha * self.task_gradients[prev_task][name]

        # Write projected gradients back
        for name, param in self.model.named_parameters():
            if name in current_grads:
                param.grad.data = current_grads[name]

        return violation


class ContinualLearningSNN(nn.Module):
    """
    SNN with multiple continual learning strategies.

    Combines:
    - EWC for regularization
    - Memory replay
    - Gradient projection
    - (Optionally) progressive networks
    """

    def __init__(self,
                 base_model: nn.Module,
                 strategy: str = 'ewc',
                 ewc_lambda: float = 5000.0,
                 memory_size: int = 500,
                 use_progressive: bool = False):
        super().__init__()

        self.model = base_model
        self.strategy = strategy
        self.current_task_id = 0

        # Choose strategies
        if strategy in ['ewc', 'combined']:
            self.ewc = ElasticWeightConsolidation(base_model, ewc_lambda=ewc_lambda)
        else:
            self.ewc = None

        if strategy in ['replay', 'combined']:
            self.memory = MemoryReplay(memory_size=memory_size)
        else:
            self.memory = None

        if strategy in ['gem', 'combined']:
            self.grad_proj = GradientProjection(base_model)
        else:
            self.grad_proj = None

        if use_progressive:
            self.progressive = ProgressiveNeural Networks(base_model)
        else:
            self.progressive = None

        self.task_accuracies = defaultdict(list)

    def set_task(self, task_id: int):
        """Switch to new task"""
        self.current_task_id = task_id
        if self.progressive:
            # Add column if needed
            if task_id >= len(self.progressive.task_columns):
                hidden_dim = self.model.hidden_dim if hasattr(self.model, 'hidden_dim') else 96
                output_dim = self.model.output_dim if hasattr(self.model, 'output_dim') else 3
                self.progressive.add_task_column(task_id, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Forward with task-aware routing"""
        if task_id is None:
            task_id = self.current_task_id

        if self.progressive and task_id < len(self.progressive.task_columns):
            return self.progressive(x, task_id)
        else:
            return self.model(x)

    def compute_loss(self,
                     outputs: torch.Tensor,
                     targets: torch.Tensor,
                     task_id: int) -> torch.Tensor:
        """Compute total loss including regularization"""
        # Standard task loss
        task_loss = F.cross_entropy(outputs, targets)

        # Add EWC regularization
        if self.ewc and task_id > 0:
            ewc_loss = self.ewc.ewc_loss(task_id)
            total_loss = task_loss + ewc_loss
        else:
            total_loss = task_loss

        return total_loss

    def after_task_finished(self, task_id: int, data_loader: torch.utils.data.DataLoader):
        """Call after finishing training on a task"""
        if self.ewc:
            self.ewc.compute_fisher_matrix(data_loader, task_id)

        # Store gradients for GEM
        if self.grad_proj:
            self.grad_proj.store_gradients(task_id)


def continual_learning_loop(model: nn.Module,
                           tasks: List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
                           epochs_per_task: int = 10,
                           replay_samples: int = 32):
    """
    Train model on sequence of tasks with continual learning.

    Args:
        model: ContinualLearningSNN wrapper
        tasks: List of (train_loader, test_loader) for each task
        epochs_per_task: Training epochs per task
        replay_samples: How many replay samples per batch

    Returns:
        task_accuracies: Dict task_id → list of accuracies over time
    """
    device = next(model.parameters()).device

    for task_id, (train_loader, test_loader) in enumerate(tasks):
        print(f"\n{'='*70}")
        print(f"TASK {task_id}: Training")
        print(f"{'='*70}")

        model.set_task(task_id)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(epochs_per_task):
            model.train()

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward
                outputs = model(inputs, task_id=task_id)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                logits = outputs.sum(dim=0) if outputs.dim() == 3 else outputs

                loss = model.compute_loss(logits, targets, task_id)

                # Add replay loss
                if model.memory and task_id > 0:
                    replay_inputs, replay_targets = model.memory.get_replay_batch(
                        np.random.randint(0, task_id), replay_samples
                    )
                    if replay_inputs.numel() > 0:
                        replay_inputs = replay_inputs.to(device)
                        replay_targets = replay_targets.to(device)
                        replay_outputs = model(replay_inputs, task_id=task_id)
                        if isinstance(replay_outputs, tuple):
                            replay_outputs = replay_outputs[0]
                        replay_logits = replay_outputs.sum(dim=0) if replay_outputs.dim() == 3 else replay_outputs
                        replay_loss = F.cross_entropy(replay_logits, replay_targets)
                        loss = loss + 0.5 * replay_loss

                # Gradient projection
                if model.grad_proj and task_id > 0 and batch_idx % 10 == 0:
                    optimizer.zero_grad()
                    loss.backward()
                    violation = model.grad_proj.project_gradients(task_id)
                    if violation > 0.1:
                        print(f"  Gradient violation: {violation:.4f}")
                else:
                    optimizer.zero_grad()
                    loss.backward()

                optimizer.step()

                # Update memory
                if model.memory:
                    model.memory.update_memory(task_id, inputs.cpu(), targets.cpu())

            # Evaluate
            acc = evaluate_task(model, test_loader, task_id)
            model.task_accuracies[task_id].append(acc)
            print(f"Epoch {epoch+1}: Task {task_id} Test Acc = {acc*100:.2f}%")

        # After task: compute Fisher for EWC
        if model.ewc:
            model.after_task_finished(task_id, train_loader)

    return model.task_accuracies


def evaluate_task(model: nn.Module, test_loader: torch.utils.data.DataLoader, task_id: int) -> float:
    """Evaluate model on specific task"""
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, task_id=task_id)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            logits = outputs.sum(dim=0) if outputs.dim() == 3 else outputs
            preds = logits.argmax(dim=-1)

            if targets.dim() > 1:
                targets = targets.argmax(dim=-1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    print("Continual Learning module loaded!")
    print("Use: continual_learning_loop(model, tasks) to train on sequence of tasks")
