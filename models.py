import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from typing import Tuple, Optional


def rate_encode(x: torch.Tensor, num_steps: int, threshold: float = 0.5) -> torch.Tensor:
    """
    Rate encoding: spike probability proportional to input magnitude.

    Args:
        x: Input tensor [batch, features]
        num_steps: Number of time steps
        threshold: Normalization threshold (default: 0.5)

    Returns:
        Binary spike tensor [num_steps, batch, features]
    """
    # Normalize to [0, 1] if not already
    x_norm = torch.clamp(x, 0, 1)
    x_rep = x_norm.unsqueeze(0).repeat(num_steps, 1, 1)
    return torch.bernoulli(x_rep)


def latency_encode(x: torch.Tensor, num_steps: int, threshold: float = 0.1) -> torch.Tensor:
    """
    Latency (time-to-first) encoding: stronger inputs fire earlier.

    Args:
        x: Input tensor [batch, features]
        num_steps: Number of time steps
        threshold: Minimum activation to trigger spike

    Returns:
        One-hot spike tensor [num_steps, batch, features]
    """
    batch_size, num_features = x.shape
    device = x.device

    # Convert to probability and compute latency
    x_norm = torch.clamp(x, 0, 1)
    # Higher input → earlier spike (inverse relationship)
    latency = (1.0 - x_norm) * (num_steps - 1)
    latency = torch.clamp(latency, 0, num_steps - 1).long()

    # Create one-hot spike encoding
    spikes = torch.zeros(num_steps, batch_size, num_features, device=device)
    for t in range(num_steps):
        spikes[t] = (latency == t).float()

    return spikes


def population_encode(x: torch.Tensor, num_neurons: int = 5,
                     num_steps: int = 25, threshold: float = 0.5) -> torch.Tensor:
    """
    Population coding: distributed representation across neuron populations.

    Each input feature is represented by a population of neurons with
    different activation thresholds (tuning curves).

    Args:
        x: Input tensor [batch, features]
        num_neurons: Number of neurons per feature population
        num_steps: Number of timesteps
        threshold: Base threshold parameter

    Returns:
        Spike tensor [num_steps, batch, features * num_neurons]
    """
    batch_size, num_features = x.shape
    device = x.device

    # Create tuning curves: each neuron responds to different input ranges
    x_expanded = x.unsqueeze(-1)  # [batch, features, 1]
    x_expanded = x_expanded.repeat(1, 1, num_neurons)

    # Create preferred values for each neuron in population
    # Range from 0 to 1, evenly spaced
    preferred = torch.linspace(0, 1, num_neurons, device=device)
    preferred = preferred.view(1, 1, num_neurons)

    # Gaussian tuning curve
    width = 1.0 / (num_neurons * 0.5)
    activity = torch.exp(-((x_expanded - preferred) ** 2) / (2 * width ** 2))

    # Rate encode over time
    activity = activity.reshape(batch_size, num_features * num_neurons)
    activity_rep = activity.unsqueeze(0).repeat(num_steps, 1, 1)

    return torch.bernoulli(activity_rep)


def multi_step_encode(x: torch.Tensor, num_steps: int, encoding_type: str = "rate",
                     **kwargs) -> torch.Tensor:
    """
    Unified encoding interface.

    Args:
        x: Input tensor [batch, features]
        num_steps: Number of timesteps
        encoding_type: 'rate', 'latency', or 'population'
        **kwargs: Additional parameters for specific encoders

    Returns:
        Spike tensor [num_steps, batch, features]
    """
    if encoding_type == "rate":
        threshold = kwargs.get('threshold', 0.5)
        return rate_encode(x, num_steps, threshold)
    elif encoding_type == "latency":
        threshold = kwargs.get('threshold', 0.1)
        return latency_encode(x, num_steps, threshold)
    elif encoding_type == "population":
        num_neurons = kwargs.get('population_num_neurons', 5)
        threshold = kwargs.get('threshold', 0.5)
        return population_encode(x, num_neurons, num_steps, threshold)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


class ANNController(nn.Module):
    """Standard feedforward ANN controller"""
    def __init__(self, input_dim: int = 9, hidden_dim: int = 96,
                 output_dim: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]

        # Add second hidden layer
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ])

        # Optional dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSNNController(nn.Module):
    """
    Deep SNN controller with multiple LIF layers and advanced features.

    Architecture:
        Input → [FC1 → LIF1 → FC2 → LIF2 → FC3 → LIF3] → Output
    """
    def __init__(self, input_dim: int = 9, hidden_dim: int = 96,
                 output_dim: int = 3, beta: float = 0.92,
                 spike_grad=surrogate.fast_sigmoid(), dropout: float = 0.0,
                 use_batch_norm: bool = False):
        super().__init__()

        # First layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Second layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Third layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2) if use_batch_norm else nn.Identity()
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Output layer (no LIF, direct readout)
        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)

        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def forward(self, spike_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the deep SNN.

        Args:
            spike_input: Input spikes [num_steps, batch, input_dim]

        Returns:
            output_spikes: Output spikes [num_steps, batch, output_dim]
            membrane_potentials: Final membrane potentials [num_steps, batch, output_dim]
        """
        batch_size = spike_input.size(1)

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec, mem3_rec = [], []

        for t in range(spike_input.size(0)):
            # Layer 1
            cur1 = self.fc1(spike_input[t])
            cur1 = self.bn1(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            if self.training and self.dropout > 0:
                spk1 = self.dropout1(spk1)

            # Layer 2
            cur2 = self.fc2(spk1)
            cur2 = self.bn2(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            if self.training and self.dropout > 0:
                spk2 = self.dropout2(spk2)

            # Layer 3
            cur3 = self.fc3(spk2)
            cur3 = self.bn3(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        output_spikes = torch.stack(spk3_rec)
        membrane_potentials = torch.stack(mem3_rec)

        return output_spikes, membrane_potentials


class DualPathSNNController(nn.Module):
    """
    Dual-path SNN with separate fast and slow integration pathways.
    Inspired by biological neural circuits.
    """
    def __init__(self, input_dim: int = 9, hidden_dim: int = 96,
                 output_dim: int = 3, beta_fast: float = 0.85,
                 beta_slow: float = 0.95):
        super().__init__()

        # Fast pathway (quick reactions)
        self.fc1_fast = nn.Linear(input_dim, hidden_dim // 2)
        self.lif1_fast = snn.Leaky(beta=beta_fast, spike_grad=surrogate.fast_sigmoid())

        # Slow pathway (context integration)
        self.fc1_slow = nn.Linear(input_dim, hidden_dim // 2)
        self.lif1_slow = snn.Leaky(beta=beta_slow, spike_grad=surrogate.fast_sigmoid())

        # Integration
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

        # Output
        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, spike_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = spike_input.size(1)

        mem1_fast = self.lif1_fast.init_leaky()
        mem1_slow = self.lif1_slow.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_out_rec, mem_out_rec = [], []

        for t in range(spike_input.size(0)):
            # Fast path
            cur_fast = self.fc1_fast(spike_input[t])
            spk_fast, mem1_fast = self.lif1_fast(cur_fast, mem1_fast)

            # Slow path
            cur_slow = self.fc1_slow(spike_input[t])
            spk_slow, mem1_slow = self.lif1_slow(cur_slow, mem1_slow)

            # Concatenate pathways
            combined = torch.cat([spk_fast, spk_slow], dim=1)

            # Integration layer
            cur2 = self.fc2(combined)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Output (no LIF, membrane potential readout)
            out = self.fc_out(spk2)
            spk_out_rec.append(out)
            mem_out_rec.append(mem2)

        return torch.stack(spk_out_rec), torch.stack(mem_out_rec)


class SNNController(nn.Module):
    """Standard SNN controller (backward compatibility)"""
    def __init__(self, input_dim: int = 9, hidden_dim: int = 96,
                 output_dim: int = 3, beta: float = 0.92, dropout: float = 0.0,
                 use_batch_norm: bool = False, deep: bool = False):
        super().__init__()

        if deep:
            # Use deep architecture
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            self.fc_out = nn.Linear(hidden_dim, output_dim)
            self.deep = True
        else:
            # Original two-layer architecture
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.deep = False

    def forward(self, spike_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = spike_input.size(1)

        if self.deep:
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            spk_out_rec, mem_out_rec = [], []

            for t in range(spike_input.size(0)):
                cur1 = self.fc1(spike_input[t])
                cur1 = self.bn1(cur1)
                spk1, mem1 = self.lif1(cur1, mem1)
                if self.training and self.dropout > 0:
                    spk1 = self.dropout1(spk1)

                cur2 = self.fc2(spk1)
                cur2 = self.bn2(cur2)
                spk2, mem2 = self.lif2(cur2, mem2)
                if self.training and self.dropout > 0:
                    spk2 = self.dropout2(spk2)

                out = self.fc_out(spk2)
                spk_out_rec.append(out)
                mem_out_rec.append(mem2)

            return torch.stack(spk_out_rec), torch.stack(mem_out_rec)
        else:
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            spk2_rec, mem2_rec = [], []

            for t in range(spike_input.size(0)):
                cur1 = self.fc1(spike_input[t])
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2_rec.append(spk2)
                mem2_rec.append(mem2)

            return torch.stack(spk2_rec), torch.stack(mem2_rec)


def count_spikes(spike_tensor: torch.Tensor) -> torch.Tensor:
    """Count total spikes in a spike tensor"""
    return spike_tensor.sum()


def get_spike_activity(spike_tensor: torch.Tensor) -> dict:
    """
    Calculate spike activity statistics.

    Returns:
        Dictionary with total_spikes, mean_spike_rate, active_neurons_ratio
    """
    total_spikes = spike_tensor.sum().item()
    num_steps, batch_size, num_neurons = spike_tensor.shape
    total_possible = num_steps * batch_size * num_neurons
    spike_rate = total_spikes / total_possible
    active_neurons = (spike_tensor.sum(dim=0) > 0).float().mean().item()

    return {
        'total_spikes': total_spikes,
        'spike_rate': spike_rate,
        'active_neurons_ratio': active_neurons,
    }


class ANNPolicy:
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
        sparse_proxy = float(np.mean(np.abs(scores)))
        return action, scores, sparse_proxy


class SNNPolicy:
    def __init__(self, model, num_steps, device):
        self.model = model
        self.num_steps = num_steps
        self.device = device

    @torch.no_grad()
    def act(self, sensor_values):
        self.model.eval()
        x = torch.tensor([sensor_values], dtype=torch.float32, device=self.device)
        spk_in = rate_encode(x, self.num_steps).to(self.device)
        spk_out, _ = self.model(spk_in)
        logits = spk_out.sum(dim=0)
        action = int(logits.argmax(dim=1).item())
        scores = logits.squeeze(0).detach().cpu().numpy()
        spike_count = float(spk_out.sum().item())
        return action, scores, spike_count
