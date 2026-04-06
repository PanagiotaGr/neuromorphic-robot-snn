"""Energy estimation utilities for SNN and ANN models"""
import numpy as np
from typing import Dict, Any


# Energy constants (approximate, based on academic literature)
# These are normalized values for relative comparison
ENERGY_PER_SPIKE = 1.0  # Relative energy cost for one spike
ENERGY_PER_MAC = 0.1  # Relative energy cost for one multiply-accumulate (ANN)
ENERGY_PER_ACCESS = 0.01  # Memory access energy


def estimate_snn_energy(
    spike_count: int,
    num_neurons: int,
    timesteps: int,
    batch_size: int = 1
) -> Dict[str, float]:
    """
    Estimate energy consumption for SNN.

    Based on:
    - Spike communication energy
    - Membrane computation energy
    - Synaptic computation energy

    Returns:
        Dictionary with energy breakdown (in relative units)
    """
    total_spikes = spike_count

    # Each spike requires:
    # - spike generation (1 unit)
    # - synaptic transmission (1 unit)
    # - postsynaptic integration (1 unit)

    spike_energy = total_spikes * ENERGY_PER_SPIKE * 3

    # Membrane update and leakage
    membrane_energy = num_neurons * timesteps * ENERGY_PER_MAC * batch_size

    total_energy = spike_energy + membrane_energy

    return {
        'spike_energy': spike_energy,
        'membrane_energy': membrane_energy,
        'total_energy': total_energy,
        'energy_per_inference': total_energy / batch_size if batch_size > 0 else 0,
        'spike_count': total_spikes,
        'spikes_per_inference': total_spikes / batch_size if batch_size > 0 else 0,
        'sparsity': total_spikes / (num_neurons * timesteps * batch_size) if num_neurons * timesteps * batch_size > 0 else 0
    }


def estimate_ann_energy(
    num_parameters: int,
    num_macs: int,
    batch_size: int = 1
) -> Dict[str, float]:
    """
    Estimate energy consumption for ANN.

    Args:
        num_parameters: Total number of parameters
        num_macs: Number of multiply-accumulate operations per inference
        batch_size: Batch size

    Returns:
        Dictionary with energy breakdown
    """
    # Parameter loading and MAC operations
    mac_energy = num_macs * ENERGY_PER_MAC
    param_energy = num_parameters * ENERGY_PER_ACCESS

    total_energy = mac_energy + param_energy

    return {
        'mac_energy': mac_energy,
        'parameter_energy': param_energy,
        'total_energy': total_energy,
        'energy_per_inference': total_energy / batch_size if batch_size > 0 else 0,
        'mac_count': num_macs,
        'parameter_count': num_parameters
    }


def compute_model_stats(model: nn.Module, timesteps: int = 25) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a model including energy estimates.

    Args:
        model: PyTorch model
        timesteps: Number of timesteps for SNN (for energy estimation)

    Returns:
        Dictionary with statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate MACs for ANN
    # Simplified: assume one MAC per parameter per forward pass
    estimated_macs = total_params

    stats = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'estimated_macs': estimated_macs,
        'model_size_mb': total_params * 4 / (1024 ** 2),  # float32
    }

    # If it's an SNN, compute per-timestep energy
    if hasattr(model, 'deep'):
        # Estimate energy based on timesteps
        snn_energy = estimate_snn_energy(
            spike_count=estimated_macs * 0.5 * timesteps,  # Assuming 50% sparsity on average
            num_neurons=total_params // 2,  # Rough estimate
            timesteps=timesteps,
            batch_size=1
        )
        stats['energy_estimate'] = snn_energy
    else:
        ann_energy = estimate_ann_energy(
            num_parameters=total_params,
            num_macs=estimated_macs,
            batch_size=1
        )
        stats['energy_estimate'] = ann_energy

    return stats


def compare_energy_efficiency(ann_stats: Dict, snn_stats: Dict) -> Dict[str, float]:
    """
    Compare energy efficiency between ANN and SNN.

    Returns:
        Dictionary with comparison metrics
    """
    ann_energy = ann_stats['energy_estimate']['total_energy']
    snn_energy = snn_stats['energy_estimate']['total_energy']

    comparison = {
        'ann_energy': ann_energy,
        'snn_energy': snn_energy,
        'energy_ratio_snn_to_ann': snn_energy / ann_energy if ann_energy > 0 else float('inf'),
        'energy_savings_percent': (1 - snn_energy / ann_energy) * 100 if ann_energy > 0 else 0,
        'snn_more_efficient': snn_energy < ann_energy,
    }

    return comparison


def estimate_inference_time(model: nn.Module, input_shape: Tuple[int, ...],
                           device: torch.device, num_runs: int = 100,
                           warmup: int = 10) -> Dict[str, float]:
    """
    Benchmark inference time.

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch, ...)
        device: Device to run on
        num_runs: Number of runs for averaging
        warmup: Number of warmup runs

    Returns:
        Dictionary with timing statistics
    """
    import time

    model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    times = []

    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    mean_time = np.mean(times)
    std_time = np.std(times)

    return {
        'mean_ms': mean_time,
        'std_ms': std_time,
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'fps': 1000.0 / mean_time if mean_time > 0 else 0
    }
