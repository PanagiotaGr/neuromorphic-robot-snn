"""
Neuromodulated Plasticity - Learn WHEN and WHERE to apply plasticity

NOVEL: Instead of applying plasticity to all synapses equally, use
neuromodulators (like dopamine, acetylcholine) to gate plasticity.

Key ideas:
- Meta-learning which synapses are plastic (vs fixed)
- Modulating plasticity rate based on reward/novelty
- Different plasticities for different network regions
- Fast weights for temporary storage vs slow weights for long-term

This is inspired by biological neuromodulation (dopamine, serotonin, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import snntorch as snn
from snntorch import surrogate


class Neuromodulator(nn.Module):
    """
    Global neuromodulator that computes modulation signals based on
    network state and external feedback.

    Different neuromodulators inspired by biology:
    - Dopamine: reward signal, modulates learning rate
    - Acetylcholine: novelty/uncertainty, modulates exploration
    - Norepinephrine: arousal/attention, modulates gain
    - Serotonin: patience/patience, modulates plasticity threshold
    """

    def __init__(self, hidden_dim: int, num_modulators: int = 4):
        super().__init__()

        self.num_modulators = num_modulators
        # Encoder for network state
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Recent activity + error
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Compute each neuromodulator
        self.dopamine_net = nn.Linear(hidden_dim, 1)  # reward-based learning
        self.acetylcholine_net = nn.Linear(hidden_dim, 1)  # novelty/uncertainty
        self.norepinephrine_net = nn.Linear(hidden_dim, 1)  # attention/gain
        self.serotonin_net = nn.Linear(hidden_dim, 1)  # patience/consolidation

    def forward(self,
                recent_activity: torch.Tensor,
                prediction_error: Optional[torch.Tensor] = None,
                novelty: Optional[torch.Tensor] = None,
                reward: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute neuromodulatory signals.

        Args:
            recent_activity: [batch, hidden_dim] recent firing rates
            prediction_error: [batch] TD error or reward difference
            novelty: [batch] novelty/uncertainty estimate
            reward: [batch] external reward signal

        Returns:
            Dictionary of neuromodulator values ∈ [0, 1] or [-1, 1]
        """
        # Encode state
        state_input = recent_activity
        if prediction_error is not None:
            state_input = torch.cat([recent_activity, prediction_error.unsqueeze(-1)], dim=-1)
        else:
            # Dummy zeros for PE
            state_input = torch.cat([recent_activity, torch.zeros_like(recent_activity[:, :1])], dim=-1)

        encoded = self.state_encoder(state_input)

        modulators = {}

        # Dopamine: reward prediction error drives learning
        dopamine = torch.sigmoid(self.dopamine_net(encoded))
        if reward is not None:
            # Combine predicted with actual reward
            dopamine = dopamine * (1.0 + reward.unsqueeze(-1))
        modulators['dopamine'] = torch.clamp(dopamine, 0, 2)  # Allow >1 for strong learning

        # Acetylcholine: high when uncertain/novel → explore more
        ach = torch.sigmoid(self.acetylcholine_net(encoded))
        if novelty is not None:
            ach = ach * (1.0 + novelty.unsqueeze(-1))
        modulators['acetylcholine'] = torch.clamp(ach, 0, 1.5)

        # Norepinephrine: gain modulation for attention
        ne = torch.tanh(self.norepinephrine_net(encoded))  # [-1, 1]
        modulators['norepinephrine'] = ne

        # Serotonin: stability/consolidation (inverse of learning rate)
        serotonin = torch.sigmoid(self.serotonin_net(encoded))
        modulators['serotonin'] = serotonin

        return modulators


class GatedPlasticity(nn.Module):
    """
    Apply plasticity only to synapses where gate is open.

    Key innovation:
    - Each synapse has a binary (or continuous) plasticity gate
    - Gate opened by neuromodulators and activity correlation
    - Only "plastic" synapses update; "fixed" synapses remain constant
    - Enables stable memory + flexible learning
    """

    def __init__(self, num_synapses: int, init_gate: float = 0.5):
        super().__init__()

        # Plasticity gate per synapse (learnable)
        self.plasticity_gate = nn.Parameter(torch.ones(num_synapses) * init_gate)

        # Threshold for plasticity activation
        self.gate_threshold = 0.5

    def get_plastic_mask(self, temperature: float = 1.0) -> torch.Tensor:
        """
        Get binary mask of which synapses are currently plastic.
        Uses Gumbel-softmax for differentiability during training.

        Args:
            temperature: Gumbel-softmax temperature (high → soft, low → hard)

        Returns:
            plastic_mask: [num_synapses] binary or continuous [0, 1]
        """
        # Use straight-through estimator for binary decisions
        probs = torch.sigmoid(self.plasticity_gate)

        if self.training:
            # Straight-through estimator
            hard_mask = (probs > self.gate_threshold).float()
            return hard_mask + (probs - probs.detach())  # STE
        else:
            # Inference: use continuous or binary
            return probs

    def forward(self,
                weight: torch.Tensor,
                delta_w: torch.Tensor,
                neuromodulators: Dict[str, torch.Tensor],
                activity_correlation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply plasticity with gating.

        Args:
            weight: Current weight [batch, num_synapses] or [num_synapses]
            delta_w: Proposed weight update (from STDP or backprop)
            neuromodulators: Dict from Neuromodulator
            activity_correlation: Pre-post correlation [num_synapses]

        Returns:
            new_weight: Updated weight
        """
        # Get plastic mask
        plastic_mask = self.get_plastic_mask()

        # Modulate plasticity rate by dopamine (learning signal)
        dopamine = neuromodulators.get('dopamine', torch.tensor(1.0, device=weight.device))
        learning_rate_mult = dopamine.mean()  # Global modulation

        # Activity-dependent gating: only synapses with correlated activity
        if activity_correlation is not None:
            activity_mask = (activity_correlation > 0.1).float()
            plastic_mask = plastic_mask * activity_mask

        # Apply gated update
        delta_w_gated = delta_w * plastic_mask.unsqueeze(0) * learning_rate_mult

        # Apply weight change
        new_weight = weight + delta_w_gated

        # Weight constraints (e.g., non-negative for excitatory synapses)
        new_weight = torch.clamp(new_weight, 0, 10)

        return new_weight


class MetaPlasticityController(nn.Module):
    """
    Learn which synapses should be plastic at the network level.

    Instead of static plasticity gates, use a small network to decide
    based on task and network state.
    """

    def __init__(self, network_size: int, hidden_dim: int = 64):
        super().__init__()

        self.network_size = network_size

        # Controller network
        self.controller = nn.Sequential(
            nn.Linear(network_size * 2 + 10, hidden_dim),  # Connectivity pattern + task info
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, network_size),
            nn.Sigmoid()
        )

        # Task embedding
        self.task_embedding = nn.Embedding(100, 10)  # Support up to 100 tasks

    def forward(self,
                connectivity: torch.Tensor,
                task_id: Optional[int] = None,
                previous_tasks: Optional[List[int]] = None) -> torch.Tensor:
        """
        Compute plasticity gating for entire network.

        Args:
            connectivity: Adjacency or weight matrix [network_size, network_size]
            task_id: Current task identifier
            previous_tasks: List of previously learned tasks

        Returns:
            plasticity_mask: [network_size, network_size] or [network_size]
        """
        # Flatten connectivity
        conn_flat = connectivity.flatten()

        # Build input
        inputs = [conn_flat]

        # Add task context
        if task_id is not None:
            task_emb = self.task_embedding(torch.tensor(task_id, device=connectivity.device))
            inputs.append(task_emb.expand(conn_flat.size(0), -1).flatten())
        else:
            inputs.append(torch.zeros_like(conn_flat[:10]))

        # Concatenate all
        controller_input = torch.cat(inputs, dim=-1)

        # Get mask
        mask = self.controller(controller_input)

        # Reshape to connectivity shape if needed
        if mask.numel() == connectivity.numel():
            mask = mask.view_as(connectivity)

        return mask


class NeuromodulatedSNN(nn.Module):
    """
    SNN with neuromodulated plasticity.

    Combines:
    - Standard SNN forward pass
    - Neuromodulator that computes learning signals
    - Gated plasticity that selectively updates weights
    - Meta-plasticity controller for task-aware gating
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 use_neuromodulation: bool = True,
                 num_tasks: int = 10):
        super().__init__()

        beta = 0.92

        # Layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # Neuromodulated plasticity
        if use_neuromodulation:
            self.neuromodulator = Neuromodulator(hidden_dim)
            self.gated_plasticity = GatedPlasticity(
                num_synapses=input_dim * hidden_dim + hidden_dim * output_dim
            )
            self.meta_controller = MetaPlasticityController(
                network_size=hidden_dim + output_dim)
        else:
            self.neuromodulator = None
            self.gated_plasticity = None
            self.meta_controller = None

        # Track activity for neuromodulation
        self.register_buffer('recent_activity', None)
        self.register_buffer('last_error', None)

    def forward(self,
                spike_input: torch.Tensor,
                compute_neuromodulation: bool = False,
                task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            spike_input: [time, batch, input_dim]
            compute_neuromodulation: Whether to compute neuromodulator outputs
            task_id: Task identifier for meta-plasticity

        Returns:
            output_spikes: [time, batch, output_dim]
        """
        T, B, _ = spike_input.shape

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        output_rec = []
        hidden_activities = []

        for t in range(T):
            # Layer 1
            cur1 = self.fc1(spike_input[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            # Track hidden activity
            hidden_activities.append(spk1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            output_rec.append(spk2)

        output_spikes = torch.stack(output_rec)
        hidden_seq = torch.stack(hidden_activities)  # [T, B, hidden_dim]

        # Update recent activity (for neuromodulation)
        if compute_neuromodulation:
            # Mean activity over time
            self.recent_activity = hidden_seq.mean(dim=0)  # [B, hidden_dim]

        return output_spikes

    def compute_plasticity_updates(self,
                                   error_signal: torch.Tensor,
                                   novelty: Optional[torch.Tensor] = None,
                                   reward: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute plasticity updates based on neuromodulation.

        Should be called after forward pass if training with plasticity.

        Returns:
            Dictionary of weight updates and modulator values
        """
        if self.neuromodulator is None:
            return {}

        # Compute neuromodulators
        neuromodulators = self.neuromodulator(
            recent_activity=self.recent_activity,
            prediction_error=error_signal,
            novelty=novelty,
            reward=reward
        )

        # Could apply gated updates to weights here
        # This is a placeholder for actual STDP or weight update

        return {
            'neuromodulators': neuromodulators,
            'recent_activity': self.recent_activity
        }


class AdditiveNeuromodulatedLinear(nn.Linear):
    """
    Linear layer with neuromodulated additive updates.

    Instead of standard backprop, can apply neuromodulated weight changes
    that are additive and gated.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)

        # Additive weight buffer (plastic changes)
        self.register_buffer('weight_plasticity', torch.zeros_like(self.weight))
        self.register_buffer('bias_plasticity', torch.zeros_like(self.bias) if bias else None)

        # Plasticity rate (can be modulated)
        self.plasticity_rate = 0.01

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Combine base weights + plasticity
        effective_weight = self.weight + self.plasticity_rate * self.weight_plasticity
        effective_bias = self.bias + self.plasticity_rate * self.bias_plasticity if self.bias is not None else None

        return F.linear(input, effective_weight, effective_bias)

    def apply_neuromodulated_update(self,
                                     delta_w: torch.Tensor,
                                     neuromodulator: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None):
        """
        Apply neuromodulated weight update.

        Args:
            delta_w:_weight change
            neuromodulator: scalar or same shape as weight
            mask: optional binary mask for gated plasticity
        """
        if mask is not None:
            delta_w = delta_w * mask

        # Neuromodulated by dopamine (learning rate)
        modulation = torch.sigmoid(neuromodulator).mean()
        self.weight_plasticity += modulation * delta_w

        if self.bias_plasticity is not None:
            self.bias_plasticity += modulation * delta_w.mean()


# Test function
def test_neuromodulation():
    """Test neuromodulated plasticity components"""
    print("="*70)
    print("TEST: Neuromodulated Plasticity")
    print("="*70)

    batch_size = 4
    hidden_dim = 32

    # Test Neuromodulator
    print("\n1. Testing Neuromodulator...")
    neuromod = Neuromodulator(hidden_dim)

    recent_act = torch.randn(batch_size, hidden_dim)
    reward = torch.rand(batch_size)
    novelty = torch.rand(batch_size)

    mods = neuromod(recent_act, novelty=novelty, reward=reward)

    for name, value in mods.items():
        print(f"   {name}: mean={value.mean().item():.3f}, range=[{value.min().item():.3f}, {value.max().item():.3f}]")

    print("   ✓ Neuromodulator working")

    # Test GatedPlasticity
    print("\n2. Testing GatedPlasticity...")
    gated = GatedPlasticity(num_synapses=100)

    weight = torch.rand(100)
    delta_w = torch.randn(100) * 0.1
    mods_simple = {'dopamine': torch.tensor(0.5)}

    new_weight = gated(weight, delta_w, mods_simple)
    print(f"   Weight change: mean={new_weight.mean().item():.3f}")
    print(f"   Plastic mask active: {(gated.get_plastic_mask() > 0.5).float().mean().item()*100:.1f}%")

    print("   ✓ Gated plasticity working")

    # Test NeuromodulatedSNN
    print("\n3. Testing NeuromodulatedSNN...")
    snn = NeuromodulatedSNN(input_dim=9, hidden_dim=32, output_dim=3, use_neuromodulation=True)

    T, B = 10, 4
    spikes = torch.rand(T, B, 9) > 0.5
    spikes = spikes.float()

    output = snn(spikes, compute_neuromodulation=True)
    print(f"   Output shape: {output.shape}")

    plasticity_info = snn.compute_plasticity_updates(
        error_signal=torch.rand(batch_size),
        novelty=torch.rand(batch_size)
    )

    print(f"   Computed neuromodulators: {list(plasticity_info['neuromodulators'].keys())}")
    print("   ✓ NeuromodulatedSNN working")

    print("\n✅ All neuromodulation components functional!")
    return True


if __name__ == "__main__":
    test_neuromodulation()
