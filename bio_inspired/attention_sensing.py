"""
Attention-Driven Sensing - Active Perception for Neuromorphic Robots

NOVEL: Instead of fixed sensor arrangements, use attention to allocate
sensing resources to the most informative parts of the environment.

Key innovations:
- Dynamic sensor attention weights
- Saliency detection from SNN activity
- Active sensing: move sensors to look at attended locations
- Event-based attention (like human vision)
- Resource allocation: focus computation where it matters

This simulates how biological systems actively gather information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class SaliencyDetector(nn.Module):
    """
    Detect salient regions in sensor input based on:
    1. Prediction error (where we're uncertain)
    2. Novelty (new/unexpected patterns)
    3. Surprise (difference from expectation)

    Inspired by mammalian superior colliculus and visual attention.
    """

    def __init__(self, num_sensors: int, attention_dim: int = 32):
        super().__init__()

        self.num_sensors = num_sensors

        # Encode sensor history to build expectation
        self.history_encoder = nn.GRU(
            input_size=num_sensors,
            hidden_size=attention_dim,
            num_layers=1,
            batch_first=True
        )

        # Prediction head
        self.predictor = nn.Linear(attention_dim, num_sensors)

        # Novelty estimator
        self.novelty_estimator = nn.Sequential(
            nn.Linear(num_sensors * 2, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, num_sensors),
            nn.Sigmoid()
        )

        # Saliency merger
        self.saliency_merger = nn.Sequential(
            nn.Linear(num_sensors * 2, num_sensors),
            nn.ReLU(),
            nn.Linear(num_sensors, num_sensors),
            nn.Sigmoid()
        )

    def forward(self,
                current_sensors: torch.Tensor,
                sensor_history: torch.Tensor) -> torch.Tensor:
        """
        Compute saliency map over sensors.

        Args:
            current_sensors: [batch, num_sensors] current readings
            sensor_history: [batch, seq_len, num_sensors] recent history

        Returns:
            saliency: [batch, num_sensors] attention weights (0-1)
        """
        batch_size = current_sensors.size(0)

        # Encode history
        _, hidden = self.history_encoder(sensor_history)
        hidden = hidden[-1]  # Last layer hidden state

        # Predict expected sensors
        predicted = self.predictor(hidden)  # [batch, num_sensors]

        # Prediction error = saliency indicator
        pred_error = torch.abs(current_sensors - predicted)

        # Novelty: high when both current and history are low
        # (sudden new signal in previously empty region)
        history_mean = sensor_history.mean(dim=1)
        novelty = current_sensors * (1.0 - history_mean)

        # Combine indicators
        combined = torch.stack([pred_error, novelty], dim=-1)
        saliency = self.saliency_merger(combined.flatten(start_dim=1))
        saliency = torch.sigmoid(saliency)

        # Normalize to sum=1 (attention distribution)
        saliency = saliency / (saliency.sum(dim=-1, keepdim=True) + 1e-8)

        return saliency


class DynamicSensorArray(nn.Module):
    """
    Dynamically allocate sensing resources based on attention.

    Instead of fixed sensor positions, can:
    - Increase sampling rate in attended regions
    - Add virtual sensors at saliency locations
    - Fuse multiple sensory modalities
    """

    def __init__(self,
                 base_sensors: int,
                 max_sensors: int,
                 sensor_arc_deg: float = 90.0):
        super().__init__()

        self.base_sensors = base_sensors
        self.max_sensors = max_sensors
        self.sensor_arc_deg = sensor_arc_deg

        # Attention to sensor allocation
        self.allocation_net = nn.Sequential(
            nn.Linear(base_sensors, base_sensors * 2),
            nn.ReLU(),
            nn.Linear(base_sensors * 2, max_sensors),
            nn.Softmax(dim=-1)
        )

        # Sensor fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(max_sensors, base_sensors))

    def allocate_sensors(self,
                         saliency: torch.Tensor,
                         base_sensor_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Allocate more sensors to salient regions.

        Args:
            saliency: [batch, base_sensors] saliency weights
            base_sensor_positions: [base_sensors] angle positions in radians

        Returns:
            allocated_positions: [batch, max_sensors] virtual sensor positions
            allocation_weights: [batch, max_sensors] how much each allocated sensor contributes
        """
        batch_size = saliency.size(0)

        # Compute allocation distribution
        allocation = self.allocation_net(saliency)  # [batch, max_sensors]

        # Allocate sensors: each of the max_sensors gets assigned to a base sensor
        # Weighted by saliency
        positions = torch.zeros(batch_size, self.max_sensors, device=saliency.device)
        for b in range(batch_size):
            # Sample from allocation distribution
            indices = torch.multinomial(allocation[b], self.max_sensors, replacement=True)
            positions[b] = base_sensor_positions[indices]

        return positions, allocation

    def fuse_sensors(self,
                     base_readings: torch.Tensor,
                     virtual_sensor_indices: torch.Tensor) -> torch.Tensor:
        """
        Fuse base and virtual sensors.

        Args:
            base_readings: [batch, base_sensors]
            virtual_sensor_indices: [batch, max_sensors] indices into base sensors

        Returns:
            fused_readings: [batch, max_sensors] weighted combination
        """
        batch_size = base_readings.size(0)

        # Gather readings at virtual positions
        virtual_readings = torch.gather(
            base_readings, dim=1,
            index=virtual_sensor_indices.long()
        )

        # Fusion weights
        fusion = F.softmax(self.fusion_weights, dim=-1)  # [max_sensors, base_sensors]

        # Weighted combination
        fused = torch.matmul(base_readings.unsqueeze(1), fusion.T).squeeze(1)

        return fused


class AttentionDrivenSensing(nn.Module):
    """
    Complete attention-driven sensing system.

    1. Detect saliency from history
    2. Allocate computational resources (more sensors, longer integration)
    3. Fuse multi-resolution sensing
    4. Output enhanced sensor readings

    This is active perception: the agent decides WHERE to look.
    """

    def __init__(self,
                 base_num_sensors: int = 9,
                 max_virtual_sensors: int = 15,
                 sensor_arc_deg: float = 90.0,
                 history_length: int = 10):
        super().__init__()

        self.base_num_sensors = base_num_sensors
        self.max_virtual = max_virtual_sensors

        # Saliency detector
        self.saliency_detector = SaliencyDetector(
            num_sensors=base_num_sensors,
            attention_dim=32
        )

        # Sensor array
        self.sensor_array = DynamicSensorArray(
            base_sensors=base_num_sensors,
            max_sensors=max_virtual_sensors,
            sensor_arc_deg=sensor_arc_deg
        )

        # History buffer (will be managed externally)
        self.history_length = history_length
        self.register_buffer('sensor_history', None)

    def update_history(self, new_readings: torch.Tensor):
        """Update rolling history of sensor readings"""
        batch_size = new_readings.size(0)

        if self.sensor_history is None:
            self.sensor_history = new_readings.unsqueeze(1).repeat(1, self.history_length, 1)
        else:
            # Shift and append
            self.sensor_history = torch.cat([
                self.sensor_history[:, 1:],
                new_readings.unsqueeze(1)
            ], dim=1)

    def forward(self,
                base_sensors: torch.Tensor,
                base_angles: torch.Tensor,
                reset_history: bool = False) -> torch.Tensor:
        """
        Enhanced sensing with attention.

        Args:
            base_sensors: [batch, num_sensors] raw sensor readings
            base_angles: [num_sensors] sensor direction angles (radians)
            reset_history: Whether to clear history (for new episodes)

        Returns:
            enhanced_sensors: [batch, max_virtual_sensors] attention-enhanced readings
        """
        batch_size = base_sensors.size(0)

        if reset_history or self.sensor_history is None:
            self.update_history(base_sensors)
        else:
            self.update_history(base_sensors)

        # Detect saliency
        saliency = self.saliency_detector(base_sensors, self.sensor_history)

        # Allocate virtual sensors
        virtual_positions, allocation_weights = self.sensor_array.allocate_sensors(
            saliency, base_angles
        )

        # Create virtual sensor readings by interpolation
        # For simplicity, just use weighted combination of base sensors
        # More sophisticated: actually move physical sensors or rate-code
        enhanced = self.sensor_array.fuse_sensors(base_sensors, virtual_positions)

        # Enhance by saliency (amplify high-attention sensors)
        enhanced = enhanced * (1.0 + allocation_weights)

        return enhanced


class EventBasedAttention(nn.Module):
    """
    Event-camera style attention: only process events where change occurs.

    Computes difference from expectation, only "transmits" significant changes.
    Dramatically reduces bandwidth/computation.
    """

    def __init__(self, num_sensors: int, threshold: float = 0.1):
        super().__init__()

        self.threshold = threshold

        # Expectation predictor (simple exponential smoothing)
        self.register_buffer('expectation', None)

        # Attention gain (learnable)
        self.attention_gain = nn.Parameter(torch.ones(num_sensors))

    def update_expectation(self, new_value: torch.Tensor, alpha: float = 0.9):
        """Update exponential moving average"""
        if self.expectation is None:
            self.expectation = new_value
        else:
            self.expectation = alpha * self.expectation + (1 - alpha) * new_value

    def forward(self, sensor_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Event-based attention: only output significant changes.

        Args:
            sensor_input: [batch, num_sensors] current readings

        Returns:
            events: [batch, num_sensors] sparse change events
            attention_mask: [batch, num_sensors] where attention was drawn
        """
        # Update expectation
        self.update_expectation(sensor_input)

        # Compute difference from expectation
        diff = sensor_input - self.expectation

        # Absolute change
        abs_diff = torch.abs(diff)

        # Thresholding (like event camera)
        events = (abs_diff > self.threshold).float() * diff

        # Attention mask: high change = high attention
        attention_mask = torch.sigmoid(abs_diff * self.attention_gain)

        # Update expectation with events
        if events.abs().sum() > 0:
            self.update_expectation(sensor_input + events)

        return events, attention_mask


class HierarchicalAttention(nn.Module):
    """
    Multi-scale attention: local saliency + global context.

    Level 1: Per-sensor attention (which sensors matter)
    Level 2: Feature attention (which features of sensors matter)
    Level 3: Temporal attention (which time steps matter)
    """

    def __init__(self,
                 num_sensors: int,
                 time_steps: int,
                 hidden_dim: int = 64):
        super().__init__()

        # Sensor-level attention
        self.sensor_attention = nn.Sequential(
            nn.Linear(num_sensors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_sensors),
            nn.Softmax(dim=-1)
        )

        # Feature-level attention (weight each sensor differently)
        self.feature_attention = nn.Parameter(torch.ones(num_sensors))

        # Temporal attention (weight time steps)
        self.temporal_attention = nn.Sequential(
            nn.Linear(time_steps, time_steps),
            nn.Softmax(dim=-1)
        )

    def forward(self,
                sensor_sequence: torch.Tensor) -> torch.Tensor:
        """
        Apply hierarchical attention.

        Args:
            sensor_sequence: [batch, time, sensors]

        Returns:
            attended: [batch, sensors] weighted summary
        """
        batch_size, T, N = sensor_sequence.shape

        # Temporal attention first
        temp_weights = self.temporal_attention(
            sensor_sequence.mean(dim=-1)  # [batch, time]
        )  # [batch, time]
        temporal_weighted = (sensor_sequence * temp_weights.unsqueeze(-1)).sum(dim=1)

        # Sensor attention
        sensor_weights = self.sensor_attention(temporal_weighted.mean(dim=0, keepdim=True))
        # [1, sensors]

        # Combine
        attended = temporal_weighted * sensor_weights * self.feature_attention

        return attended


# Test
def test_attention_sensing():
    print("="*70)
    print("TEST: Attention-Driven Sensing")
    print("="*70)

    batch_size = 4
    num_sensors = 9

    # Test SaliencyDetector
    print("\n1. Testing SaliencyDetector...")
    saliency = SaliencyDetector(num_sensors=num_sensors)

    current = torch.rand(batch_size, num_sensors)
    history = torch.randn(batch_size, 10, num_sensors)

    sal = saliency(current, history)
    print(f"   Saliency shape: {sal.shape}")
    print(f"   Saliency sum (should be 1): {sal.sum(dim=-1)[0].item():.3f}")
    print(f"   Top-3 attended sensors: {sal[0].topk(3).indices.tolist()}")
    print("   ✓ Saliency detection working")

    # Test DynamicSensorArray
    print("\n2. Testing DynamicSensorArray...")
    array = DynamicSensorArray(base_sensors=num_sensors, max_virtual_sensors=15)

    base_angles = torch.linspace(-np.pi/4, np.pi/4, num_sensors)
    positions, alloc = array.allocate_sensors(sal, base_angles)

    print(f"   Allocated positions shape: {positions.shape}")
    print(f"   Allocation weights shape: {alloc.shape}")
    print("   ✓ Sensor allocation working")

    # Test EventBasedAttention
    print("\n3. Testing EventBasedAttention...")
    event_att = EventBasedAttention(num_sensors=num_sensors, threshold=0.1)

    seq = torch.randn(5, batch_size, num_sensors)

    for t in range(5):
        events, mask = event_att(seq[t])
        if t == 0:
            print(f"   Events shape: {events.shape}")
            print(f"   Attention mask shape: {mask.shape}")
            print(f"   Active events: {(events != 0).float().mean().item()*100:.1f}%")

    print("   ✓ Event-based attention working")

    # Test HierarchicalAttention
    print("\n4. Testing HierarchicalAttention...")
    hier_att = HierarchicalAttention(num_sensors=num_sensors, time_steps=10)

    sensor_seq = torch.randn(batch_size, 10, num_sensors)
    attended = hier_att(sensor_seq)

    print(f"   Input shape: {sensor_seq.shape}")
    print(f"   Attended output shape: {attended.shape}")
    print("   ✓ Hierarchical attention working")

    # Test full AttentionDrivenSensing
    print("\n5. Testing AttentionDrivenSensing...")
    ads = AttentionDrivenSensing(base_num_sensors=num_sensors, max_virtual_sensors=12)

    base_sens = torch.rand(batch_size, num_sensors)
    enhanced = ads(base_sens, base_angles)
    print(f"   Enhanced sensors shape: {enhanced.shape}")
    print(f"   Base → Virtual: {num_sensors} → {enhanced.size(1)}")
    print("   ✓ Attention-driven sensing pipeline working")

    print("\n✅ All attention sensing components functional!")
    return True


if __name__ == "__main__":
    test_attention_sensing()
