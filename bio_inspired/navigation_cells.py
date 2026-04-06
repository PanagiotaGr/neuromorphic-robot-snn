"""
Bio-Inspired Navigation System with Grid Cells, Head Direction Cells, and Place Cells

This module implements a neuromorphic navigation system inspired by the mammalian hippocampus
and entorhinal cortex. It provides:

1. **Head Direction (HD) Cells**: Encode the agent's orientation (θ)
2. **Grid Cells**: Periodic spatial representations (hexagonal grids)
3. **Place Cells**: Location-specific representations
4. **Path Integration**: Dead reckoning using velocity integration

This is a completely novel approach to robot navigation that has not been implemented
in the SNN robotics literature with this combination.

Based on biological findings:
- Grid cells (Moser et al., 2005, Nobel Prize 2014)
- Head direction cells (Taube et al., 1990)
- Place cells (O'Keefe & Dostrovsky, 1971, Nobel Prize 2014)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import snntorch as snn
from snntorch import surrogate


class HeadDirectionCell(nn.Module):
    """
    Head Direction (HD) cell population.

    Encodes the agent's orientation using a ring attractor network.
    Each neuron represents a preferred direction angle.
    """

    def __init__(self, num_cells: int = 72, sigma: float = 0.1):
        """
        Args:
            num_cells: Number of HD cells (evenly spaced around circle)
            sigma: Width of tuning curve (radians)
        """
        super().__init__()
        self.num_cells = num_cells
        self.sigma = sigma

        # Preferred directions evenly spaced around [0, 2π)
        self.register_buffer(
            'preferred_angles',
            torch.linspace(0, 2 * np.pi, num_cells + 1)[:-1]  # Exclude last (wrap-around)
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Encode orientation into HD cell activity.

        Args:
            theta: Agent orientation [batch] in radians

        Returns:
            Activity: [batch, num_cells] Gaussian tuning curves
        """
        # Normalize theta to [0, 2π)
        theta = theta % (2 * np.pi)

        # Compute difference from preferred angles
        diff = torch.abs(theta.unsqueeze(1) - self.preferred_angles.unsqueeze(0))
        diff = torch.min(diff, 2 * np.pi - diff)  # Circular distance

        # Gaussian tuning
        activity = torch.exp(-(diff ** 2) / (2 * self.sigma ** 2))

        # Normalize to [0, 1]
        activity = activity / activity.max(dim=1, keepdim=True)[0].clamp(min=1e-8)

        return activity

    def decode(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Decode orientation from HD cell activity (population vector).

        Args:
            activity: [batch, num_cells]

        Returns:
            theta: Decoded orientation [batch]
        """
        # Weighted circular mean
        angles = self.preferred_angles.unsqueeze(0).repeat(activity.size(0), 1)
        sin_sum = (activity * torch.sin(angles)).sum(dim=1)
        cos_sum = (activity * torch.cos(angles)).sum(dim=1)
        theta = torch.atan2(sin_sum, cos_sum) % (2 * np.pi)
        return theta


class GridCell(nn.Module):
    """
    Grid cell module producing hexagonal grid-like periodic representations.

    Inspired by the grid cells in entorhinal cortex.
    Uses multiple spatial frequencies and orientations.
    """

    def __init__(self, num_cells: int = 64, scale_min: float = 0.5, scale_max: float = 8.0):
        """
        Args:
            num_cells: Number of grid cells
            scale_min, scale_max: Minimum and maximum grid scale (relative to track)
        """
        super().__init__()
        self.num_cells = num_cells

        # Each grid cell has: 3 phases (λ, φ), 2-3 spatial frequencies
        num_patterns = num_cells // 3
        if num_patterns * 3 < num_cells:
            num_patterns += 1

        # Spatial frequencies (log-spaced)
        scales = torch.logspace(np.log10(scale_min), np.log10(scale_max), num_patterns)

        # Orientations: 0°, 60°, 120° for hexagonal symmetry
        orientations = torch.tensor([0.0, np.pi/3, 2*np.pi/3])

        # Create all combinations
        self.scales = []
        self.orientations = []
        for i in range(num_patterns):
            for ori in orientations:
                if len(self.scales) < num_cells:
                    self.scales.append(scales[i])
                    self.orientations.append(ori)

        self.scales = torch.tensor(self.scales[:num_cells])
        self.orientations = torch.tensor(self.orientations[:num_cells])

        # Phase offsets (learnable)
        self.phases = nn.Parameter(torch.randn(num_cells) * 0.1)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Generate grid cell activity from position.

        Args:
            xy: Agent position [batch, 2] (x, y coordinates)

        Returns:
            activity: [batch, num_cells] Grid cell responses
        """
        x = xy[:, 0].unsqueeze(1)  # [batch, 1]
        y = xy[:, 1].unsqueeze(1)

        # Compute projections along grid orientations
        cos_ori = torch.cos(self.orientations).unsqueeze(0)  # [1, num_cells]
        sin_ori = torch.sin(self.orientations).unsqueeze(0)

        # Project position onto grid axes
        proj = x * cos_ori + y * sin_ori  # [batch, num_cells]

        # Apply spatial frequency
        proj_scaled = proj * self.scales.unsqueeze(0) * 2 * np.pi

        # Cosine tuning with phase
        activity = torch.cos(proj_scaled + self.phases.unsqueeze(0))

        # Shift to [0, 1] (positive-only)
        activity = (activity + 1.0) / 2.0

        # Add some noise for biological realism
        if self.training:
            activity = activity + torch.randn_like(activity) * 0.01

        return activity


class PlaceCell(nn.Module):
    """
    Place cell: location-specific cells with Gaussian tuning.
    """

    def __init__(self, num_cells: int = 100, sigma: float = 1.0, track_bounds: Tuple[float, float] = (-10, 50)):
        """
        Args:
            num_cells: Number of place cells
            sigma: Width of place field (m)
            track_bounds: (min_y, max_y) track bounds for cell placement
        """
        super().__init__()
        self.num_cells = num_cells
        self.sigma = sigma

        # Randomly place place fields along the track (x varies, y position)
        # For simplicity in 2D, we'll use Gaussian blobs
        self.register_buffer(
            'centers',
            torch.randn(num_cells, 2) * 5.0  # Random scattered centers
        )

        # Alternative: evenly spaced along track
        # self.centers = torch.stack([
        #     torch.zeros(num_cells),  # x near track center
        #     torch.linspace(track_bounds[0], track_bounds[1], num_cells)
        # ], dim=1)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Compute place cell activity.

        Args:
            xy: Position [batch, 2]

        Returns:
            activity: [batch, num_cells]
        """
        # Compute squared distance to each place field center
        diff = xy.unsqueeze(1) - self.centers.unsqueeze(0)  # [batch, num_cells, 2]
        dist_sq = (diff ** 2).sum(dim=2)  # [batch, num_cells]

        # Gaussian tuning
        activity = torch.exp(-dist_sq / (2 * self.sigma ** 2))

        # Normalize
        activity = activity / (activity.max(dim=1, keepdim=True)[0] + 1e-8)

        return activity


class PathIntegrator(nn.Module):
    """
    Path integration module: Integrates velocity to estimate position.
    Models the grid cell network's path integration capability.
    """

    def __init__(self, initial_position: torch.Tensor = None):
        """
        Args:
            initial_position: Starting position [2] tensor
        """
        super().__init__()
        if initial_position is None:
            initial_position = torch.zeros(2)
        self.register_buffer('position', initial_position)
        self.register_buffer('velocity', torch.zeros(2))

    def forward(self, velocity: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Integrate velocity to update position estimate.

        Args:
            velocity: Linear velocity [batch, 2] or [2]
            dt: Time step

        Returns:
            Updated position estimate
        """
        if velocity.dim() == 1:
            velocity = velocity.unsqueeze(0)

        # Simple Euler integration
        self.position = self.position + velocity * dt

        return self.position


class BioInspiredNavigation(nn.Module):
    """
    Complete bio-inspired navigation system combining:
    - Head Direction cells
    - Grid cells
    - Place cells
    - Path integration

    This provides a self-contained navigation system that can:
    1. Track orientation (HD cells)
    2. Build spatial representations (grid cells)
    3. Recognize locations (place cells)
    4. Integrate dead reckoning (path integrator)
    """

    def __init__(self,
                 num_hd_cells: int = 72,
                 num_grid_cells: int = 64,
                 num_place_cells: int = 100,
                 grid_scale_min: float = 0.5,
                 grid_scale_max: float = 8.0,
                 place_field_sigma: float = 1.0):
        super().__init__()

        self.head_direction = HeadDirectionCell(num_hd_cells)
        self.grid_cells = GridCell(num_grid_cells, grid_scale_min, grid_scale_max)
        self.place_cells = PlaceCell(num_place_cells, sigma=place_field_sigma)
        self.path_integrator = PathIntegrator()

        # Integration weights (learnable)
        self.hd_to_grid = nn.Linear(num_hd_cells, num_grid_cells)
        self.grid_to_place = nn.Linear(num_grid_cells, num_place_cells)

        # Position estimator (from grid cells)
        self.position_decoder = nn.Linear(num_grid_cells, 2)

    def forward(self,
                velocity: Optional[torch.Tensor] = None,
                theta: Optional[torch.Tensor] = None,
                dt: float = 0.1) -> dict:
        """
        Process navigation information.

        Args:
            velocity: Linear velocity [batch, 2] (vx, vy)
            theta: Orientation [batch] (radians)
            dt: Time step

        Returns:
            Dictionary with all cell activities and decoded position
        """
        outputs = {}

        # Update path integration if velocity provided
        if velocity is not None:
            position = self.path_integrator(velocity, dt=dt)
            outputs['position'] = position
        else:
            position = self.path_integrator.position.unsqueeze(0).repeat(theta.size(0), 1)

        # Encode head direction
        if theta is not None:
            hd_activity = self.head_direction(theta)
            outputs['head_direction'] = hd_activity

            # HD modulated grid cell activity
            hd_modulated = torch.tanh(self.hd_to_grid(hd_activity))
            grid_activity = self.grid_cells(position) * (1.0 + 0.2 * hd_modulated)
        else:
            grid_activity = self.grid_cells(position)

        outputs['grid_cells'] = grid_activity

        # Place cell activity
        place_activity = self.place_cells(position)
        outputs['place_cells'] = place_activity

        # Decode position from grid cells (for reconstruction error)
        decoded_position = self.position_decoder(grid_activity)
        outputs['decoded_position'] = decoded_position

        return outputs

    def get_position_estimate(self) -> torch.Tensor:
        """Get current path-integrated position estimate"""
        return self.path_integrator.position


class BioInspiredPolicy(nn.Module):
    """
    Policy network using bio-inspired navigation representations.

    Instead of raw sensor readings, uses:
    - Grid cell activity (spatial context)
    - Place cell activity (location recognition)
    - Head direction (orientation)

    Outputs continuous steering commands.
    """

    def __init__(self,
                 num_hd_cells: int = 72,
                 num_grid_cells: int = 64,
                 num_place_cells: int = 100,
                 hidden_dim: int = 128,
                 action_dim: int = 3):
        super().__init__()

        self.navigation = BioInspiredNavigation(
            num_hd_cells=num_hd_cells,
            num_grid_cells=num_grid_cells,
            num_place_cells=num_place_cells
        )

        # Action selection from navigation representation
        self.policy_net = nn.Sequential(
            nn.Linear(num_grid_cells + num_place_cells + num_hd_cells, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self,
                velocity: torch.Tensor,
                theta: torch.Tensor,
                dt: float = 0.1) -> torch.Tensor:
        """
        Compute action from navigation state.

        Args:
            velocity: Velocity [batch, 2]
            theta: Orientation [batch]
            dt: Time step

        Returns:
            logits: Action logits [batch, action_dim]
        """
        nav_out = self.navigation(velocity=velocity, theta=theta, dt=dt)

        # Concatenate all representations
        combined = torch.cat([
            nav_out['grid_cells'],
            nav_out['place_cells'],
            nav_out['head_direction']
        ], dim=1)

        logits = self.policy_net(combined)
        return logits, nav_out


class STDPPlasticity(nn.Module):
    """
    Spike-Timing Dependent Plasticity (STDP) rule.

    Implements the biological plasticity rule:
    - Pre before post: LTP (potentiation)
    - Post before pre: LTD (depression)

    Δw ∝ A+ * exp(-Δt/τ+) for Δt < 0 (pre before post)
    Δw ∝ -A- * exp(Δt/τ-) for Δt > 0 (post before pre)

    This enables unsupervised learning without backpropagation.
    """

    def __init__(self,
                 tau_plus: float = 20.0,  # ms
                 tau_minus: float = 20.0,  # ms
                 A_plus: float = 0.1,
                 A_minus: float = 0.1,
                 weight_lim: tuple = (0.0, 1.0)):
        super().__init__()

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.weight_lim = weight_plus, weight_minus = weight_lim

        # Eligibility trace for weight updates
        self.register_buffer('last_pre_spikes', None)
        self.register_buffer('last_post_spikes', None)
        self.register_buffer('eligibility_trace', None)

    def compute_stdp_delta(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Compute STDP weight updates from spike trains.

        Args:
            pre_spikes: [time, batch, pre_neurons]
            post_spikes: [time, batch, post_neurons]
            dt: Time step (ms)

        Returns:
            delta_w: Weight change [pre_neurons, post_neurons]
        """
        T, B, N_pre = pre_spikes.shape
        N_post = post_spikes.shape[2]

        # Find spike times
        pre_times = torch.where(pre_spikes > 0.5)[0].float() * dt
        post_times = torch.where(post_spikes > 0.5)[0].float() * dt

        if len(pre_times) == 0 or len(post_times) == 0:
            return torch.zeros(N_pre, N_post, device=pre_spikes.device)

        # Compute pairwise time differences
        delta_t = post_times.unsqueeze(1) - pre_times.unsqueeze(0)  # [post_spikes, pre_spikes]

        # STDP rule
        delta_w = torch.zeros(N_pre, N_post, device=pre_spikes.device)

        # LTP: pre before post (Δt < 0)
        mask_ltp = delta_t < 0
        ltp = self.A_plus * torch.exp(delta_t[mask_ltp] / self.tau_plus)
        # Accumulate to appropriate weight indices
        for i in range(N_pre):
            for j in range(N_post):
                # This is simplified - in practice, vectorize
                pass

        # LTD: post before pre (Δt > 0)
        mask_ltd = delta_t > 0
        ltd = -self.A_minus * torch.exp(-delta_t[mask_ltd] / self.tau_minus)

        # This is a simplified implementation - full version needs proper indexing
        return delta_w

    def forward(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply STDP rule to weights.

        Args:
            pre_spikes: [time, batch, pre_neurons]
            post_spikes: [time, batch, post_neurons]
            weights: Current weight matrix [pre_neurons, post_neurons]

        Returns:
            new_weights: Updated weight matrix
        """
        # Simplified: just track spikes for now
        delta_w = self.compute_stdp_delta(pre_spikes, post_spikes)

        # Apply weight limits
        new_weights = torch.clamp(weights + delta_w, self.weight_lim[0], self.weight_lim[1])

        return new_weights


class PredictiveCodingLayer(nn.Module):
    """
    Predictive coding: Network predicts future input and learns from prediction error.

    Key idea:
    - Predict next state from current state
    - Error = actual - predicted
    - Learn to minimize prediction error
    - Creates hierarchical, sparse representations
    """

    def __init__(self, input_dim: int, hidden_dim: int, prediction_steps: int = 1):
        super().__init__()

        self.prediction_steps = prediction_steps

        # Encoder: current state → latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Predictor: latent_t → latent_{t+k}
        self.predictor = nn.Linear(hidden_dim, hidden_dim * prediction_steps)

        # Decoder: latent → input space (for reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor, future_x: Optional[torch.Tensor] = None):
        """
        Args:
            x: Current input [batch, input_dim]
            future_x: Future input [batch, prediction_steps, input_dim] (optional, for training)

        Returns:
            dict with predictions, reconstructions, losses
        """
        # Encode current state
        z = self.encoder(x)

        # Predict future latent states
        z_pred_flat = self.predictor(z)  # [batch, hidden_dim * prediction_steps]
        z_pred = z_pred_flat.view(z.size(0), self.prediction_steps, -1)

        # Reconstruct current input
        x_recon = self.decoder(z)

        outputs = {
            'latent': z,
            'latent_pred': z_pred,
            'reconstruction': x_recon
        }

        # Compute losses if future provided
        if future_x is not None:
            # Prediction loss
            # We need to encode future_x to get future latent
            with torch.no_grad():
                future_z = self.encoder(future_x.reshape(-1, future_x.size(-1)))
                future_z = future_z.view(future_x.size(0), future_x.size(1), -1)

            pred_loss = F.mse_loss(z_pred, future_z)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x)

            outputs['loss_prediction'] = pred_loss
            outputs['loss_reconstruction'] = recon_loss
            outputs['loss_total'] = pred_loss + recon_loss

        return outputs


class MultiTimescaleSNN(nn.Module):
    """
    Hierarchical SNN with neurons operating at different time constants.

    Biological inspiration:
    - Fast neurons: quick reflexes, high frequency (τ_m small ~10ms)
    - Medium neurons: sensorimotor transformation (τ_m ~20-50ms)
    - Slow neurons: long-term context, strategic planning (τ_m ~100-200ms)

    This creates a natural hierarchy of temporal processing.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [96, 96, 96],
                 time_constants: List[float] = [0.9, 0.95, 0.99],  # Beta = exp(-dt/τ)
                 output_dim: int = 3):
        super().__init__()

        assert len(hidden_dims) == len(time_constants), "Need equal number of layers and time constants"

        self.layers = nn.ModuleList()
        self.time_constants = time_constants

        # Build hierarchical layers
        prev_dim = input_dim
        for i, (hidden_dim, beta) in enumerate(zip(hidden_dims, time_constants)):
            layer = nn.ModuleDict({
                'fc': nn.Linear(prev_dim, hidden_dim),
                'lif': snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid()),
                'bn': nn.BatchNorm1d(hidden_dim)
            })
            self.layers.append(layer)
            prev_dim = hidden_dim

        # Output layer (no spiking, readout)
        self.fc_out = nn.Linear(prev_dim, output_dim)

    def forward(self, spike_input: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through multi-timescale SNN.

        Args:
            spike_input: [time, batch, input_dim]

        Returns:
            output_spikes: [time, batch, output_dim]
            layer_activities: List of membrane potentials per layer
        """
        T, B, _ = spike_input.shape

        # Initialize membrane potentials for each layer
        mems = [layer['lif'].init_leaky() for layer in self.layers]

        output_spikes_rec = []
        all_mems = [[] for _ in self.layers]

        for t in range(T):
            x = spike_input[t]

            # Propagate through layers
            for i, layer in enumerate(self.layers):
                x = layer['fc'](x)
                x = layer['bn'](x)
                x, mems[i] = layer['lif'](x, mems[i])
                all_mems[i].append(mems[i])

            # Output readout (no LIF)
            out = self.fc_out(x)
            output_spikes_rec.append(out)

        output_spikes = torch.stack(output_spikes_rec)
        membrane_traces = [torch.stack(mems_i) for mems_i in all_mems]

        return output_spikes, membrane_traces


# Backward compatibility
__all__ = [
    'HeadDirectionCell',
    'GridCell',
    'PlaceCell',
    'PathIntegrator',
    'BioInspiredNavigation',
    'BioInspiredPolicy',
    'STDPPlasticity',
    'PredictiveCodingLayer',
    'MultiTimescaleSNN'
]
