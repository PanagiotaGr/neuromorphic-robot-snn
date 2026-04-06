"""
World Model & Planning - "Think Before You Act"

NOVEL: SNN that builds internal model of environment and plans
by imagining consequences before taking action.

Key ideas:
- Transition model: predict next state given current state + action
- Reward model: predict expected reward
- Planning: simulate multiple futures, choose best action
- Model-based RL with SNN dynamics
- "Dreaming" or "imagining" rollouts

This enables:
- Sample-efficient learning (fewer real interactions)
- Safety: test actions in mental simulation first
- Explaining decisions: visualize imagined futures
- Transfer: learned world model works across tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import snntorch as snn
from snntorch import surrogate


class SNNTransitionModel(nn.Module):
    """
    Learn environment dynamics using SNN.

    f(s_t, a_t) → s_{t+1}

    Uses SNN for temporal processing of state-action pairs.
    Can handle stochastic transitions (multiple possible next states).
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 snn_steps: int = 15):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.snn_steps = snn_steps

        # Encode state-action pair
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)

        # SNN for temporal dynamics
        self.snn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid()),
            nn.Linear(hidden_dim, hidden_dim),
            snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        )

        # Decode to next state
        self.state_decoder = nn.Linear(hidden_dim, state_dim)

        # Uncertainty estimator (predict variance)
        self.uncertainty_head = nn.Linear(hidden_dim, state_dim)

    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state.

        Args:
            state: [batch, state_dim] current state
            action: [batch, action_dim] one-hot or continuous action

        Returns:
            next_state_mean: [batch, state_dim]
            uncertainty: [batch, state_dim] predictive variance
        """
        # Encode
        state_enc = F.relu(self.state_encoder(state))
        action_enc = F.relu(self.action_encoder(action))
        combined = torch.cat([state_enc, action_enc], dim=-1)

        # Unfold over SNN timesteps
        batch_size = state.size(0)
        combined = combined.unsqueeze(0).repeat(self.snn_steps, 1, 1)

        # SNN forward
        mem1 = self.snn[1].init_leaky()
        mem2 = self.snn[3].init_leaky()
        for t in range(self.snn_steps):
            cur = self.snn[0](combined[t])
            spk1, mem1 = self.snn[1](cur, mem1)
            cur2 = self.snn[2](spk1)
            spk2, mem2 = self.snn[3](cur2, mem2)

        # Final hidden state
        final_hidden = mem2

        # Decode
        next_state = self.state_decoder(final_hidden)
        uncertainty = F.softplus(self.uncertainty_head(final_hidden))

        return next_state, uncertainty

    def sample_next_state(self,
                          state: torch.Tensor,
                          action: torch.Tensor) -> torch.Tensor:
        """Sample next state from predicted distribution"""
        mean, var = self.forward(state, action)
        std = torch.sqrt(var + 1e-8)
        return mean + torch.randn_like(mean) * std


class SNNRewardModel(nn.Module):
    """
    Predict reward given state (or state-action).

    r(s_t, a_t) → reward

    Can be used for:
    - Intrinsic motivation (curiosity)
    - Value estimation
    - Planning objective
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int = 0,  # Optional
                 hidden_dim: int = 64):
        super().__init__()

        input_dim = state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self,
                state: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict reward.

        Args:
            state: [batch, state_dim]
            action: [batch, action_dim] (optional)

        Returns:
            reward: [batch, 1]
        """
        if action is not None:
            x = torch.cat([state, action], dim=-1)
        else:
            x = state

        return self.net(x)


class PlanningHorizon(nn.Module):
    """
    Perform mental simulation/planning over multiple steps.

    Algorithm:
    1. For each candidate action, simulate multiple futures
    2. Evaluate each trajectory using reward model
    3. Choose action with best predicted cumulative reward

    This is Model-Based RL with SNN dynamics.
    """

    def __init__(self,
                 world_model: SNNTransitionModel,
                 reward_model: SNNRewardModel,
                 planning_horizon: int = 10,
                 num_candidates: int = 20):
        super().__init__()

        self.world_model = world_model
        self.reward_model = reward_model
        self.horizon = planning_horizon
        self.num_candidates = num_candidates

    def plan_action(self,
                    current_state: torch.Tensor,
                    action_space) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Plan best action via Monte Carlo rollouts.

        Args:
            current_state: [batch, state_dim]
            action_space: Function that returns random actions [batch, action_dim]

        Returns:
            best_action: [batch, action_dim]
            planned_value: [batch] expected cumulative reward
        """
        batch_size = current_state.size(0)
        device = current_state.device

        # Generate candidate action sequences
        # For simplicity: sample actions, simulate each
        trajectories = []
        cumulative_rewards = []

        for _ in range(self.num_candidates):
            # Sample action sequence
            actions = torch.stack([action_space() for _ in range(self.horizon)], dim=1)  # [B, H, A]

            # Simulate
            states = [current_state]
            for t in range(self.horizon):
                next_state = self.world_model.sample_next_state(states[-1], actions[:, t])
                states.append(next_state)

            states = torch.stack(states[1:], dim=1)  # [B, H, S]

            # Compute rewards
            rewards = []
            for t in range(self.horizon):
                r = self.reward_model(states[:, t], actions[:, t])
                rewards.append(r)
            rewards = torch.stack(rewards, dim=1).squeeze(-1)  # [B, H]

            cumulative_reward = rewards.sum(dim=1)  # [B]
            trajectories.append(actions)
            cumulative_rewards.append(cumulative_reward)

        # Stack candidates
        cumulative_rewards = torch.stack(cumulative_rewards, dim=1)  # [B, num_candidates]

        # Select best
        best_idx = cumulative_rewards.argmax(dim=1)
        best_actions = torch.stack([trajectories[i][torch.arange(batch_size), best_idx[i]]
                                    for i in range(batch_size)])

        best_value = cumulative_rewards[torch.arange(batch_size), best_idx]

        return best_actions, best_value

    def plan_with_value_iteration(self,
                                  state: torch.Tensor,
                                  discrete_actions: torch.Tensor) -> torch.Tensor:
        """
        Value iteration planning (for discrete small action spaces).

        Args:
            state: Current state
            discrete_actions: [num_actions, action_dim] all possible actions

        Returns:
            Q-values: [batch, num_actions]
        """
        num_actions = discrete_actions.size(0)
        batch_size = state.size(0)
        device = state.device

        Q = torch.zeros(batch_size, num_actions, device=device)

        for a_idx in range(num_actions):
            action = discrete_actions[a_idx].unsqueeze(0).expand(batch_size, -1)
            # One-step lookahead
            next_state = self.world_model.sample_next_state(state, action)
            immediate_reward = self.reward_model(state, action).squeeze(-1)

            # Assume zero future value for simplicity (could iterate)
            Q[:, a_idx] = immediate_reward

        return Q


class WorldModelSNN(nn.Module):
    """
    Complete world model package: transition + reward + planning.

    Usage:
    1. Train world model on real transitions (offline)
    2. Use for planning during policy execution
    3. Can also collect imagined data for policy learning (Dreamer)
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 snn_steps: int = 15,
                 planning_horizon: int = 10):
        super().__init__()

        self.transition = SNNTransitionModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            snn_steps=snn_steps
        )

        self.reward = SNNRewardModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim // 2
        )

        self.planner = PlanningHorizon(
            world_model=self.transition,
            reward_model=self.reward,
            planning_horizon=planning_horizon,
            num_candidates=30
        )

    def plan(self,
             current_state: torch.Tensor,
             action_space_fn) -> Tuple[torch.Tensor, torch.Tensor]:
        """High-level planning interface"""
        return self.planner.plan_action(current_state, action_space_fn)

    def imagine_rollout(self,
                        initial_state: torch.Tensor,
                        actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Simulate a trajectory (for data augmentation or policy learning).

        Args:
            initial_state: [batch, state_dim]
            actions: [batch, horizon, action_dim]

        Returns:
            Dict with states, rewards, next_states
        """
        horizon = actions.size(1)
        states = [initial_state]
        rewards = []

        for t in range(horizon):
            next_state = self.transition.sample_next_state(states[-1], actions[:, t])
            reward = self.reward_model(states[-1], actions[:, t])
            states.append(next_state)
            rewards.append(reward)

        return {
            'states': torch.stack(states[:-1], dim=1),  # [B, H, S]
            'next_states': torch.stack(states[1:], dim=1),
            'rewards': torch.stack(rewards, dim=1).squeeze(-1)  # [B, H]
        }

    def compute_model_loss(self,
                           real_states: torch.Tensor,
                           actions: torch.Tensor,
                           next_states: torch.Tensor,
                           rewards: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss for training world model on real data.

        Args:
            real_states: [batch, state_dim]
            actions: [batch, action_dim]
            next_states: [batch, state_dim]
            rewards: [batch]

        Returns:
            loss_dict with transition_loss, reward_loss, total_loss
        """
        # Predict next state
        pred_next_state, uncertainty = self.transition(real_states, actions)

        # Transition loss (negative log-likelihood)
        trans_loss = F.mse_loss(pred_next_state, next_states)

        # Reward prediction
        pred_reward = self.reward_model(real_states, actions).squeeze(-1)
        reward_loss = F.mse_loss(pred_reward, rewards)

        total_loss = trans_loss + reward_loss

        return {
            'transition_loss': trans_loss,
            'reward_loss': reward_loss,
            'total_loss': total_loss,
            'uncertainty_mean': uncertainty.mean()
        }


# Test
def test_world_model():
    print("="*70)
    print("TEST: World Model & Planning")
    print("="*70)

    batch_size = 8
    state_dim = 9  # Sensor readings
    action_dim = 3  # Discrete actions (one-hot would be 3, but use continuous for simplicity)

    # Test SNNTransitionModel
    print("\n1. Testing SNNTransitionModel...")
    trans = SNNTransitionModel(state_dim=state_dim, action_dim=action_dim, snn_steps=10)

    state = torch.randn(batch_size, state_dim)
    action = torch.randn(batch_size, action_dim)

    next_state, uncertainty = trans(state, action)
    print(f"   Next state shape: {next_state.shape}")
    print(f"   Uncertainty shape: {uncertainty.shape}")
    print(f"   Uncertainty mean: {uncertainty.mean().item():.4f}")

    sampled = trans.sample_next_state(state, action)
    print(f"   Sampled next state shape: {sampled.shape}")
    print("   ✓ Transition model working")

    # Test SNNRewardModel
    print("\n2. Testing SNNRewardModel...")
    reward = SNNRewardModel(state_dim=state_dim, action_dim=action_dim)

    r = reward(state, action)
    print(f"   Reward prediction shape: {r.shape}")
    print(f"   Reward mean: {r.mean().item():.4f}")
    print("   ✓ Reward model working")

    # Test WorldModelSNN
    print("\n3. Testing WorldModelSNN...")
    wm = WorldModelSNN(
        state_dim=state_dim,
        action_dim=action_dim,
        planning_horizon=5
    )

    # Compute loss
    real_next = torch.randn(batch_size, state_dim)
    rewards = torch.rand(batch_size)

    loss_dict = wm.compute_model_loss(state, action, real_next, rewards)
    print(f"   Transition loss: {loss_dict['transition_loss'].item():.4f}")
    print(f"   Reward loss: {loss_dict['reward_loss'].item():.4f}")

    # Test planning
    print("\n4. Testing PlanningHorizon...")
    def random_action_space():
        return torch.randn(batch_size, action_dim)

    best_action, best_value = wm.plan(state, random_action_space)
    print(f"   Best action shape: {best_action.shape}")
    print(f"   Best value shape: {best_value.shape}")
    print(f"   Best value mean: {best_value.mean().item():.4f}")

    # Test imagination
    print("\n5. Testing imagination rollout...")
    horizon = 5
    actions_seq = torch.randn(batch_size, horizon, action_dim)
    imagined = wm.imagine_rollout(state, actions_seq)
    print(f"   Imagined states shape: {imagined['states'].shape}")
    print(f"   Imagined rewards shape: {imagined['rewards'].shape}")
    print("   ✓ World model imagination working")

    print("\n✅ All world model components functional!")
    return True


if __name__ == "__main__":
    test_world_model()
