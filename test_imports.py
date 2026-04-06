#!/usr/bin/env python3
"""
Quick test to validate the enhanced setup works correctly.
"""

import os
import sys
import torch

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False

    try:
        import snntorch
        print(f"  ✓ snnTorch")
    except ImportError as e:
        print(f"  ✗ snnTorch: {e}")
        return False

    try:
        import yaml
        print(f"  ✓ PyYAML")
    except ImportError as e:
        print(f"  ✗ PyYAML: {e}")
        return False

    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Matplotlib: {e}")
        return False

    try:
        from utils import checkpoint, early_stopping, scheduler, energy, statistics
        print(f"  ✓ Utils package")
    except ImportError as e:
        print(f"  ✗ Utils package: {e}")
        return False

    try:
        from config_enhanced import Config, parse_cli_args, setup_logging
        print(f"  ✓ Config system")
    except ImportError as e:
        print(f"  ✗ Config system: {e}")
        return False

    try:
        from models import ANNController, SNNController, DeepSNNController, multi_step_encode
        print(f"  ✓ Models with encodings")
    except ImportError as e:
        print(f"  ✗ Models: {e}")
        return False

    print("\nAll imports successful! ✓")
    return True


def test_config():
    """Test configuration system"""
    print("\nTesting configuration...")

    from config_enhanced import Config

    # Test default config
    config = Config()
    print(f"  ✓ Default config created")
    print(f"    - Model: {config.model.hidden_dim} hidden dim")
    print(f"    - Encoding: {config.encoding.encoding_type}")
    print(f"    - Data: {config.data.num_sensors} sensors")

    # Test config from YAML
    import yaml
    test_yaml = Path("test_config.yaml")
    if test_yaml.exists():
        config2 = Config.from_yaml(test_yaml)
        print(f"  ✓ YAML config loaded")

    return True


def test_models():
    """Test model creation and forward pass"""
    print("\nTesting models...")

    import torch
    from models import ANNController, SNNController, DeepSNNController, rate_encode

    device = torch.device("cpu")
    input_dim = 9
    batch_size = 4

    # Test ANN
    ann = ANNController(input_dim=input_dim, hidden_dim=32, output_dim=3).to(device)
    dummy_input = torch.randn(batch_size, input_dim).to(device)
    output = ann(dummy_input)
    assert output.shape == (batch_size, 3), f"ANN output shape mismatch: {output.shape}"
    print(f"  ✓ ANN: forward pass works, {sum(p.numel() for p in ann.parameters())} params")

    # Test SNN
    snn = SNNController(input_dim=input_dim, hidden_dim=32, output_dim=3, beta=0.9).to(device)
    num_steps = 10
    spike_input = rate_encode(dummy_input, num_steps)
    spike_out, mem_out = snn(spike_input)
    assert spike_out.shape[0] == num_steps, f"SNN time steps mismatch"
    print(f"  ✓ SNN: forward pass works, {sum(p.numel() for p in snn.parameters())} params")

    # Test DeepSNN
    deep_snn = DeepSNNController(input_dim=input_dim, hidden_dim=64, output_dim=3).to(device)
    spike_out, mem_out = deep_snn(spike_input)
    print(f"  ✓ DeepSNN: forward pass works, {sum(p.numel() for p in deep_snn.parameters())} params")

    return True


def test_bio_inspired():
    """Test novel bio-inspired modules"""
    print("\nTesting bio-inspired modules...")

    import torch
    from bio_inspired import (
        BioInspiredPolicy,
        MultiTimescaleSNN,
        NeuromodulatedSNN,
        AttentionDrivenSensing,
        WorldModelSNN
    )

    device = torch.device("cpu")
    batch_size = 4

    # Test BioInspiredPolicy
    print("  Testing BioInspiredPolicy...")
    policy = BioInspiredPolicy(num_hd_cells=36, num_grid_cells=32, num_place_cells=50,
                               hidden_dim=64, action_dim=3).to(device)
    velocity = torch.randn(batch_size, 2, device=device)
    theta = torch.rand(batch_size, device=device) * 2 * torch.pi
    logits, nav_out = policy(velocity, theta)
    print(f"    ✓ BioInspiredPolicy: logits shape {logits.shape}")

    # Test MultiTimescaleSNN
    print("  Testing MultiTimescaleSNN...")
    multi = MultiTimescaleSNN(input_dim=9, hidden_dims=[32, 32, 32],
                             time_constants=[0.9, 0.95, 0.99], output_dim=3).to(device)
    spikes = torch.rand(10, batch_size, 9, device=device) > 0.5
    spikes = spikes.float()
    out, mem_traces = multi(spikes)
    print(f"    ✓ MultiTimescaleSNN: output shape {out.shape}, {len(mem_traces)} layers")

    # Test NeuromodulatedSNN
    print("  Testing NeuromodulatedSNN...")
    neuromod = NeuromodulatedSNN(input_dim=9, hidden_dim=32, output_dim=3,
                                 use_neuromodulation=True).to(device)
    out = neuromod(spikes, compute_neuromodulation=True)
    print(f"    ✓ NeuromodulatedSNN: output shape {out.shape}")

    # Test AttentionDrivenSensing
    print("  Testing AttentionDrivenSensing...")
    ads = AttentionDrivenSensing(base_num_sensors=9, max_virtual_sensors=9).to(device)
    base_sens = torch.rand(batch_size, 9, device=device)
    base_angles = torch.linspace(-torch.pi/4, torch.pi/4, 9, device=device)
    enhanced = ads(base_sens, base_angles)
    print(f"    ✓ AttentionDrivenSensing: {base_sens.shape} → {enhanced.shape}")

    # Test WorldModelSNN
    print("  Testing WorldModelSNN...")
    wm = WorldModelSNN(state_dim=9, action_dim=3).to(device)
    state = torch.randn(batch_size, 9, device=device)
    action = torch.randn(batch_size, 3, device=device)
    next_pred, uncert = wm.transition(state, action)
    print(f"    ✓ WorldModelSNN: prediction shape {next_pred.shape}, uncertainty {uncert.mean().item():.4f}")

    print("  ✓ All bio-inspired modules working!")
    return True


def test_multi_agent():
    """Test multi-agent coordination"""
    print("\nTesting multi-agent coordination...")

    from multi_agent.coordination import MultiAgentSNN, emergent_flocking

    num_agents = 5
    model = MultiAgentSNN(
        num_agents=num_agents,
        agent_config={'input_dim': 6, 'hidden_dim': 16, 'output_dim': 16},
        enable_communication=True,
        enable_consensus=False
    )

    batch_size = 2
    observations = [torch.randn(batch_size, 6) for _ in range(num_agents)]
    outputs, info = model(observations)

    print(f"  ✓ Multi-agent forward: {len(outputs)} agents, output shape {outputs[0].shape}")
    if 'communication' in info:
        print(f"  ✓ Communication graph computed")

    return True


def test_checkpoint():
    """Test checkpoint manager"""
    print("\nTesting checkpoint manager...")

    from utils import CheckpointManager
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp()
    try:
        manager = CheckpointManager(Path(tmpdir), metric_name="test_acc", mode="max")
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())

        # Simulate saving a checkpoint
        metrics = {"epoch": 1, "test_acc": 0.75}
        manager.save(model, optimizer, 1, metrics)

        print(f"  ✓ Checkpoint saved")
        assert manager.best_checkpoint is not None, "Best checkpoint not set"
        print(f"  ✓ Best checkpoint tracking works")

    finally:
        shutil.rmtree(tmpdir)

    return True


def main():
    print("="*60)
    print("NEUROMORPHIC ROBOT SNN - ENHANCED SETUP VALIDATION")
    print("="*60)

    success = True

    if not test_imports():
        success = False

    if success and not test_config():
        success = False

    if success and not test_models():
        success = False

    if success and not test_checkpoint():
        success = False

    # Test novel bio-inspired features
    if success:
        try:
            if not test_bio_inspired():
                success = False
        except Exception as e:
            print(f"  ✗ Bio-inspired tests failed: {e}")
            success = False

    # Test multi-agent
    if success:
        try:
            if not test_multi_agent():
                success = False
        except Exception as e:
            print(f"  ✗ Multi-agent tests failed: {e}")
            success = False

    print("\n" + "="*60)
    if success:
        print("ALL TESTS PASSED! ✓")
        print("\nYou can now run:")
        print("  python run.py --config configs/default.yaml")
        print("  python run_all_experiments.py                    # Test all novel features")
        print("  python experimental/novel_experiments.py --all  # Run experiments")
    else:
        print("SOME TESTS FAILED ✗")
        print("Please check the error messages above.")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    from pathlib import Path
    main()
