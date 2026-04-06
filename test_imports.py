#!/usr/bin/env python3
"""
Quick test to validate the enhanced setup works correctly.
"""

import sys


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

    print("\n" + "="*60)
    if success:
        print("ALL TESTS PASSED! ✓")
        print("You can now run: python main_enhanced.py --config configs/default.yaml")
    else:
        print("SOME TESTS FAILED ✗")
        print("Please check the error messages above and install missing dependencies.")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    from pathlib import Path
    main()
