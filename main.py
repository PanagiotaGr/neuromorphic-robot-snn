import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import DATA_CFG, TRAIN_CFG, SimConfig
from dataset import generate_supervised_dataset
from models import ANNController, SNNController, ANNPolicy, SNNPolicy
from train import train_ann, train_snn
from simulator import ProceduralTrack, run_episode
from evaluate import evaluate_policy, save_eval_csv
from plots import plot_training_curves, plot_episode, plot_bar_summary, make_animation


SEED = 13
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    train_track_seeds = list(range(100, 100 + DATA_CFG.n_train_tracks))
    test_track_seeds = list(range(1000, 1000 + DATA_CFG.n_test_tracks))

    print("Generating datasets...")
    x_train, y_train = generate_supervised_dataset(train_track_seeds, DATA_CFG.samples_per_track)
    x_test, y_test = generate_supervised_dataset(test_track_seeds, DATA_CFG.samples_per_track // 2)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=TRAIN_CFG.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=TRAIN_CFG.batch_size, shuffle=False)

    print("\nTraining ANN...")
    ann_model = ANNController(input_dim=DATA_CFG.num_sensors, hidden_dim=TRAIN_CFG.hidden_dim).to(device)
    ann_hist = train_ann(ann_model, train_loader, test_loader, TRAIN_CFG.ann_epochs, TRAIN_CFG.lr, device)

    print("\nTraining SNN...")
    snn_model = SNNController(input_dim=DATA_CFG.num_sensors, hidden_dim=TRAIN_CFG.hidden_dim, beta=TRAIN_CFG.beta).to(device)
    snn_hist = train_snn(snn_model, train_loader, test_loader, TRAIN_CFG.snn_epochs, TRAIN_CFG.lr, TRAIN_CFG.snn_steps, device)

    ann_policy = ANNPolicy(ann_model, device)
    snn_policy = SNNPolicy(snn_model, TRAIN_CFG.snn_steps, device)

    plot_training_curves(ann_hist, snn_hist)

    conditions = [
        ("clean", SimConfig(noise_std=0.00, delay_steps=0, sensor_dropout_prob=0.00, dead_sensor_index=-1)),
        ("noise", SimConfig(noise_std=0.08, delay_steps=0, sensor_dropout_prob=0.00, dead_sensor_index=-1)),
        ("delay", SimConfig(noise_std=0.00, delay_steps=2, sensor_dropout_prob=0.00, dead_sensor_index=-1)),
        ("dropout", SimConfig(noise_std=0.00, delay_steps=0, sensor_dropout_prob=0.18, dead_sensor_index=-1)),
        ("dead_sensor", SimConfig(noise_std=0.00, delay_steps=0, sensor_dropout_prob=0.00, dead_sensor_index=4)),
    ]

    eval_rows = []
    for cond_name, sim_cfg in conditions:
        print(f"\nEvaluating condition: {cond_name}")
        eval_rows.extend(evaluate_policy("ANN", ann_policy, test_track_seeds, sim_cfg, DATA_CFG))
        eval_rows.extend(evaluate_policy("SNN", snn_policy, test_track_seeds, sim_cfg, DATA_CFG))

    save_eval_csv(eval_rows)
    plot_bar_summary(eval_rows)

    demo_track = ProceduralTrack(seed=2026, y_max=DATA_CFG.track_y_max)
    _, ann_demo_hist, ann_demo_metrics = run_episode(demo_track, ann_policy, SimConfig(), record=True)
    _, snn_demo_hist, snn_demo_metrics = run_episode(demo_track, snn_policy, SimConfig(), record=True)
    plot_episode(demo_track, ann_demo_hist, snn_demo_hist, title_suffix="clean")

    _, ann_noise_hist, _ = run_episode(demo_track, ann_policy, SimConfig(noise_std=0.08), record=True)
    _, snn_noise_hist, _ = run_episode(demo_track, snn_policy, SimConfig(noise_std=0.08), record=True)
    plot_episode(demo_track, ann_noise_hist, snn_noise_hist, title_suffix="noise")

    make_animation(demo_track, snn_demo_hist, "snn_demo_clean.gif", "SNN Demo - Clean")
    make_animation(demo_track, snn_noise_hist, "snn_demo_noise.gif", "SNN Demo - Noisy Sensors")

    torch.save(ann_model.state_dict(), OUT_DIR / "ann_controller.pt")
    torch.save(snn_model.state_dict(), OUT_DIR / "snn_controller.pt")

    print("\nSample episode metrics:")
    print(f"ANN clean: {ann_demo_metrics}")
    print(f"SNN clean: {snn_demo_metrics}")
    print("\nDone. Files saved in outputs/")


if __name__ == "__main__":
    main()
