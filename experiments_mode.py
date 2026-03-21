import csv
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import DATA_CFG, TRAIN_CFG, SimConfig
from dataset import generate_supervised_dataset
from models import ANNController, SNNController, ANNPolicy, SNNPolicy
from train import train_ann, train_snn
from evaluate import evaluate_policy


# ============================================================
# Experiments Mode
# ------------------------------------------------------------
# Run from inside neuromorphic_robot/
#   source .venv/bin/activate
#   python experiments_mode.py
#
# What it does:
# - trains ANN and SNN models
# - evaluates across multiple corruption conditions
# - runs multiple SNN ablations
# - saves CSV summaries and plots in outputs/experiments/
# ============================================================

SEED = 21
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path("outputs/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
# Small utility
# ------------------------------------------------------------
def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows, group_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    summary = []
    for key, vals in grouped.items():
        out = {k: v for k, v in zip(group_keys, key)}
        out["n_episodes"] = len(vals)
        out["success_rate"] = float(np.mean([r["success"] for r in vals]))
        out["mean_lateral_error"] = float(np.mean([r["mean_lateral_error"] for r in vals]))
        out["mean_steps"] = float(np.mean([r["steps"] for r in vals]))
        out["mean_activity"] = float(np.mean([r["mean_activity"] for r in vals]))
        summary.append(out)
    return summary


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_success_vs_noise(summary_rows, filename):
    plt.figure(figsize=(8, 4.5))
    policies = sorted(set(r["policy"] for r in summary_rows))
    for policy in policies:
        subset = [r for r in summary_rows if r["policy"] == policy and r["delay_steps"] == 0 and r["dropout_prob"] == 0.0 and r["dead_sensor_index"] == -1]
        subset = sorted(subset, key=lambda x: x["noise_std"])
        xs = [r["noise_std"] for r in subset]
        ys = [r["success_rate"] for r in subset]
        plt.plot(xs, ys, marker="o", linewidth=2, label=policy)
    plt.xlabel("Noise std")
    plt.ylabel("Success rate")
    plt.title("Success Rate vs Sensor Noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()


def plot_error_vs_noise(summary_rows, filename):
    plt.figure(figsize=(8, 4.5))
    policies = sorted(set(r["policy"] for r in summary_rows))
    for policy in policies:
        subset = [r for r in summary_rows if r["policy"] == policy and r["delay_steps"] == 0 and r["dropout_prob"] == 0.0 and r["dead_sensor_index"] == -1]
        subset = sorted(subset, key=lambda x: x["noise_std"])
        xs = [r["noise_std"] for r in subset]
        ys = [r["mean_lateral_error"] for r in subset]
        plt.plot(xs, ys, marker="o", linewidth=2, label=policy)
    plt.xlabel("Noise std")
    plt.ylabel("Mean lateral error")
    plt.title("Tracking Error vs Sensor Noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()


def plot_snn_ablation(summary_rows, filename, metric_key="success_rate"):
    plt.figure(figsize=(9, 4.8))
    configs = sorted(set((r["hidden_dim"], r["snn_steps"]) for r in summary_rows))
    labels = [f"H{h}-T{t}" for h, t in configs]
    xs = np.arange(len(configs))
    ys = []
    for cfg in configs:
        subset = [r for r in summary_rows if (r["hidden_dim"], r["snn_steps"]) == cfg and r["noise_std"] == 0.08]
        ys.append(float(np.mean([r[metric_key] for r in subset])) if subset else np.nan)
    plt.bar(xs, ys)
    plt.xticks(xs, labels, rotation=20, ha="right")
    plt.ylabel(metric_key.replace("_", " "))
    plt.title(f"SNN Ablation at Noise=0.08 ({metric_key.replace('_', ' ')})")
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()


# ------------------------------------------------------------
# Model training wrapper
# ------------------------------------------------------------
def train_models(hidden_dim, snn_steps, beta):
    train_track_seeds = list(range(100, 100 + DATA_CFG.n_train_tracks))
    test_track_seeds = list(range(1000, 1000 + DATA_CFG.n_test_tracks))

    print("Generating datasets...")
    x_train, y_train = generate_supervised_dataset(train_track_seeds, DATA_CFG.samples_per_track)
    x_test, y_test = generate_supervised_dataset(test_track_seeds, DATA_CFG.samples_per_track // 2)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=TRAIN_CFG.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=TRAIN_CFG.batch_size, shuffle=False)

    ann_model = ANNController(input_dim=DATA_CFG.num_sensors, hidden_dim=hidden_dim).to(device)
    ann_hist = train_ann(ann_model, train_loader, test_loader, TRAIN_CFG.ann_epochs, TRAIN_CFG.lr, device)

    snn_model = SNNController(input_dim=DATA_CFG.num_sensors, hidden_dim=hidden_dim, beta=beta).to(device)
    snn_hist = train_snn(snn_model, train_loader, test_loader, TRAIN_CFG.snn_epochs, TRAIN_CFG.lr, snn_steps, device)

    ann_policy = ANNPolicy(ann_model, device)
    snn_policy = SNNPolicy(snn_model, snn_steps, device)

    return {
        "ann_model": ann_model,
        "snn_model": snn_model,
        "ann_policy": ann_policy,
        "snn_policy": snn_policy,
        "ann_hist": ann_hist,
        "snn_hist": snn_hist,
        "test_track_seeds": test_track_seeds,
    }


# ------------------------------------------------------------
# Main experiments
# ------------------------------------------------------------
def run_baseline_experiments():
    print("\n=== BASELINE EXPERIMENTS ===")
    trained = train_models(
        hidden_dim=TRAIN_CFG.hidden_dim,
        snn_steps=TRAIN_CFG.snn_steps,
        beta=TRAIN_CFG.beta,
    )

    conditions = []
    for noise in [0.00, 0.04, 0.08, 0.12]:
        conditions.append((f"noise_{noise:.2f}", SimConfig(noise_std=noise)))
    conditions.extend([
        ("delay_2", SimConfig(delay_steps=2)),
        ("delay_4", SimConfig(delay_steps=4)),
        ("dropout_010", SimConfig(sensor_dropout_prob=0.10)),
        ("dropout_020", SimConfig(sensor_dropout_prob=0.20)),
        ("dead_center", SimConfig(dead_sensor_index=DATA_CFG.num_sensors // 2)),
    ])

    raw_rows = []
    for cond_name, sim_cfg in conditions:
        print(f"Evaluating {cond_name}")
        ann_rows = evaluate_policy("ANN", trained["ann_policy"], trained["test_track_seeds"], sim_cfg, DATA_CFG)
        snn_rows = evaluate_policy("SNN", trained["snn_policy"], trained["test_track_seeds"], sim_cfg, DATA_CFG)
        for r in ann_rows:
            r["condition"] = cond_name
        for r in snn_rows:
            r["condition"] = cond_name
        raw_rows.extend(ann_rows)
        raw_rows.extend(snn_rows)

    summary_rows = summarize_rows(
        raw_rows,
        ["policy", "condition", "noise_std", "delay_steps", "dropout_prob", "dead_sensor_index"],
    )

    save_csv(raw_rows, OUT_DIR / "baseline_raw_metrics.csv")
    save_csv(summary_rows, OUT_DIR / "baseline_summary.csv")
    plot_success_vs_noise(summary_rows, OUT_DIR / "success_vs_noise.png")
    plot_error_vs_noise(summary_rows, OUT_DIR / "error_vs_noise.png")

    torch.save(trained["ann_model"].state_dict(), OUT_DIR / "baseline_ann.pt")
    torch.save(trained["snn_model"].state_dict(), OUT_DIR / "baseline_snn.pt")

    return summary_rows



def run_snn_ablations():
    print("\n=== SNN ABLATIONS ===")
    ablation_settings = [
        {"hidden_dim": 64, "snn_steps": 25, "beta": 0.90},
        {"hidden_dim": 96, "snn_steps": 25, "beta": 0.92},
        {"hidden_dim": 96, "snn_steps": 40, "beta": 0.92},
        {"hidden_dim": 128, "snn_steps": 40, "beta": 0.95},
    ]

    noise_conditions = [0.00, 0.08, 0.12]
    all_rows = []

    for i, cfg in enumerate(ablation_settings, start=1):
        print(f"Ablation {i}/{len(ablation_settings)}: {cfg}")
        trained = train_models(
            hidden_dim=cfg["hidden_dim"],
            snn_steps=cfg["snn_steps"],
            beta=cfg["beta"],
        )

        for noise in noise_conditions:
            sim_cfg = SimConfig(noise_std=noise)
            rows = evaluate_policy("SNN", trained["snn_policy"], trained["test_track_seeds"], sim_cfg, DATA_CFG)
            for r in rows:
                r["hidden_dim"] = cfg["hidden_dim"]
                r["snn_steps"] = cfg["snn_steps"]
                r["beta"] = cfg["beta"]
            all_rows.extend(rows)

    summary_rows = summarize_rows(all_rows, ["hidden_dim", "snn_steps", "beta", "noise_std"])
    save_csv(all_rows, OUT_DIR / "snn_ablation_raw.csv")
    save_csv(summary_rows, OUT_DIR / "snn_ablation_summary.csv")
    plot_snn_ablation(summary_rows, OUT_DIR / "snn_ablation_success.png", metric_key="success_rate")
    plot_snn_ablation(summary_rows, OUT_DIR / "snn_ablation_error.png", metric_key="mean_lateral_error")

    return summary_rows



def main():
    baseline_summary = run_baseline_experiments()
    ablation_summary = run_snn_ablations()

    print("\nSaved experiment files:")
    for path in sorted(OUT_DIR.iterdir()):
        print(f" - {path.name}")

    best_baseline = sorted(
        [r for r in baseline_summary if r["policy"] == "SNN"],
        key=lambda r: (r["success_rate"], -r["mean_lateral_error"]),
        reverse=True,
    )[0]
    print("\nBest SNN baseline condition summary:")
    print(best_baseline)

    best_ablation = sorted(
        ablation_summary,
        key=lambda r: (r["success_rate"], -r["mean_lateral_error"]),
        reverse=True,
    )[0]
    print("\nBest SNN ablation summary:")
    print(best_ablation)


if __name__ == "__main__":
    main()
