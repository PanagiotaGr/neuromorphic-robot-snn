#!/usr/bin/env bash
set -e

# ============================================================
# Visual upgrade for the neuromorphic_robot project
# Run inside the folder that contains neuromorphic_robot/
#
# Example:
#   chmod +x apply_visual_upgrade.sh
#   ./apply_visual_upgrade.sh
#   cd neuromorphic_robot
#   source .venv/bin/activate
#   python main.py
# ============================================================

cd neuromorphic_robot

cat > simulator.py <<'PY'
import math
from collections import deque

import numpy as np

from config import DATA_CFG, SimConfig

ACTION_TO_NAME = {
    0: "turn_left",
    1: "forward",
    2: "turn_right",
}


def make_sensor_angles(num_sensors, arc_deg):
    arc_rad = np.deg2rad(arc_deg)
    return np.linspace(-arc_rad / 2.0, arc_rad / 2.0, num_sensors)


SENSOR_ANGLES = make_sensor_angles(DATA_CFG.num_sensors, DATA_CFG.sensor_arc_deg)


class ProceduralTrack:
    def __init__(self, seed, y_max=40.0):
        self.seed = seed
        self.y_max = y_max
        rng = np.random.default_rng(seed)
        self.a1 = rng.uniform(0.7, 1.7)
        self.a2 = rng.uniform(0.3, 1.1)
        self.a3 = rng.uniform(0.0, 0.7)
        self.f1 = rng.uniform(0.08, 0.20)
        self.f2 = rng.uniform(0.03, 0.10)
        self.f3 = rng.uniform(0.015, 0.05)
        self.p1 = rng.uniform(0.0, 2 * np.pi)
        self.p2 = rng.uniform(0.0, 2 * np.pi)
        self.p3 = rng.uniform(0.0, 2 * np.pi)
        self.drift = rng.uniform(-0.025, 0.025)

    def line_x(self, y):
        y = np.asarray(y)
        return (
            self.a1 * np.sin(self.f1 * y + self.p1)
            + self.a2 * np.sin(self.f2 * y + self.p2)
            + self.a3 * np.sin(self.f3 * y + self.p3)
            + self.drift * y
        )

    def line_dxdy(self, y):
        y = np.asarray(y)
        return (
            self.a1 * self.f1 * np.cos(self.f1 * y + self.p1)
            + self.a2 * self.f2 * np.cos(self.f2 * y + self.p2)
            + self.a3 * self.f3 * np.cos(self.f3 * y + self.p3)
            + self.drift
        )

    def tangent_theta(self, y):
        dx = self.line_dxdy(y)
        return np.arctan2(np.ones_like(np.asarray(dx)), dx)


class Robot:
    def __init__(self, x, y, theta, speed, turn_rate):
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed
        self.turn_rate = turn_rate
        self.body_radius = 0.22
        self.sensor_range = DATA_CFG.lookahead
        self.path_x = [x]
        self.path_y = [y]
        self.theta_hist = [theta]

    def step(self, action, actuator_noise=0.0):
        noise = np.random.normal(0.0, actuator_noise)
        if action == 0:
            self.theta += self.turn_rate + noise
        elif action == 2:
            self.theta -= self.turn_rate + noise
        else:
            self.theta += 0.2 * noise

        self.x += self.speed * math.cos(self.theta)
        self.y += self.speed * math.sin(self.theta)
        self.path_x.append(self.x)
        self.path_y.append(self.y)
        self.theta_hist.append(self.theta)


def get_sensor_geometry(x, y, theta, lookahead=None):
    if lookahead is None:
        lookahead = DATA_CFG.lookahead

    starts = []
    ends = []
    for ang in SENSOR_ANGLES:
        phi = theta + ang
        starts.append((x, y))
        ends.append((x + lookahead * math.cos(phi), y + lookahead * math.sin(phi)))
    return starts, ends



def sense_track(track, x, y, theta, sigma=None, lookahead=None):
    if sigma is None:
        sigma = DATA_CFG.line_sigma
    if lookahead is None:
        lookahead = DATA_CFG.lookahead

    vals = []
    starts = []
    ends = []
    hit_points = []

    for ang in SENSOR_ANGLES:
        phi = theta + ang
        sx = x + lookahead * math.cos(phi)
        sy = y + lookahead * math.sin(phi)
        target_x = float(track.line_x(sy))
        dist = abs(sx - target_x)
        val = math.exp(-(dist ** 2) / (2 * sigma ** 2))
        vals.append(val)
        starts.append((x, y))
        ends.append((sx, sy))
        hit_points.append((target_x, sy))

    return {
        "values": np.clip(np.array(vals, dtype=np.float32), 0.0, 1.0),
        "starts": starts,
        "ends": ends,
        "hits": hit_points,
    }



def corrupt_sensors(values, cfg: SimConfig):
    x = np.array(values, dtype=np.float32).copy()
    if cfg.noise_std > 0:
        x += np.random.normal(0.0, cfg.noise_std, size=x.shape)
    if cfg.sensor_dropout_prob > 0:
        mask = np.random.rand(*x.shape) < cfg.sensor_dropout_prob
        x[mask] = 0.0
    if 0 <= cfg.dead_sensor_index < len(x):
        x[cfg.dead_sensor_index] = 0.0
    return np.clip(x, 0.0, 1.0)



def run_episode(track, policy, sim_cfg: SimConfig, record=False):
    start_y = 0.5
    start_x = float(track.line_x(start_y)) + 0.25
    start_theta = float(track.tangent_theta(start_y))
    robot = Robot(start_x, start_y, start_theta, sim_cfg.speed, sim_cfg.turn_rate)

    sensor_buffer = deque(maxlen=max(1, sim_cfg.delay_steps + 1))
    hist = {
        "x": [],
        "y": [],
        "theta": [],
        "track_x": [],
        "actions": [],
        "scores": [],
        "sensors": [],
        "true_sensors": [],
        "lateral_error": [],
        "activity": [],
        "sensor_starts": [],
        "sensor_ends": [],
        "sensor_hits": [],
    }
    failure = False

    for step in range(sim_cfg.max_steps):
        sensor_info = sense_track(track, robot.x, robot.y, robot.theta)
        true_sensors = sensor_info["values"]
        corrupted = corrupt_sensors(true_sensors, sim_cfg)
        sensor_buffer.append(corrupted)

        if sim_cfg.delay_steps > 0 and len(sensor_buffer) > sim_cfg.delay_steps:
            used_sensors = sensor_buffer[0]
        else:
            used_sensors = corrupted

        action, scores, activity = policy.act(used_sensors)
        robot.step(action, actuator_noise=0.01 if sim_cfg.noise_std > 0 else 0.0)

        track_x = float(track.line_x(robot.y))
        lateral_error = abs(robot.x - track_x)

        if record:
            hist["x"].append(robot.x)
            hist["y"].append(robot.y)
            hist["theta"].append(robot.theta)
            hist["track_x"].append(track_x)
            hist["actions"].append(action)
            hist["scores"].append(np.array(scores))
            hist["sensors"].append(used_sensors.copy())
            hist["true_sensors"].append(true_sensors.copy())
            hist["lateral_error"].append(lateral_error)
            hist["activity"].append(activity)
            hist["sensor_starts"].append(sensor_info["starts"])
            hist["sensor_ends"].append(sensor_info["ends"])
            hist["sensor_hits"].append(sensor_info["hits"])

        if lateral_error > sim_cfg.failure_distance:
            failure = True
            break
        if robot.y >= track.y_max:
            break

    metrics = {
        "steps": len(hist["y"]) if record else step + 1,
        "success": int((not failure) and (robot.y >= track.y_max)),
        "final_y": robot.y,
        "mean_lateral_error": float(np.mean(hist["lateral_error"])) if record and hist["lateral_error"] else float("nan"),
        "max_lateral_error": float(np.max(hist["lateral_error"])) if record and hist["lateral_error"] else float("nan"),
        "mean_activity": float(np.mean(hist["activity"])) if record and hist["activity"] else float("nan"),
    }
    return robot, hist, metrics
PY

cat > plots.py <<'PY'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from pathlib import Path

from simulator import ACTION_TO_NAME

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def plot_training_curves(ann_hist, snn_hist):
    plt.figure(figsize=(8, 4))
    plt.plot(ann_hist["test_acc"], label="ANN test acc")
    plt.plot(snn_hist["test_acc"], label="SNN test acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ANN vs SNN Test Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_accuracy_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(ann_hist["train_loss"], label="ANN train loss")
    plt.plot(snn_hist["train_loss"], label="SNN train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ANN vs SNN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_loss_comparison.png", dpi=180)
    plt.close()



def plot_episode(track, ann_hist, snn_hist, title_suffix="clean"):
    y_line = np.linspace(0.0, track.y_max, 800)
    x_line = track.line_x(y_line)

    plt.figure(figsize=(7, 10))
    plt.plot(x_line, y_line, linestyle="--", linewidth=2, label="track centerline")
    plt.plot(ann_hist["x"], ann_hist["y"], linewidth=2, label="ANN path")
    plt.plot(snn_hist["x"], snn_hist["y"], linewidth=2, label="SNN path")
    plt.scatter([ann_hist["x"][0]], [ann_hist["y"][0]], s=50, marker="o", label="start")
    plt.scatter([snn_hist["x"][-1]], [snn_hist["y"][-1]], s=70, marker="x", label="SNN end")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Trajectory Comparison ({title_suffix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"trajectory_comparison_{title_suffix}.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(ann_hist["lateral_error"], linewidth=2, label="ANN lateral error")
    plt.plot(snn_hist["lateral_error"], linewidth=2, label="SNN lateral error")
    plt.xlabel("Step")
    plt.ylabel("|x - x_track|")
    plt.title(f"Lateral Error ({title_suffix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"lateral_error_{title_suffix}.png", dpi=180)
    plt.close()



def plot_bar_summary(eval_rows):
    policies = sorted(set(r["policy"] for r in eval_rows))
    conditions = sorted(set((r["noise_std"], r["delay_steps"], r["dropout_prob"], r["dead_sensor_index"]) for r in eval_rows))
    cond_labels = [f"n={c[0]} d={c[1]} p={c[2]} dead={c[3]}" for c in conditions]
    width = 0.35
    x = np.arange(len(conditions))

    success_vals = {p: [] for p in policies}
    error_vals = {p: [] for p in policies}
    for cond in conditions:
        for p in policies:
            subset = [r for r in eval_rows if r["policy"] == p and (r["noise_std"], r["delay_steps"], r["dropout_prob"], r["dead_sensor_index"]) == cond]
            success_vals[p].append(np.mean([r["success"] for r in subset]))
            error_vals[p].append(np.mean([r["mean_lateral_error"] for r in subset]))

    plt.figure(figsize=(12, 4.5))
    for i, p in enumerate(policies):
        plt.bar(x + (i - 0.5) * width, success_vals[p], width=width, label=p)
    plt.xticks(x, cond_labels, rotation=20, ha="right")
    plt.ylabel("Success rate")
    plt.title("Policy Success Rate Across Conditions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "success_rate_summary.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 4.5))
    for i, p in enumerate(policies):
        plt.bar(x + (i - 0.5) * width, error_vals[p], width=width, label=p)
    plt.xticks(x, cond_labels, rotation=20, ha="right")
    plt.ylabel("Mean lateral error")
    plt.title("Mean Lateral Error Across Conditions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "lateral_error_summary.png", dpi=180)
    plt.close()



def plot_sensor_panel(hist, filename):
    sensors = np.array(hist["sensors"])
    plt.figure(figsize=(9, 4))
    plt.imshow(sensors.T, aspect="auto", origin="lower")
    plt.xlabel("Step")
    plt.ylabel("Sensor index")
    plt.title("Sensor Activation Over Time")
    plt.colorbar(label="Activation")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=180)
    plt.close()



def plot_activity_panel(hist, filename, title):
    plt.figure(figsize=(8, 4))
    plt.plot(hist["activity"], linewidth=2, label="activity")
    plt.plot(hist["lateral_error"], linewidth=2, label="lateral error")
    plt.xlabel("Step")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=180)
    plt.close()



def make_animation(track, hist, filename, title):
    y_line = np.linspace(0.0, track.y_max, 800)
    x_line = track.line_x(y_line)

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_xlim(min(np.min(x_line), np.min(hist["x"])) - 2.0, max(np.max(x_line), np.max(hist["x"])) + 2.0)
    ax.set_ylim(0.0, track.y_max + 1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    ax.plot(x_line, y_line, linestyle="--", linewidth=2, label="track centerline")
    path_plot, = ax.plot([], [], linewidth=2, label="robot path")
    heading_plot, = ax.plot([], [], linewidth=2)

    robot_patch = Circle((0, 0), radius=0.22, fill=False, linewidth=2)
    ax.add_patch(robot_patch)

    sensor_lines = [ax.plot([], [], linewidth=1.5, alpha=0.9)[0] for _ in range(len(hist["sensor_starts"][0]))]
    hit_points = [ax.plot([], [], marker="o", markersize=4)[0] for _ in range(len(hist["sensor_hits"][0]))]

    txt = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", alpha=0.88),
    )
    ax.legend(loc="lower right")

    def update(frame):
        x = hist["x"][frame]
        y = hist["y"][frame]
        theta = hist["theta"][frame]

        path_plot.set_data(hist["x"][:frame+1], hist["y"][:frame+1])
        robot_patch.center = (x, y)

        hx = [x, x + 0.45 * np.cos(theta)]
        hy = [y, y + 0.45 * np.sin(theta)]
        heading_plot.set_data(hx, hy)

        sensor_vals = hist["sensors"][frame]
        starts = hist["sensor_starts"][frame]
        ends = hist["sensor_ends"][frame]
        hits = hist["sensor_hits"][frame]

        for i, (line, start, end, hit) in enumerate(zip(sensor_lines, starts, ends, hits)):
            line.set_data([start[0], end[0]], [start[1], end[1]])
            alpha = 0.25 + 0.75 * float(sensor_vals[i])
            line.set_alpha(alpha)
            hit_points[i].set_data([hit[0]], [hit[1]])
            hit_points[i].set_alpha(alpha)

        act = ACTION_TO_NAME[hist["actions"][frame]]
        txt.set_text(
            f"step: {frame}\n"
            f"action: {act}\n"
            f"error: {hist['lateral_error'][frame]:.3f}\n"
            f"activity: {hist['activity'][frame]:.2f}\n"
            f"center sensor: {sensor_vals[len(sensor_vals)//2]:.2f}"
        )

        artists = [path_plot, heading_plot, robot_patch, txt]
        artists.extend(sensor_lines)
        artists.extend(hit_points)
        return tuple(artists)

    anim = FuncAnimation(fig, update, frames=len(hist["x"]), interval=55, repeat=False)
    anim.save(OUT_DIR / filename, writer="pillow", fps=18)
    plt.close(fig)
PY

cat > main.py <<'PY'
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
from plots import (
    plot_training_curves,
    plot_episode,
    plot_bar_summary,
    plot_sensor_panel,
    plot_activity_panel,
    make_animation,
)

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
    plot_sensor_panel(snn_demo_hist, "snn_sensor_panel_clean.png")
    plot_activity_panel(snn_demo_hist, "snn_activity_clean.png", "SNN Activity vs Lateral Error (Clean)")

    _, ann_noise_hist, _ = run_episode(demo_track, ann_policy, SimConfig(noise_std=0.08), record=True)
    _, snn_noise_hist, _ = run_episode(demo_track, snn_policy, SimConfig(noise_std=0.08), record=True)
    plot_episode(demo_track, ann_noise_hist, snn_noise_hist, title_suffix="noise")
    plot_sensor_panel(snn_noise_hist, "snn_sensor_panel_noise.png")
    plot_activity_panel(snn_noise_hist, "snn_activity_noise.png", "SNN Activity vs Lateral Error (Noise)")

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
PY

echo "Visual upgrade applied."
echo "Now run: cd neuromorphic_robot && source .venv/bin/activate && python main.py"
