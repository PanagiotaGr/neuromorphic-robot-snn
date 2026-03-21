import csv
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import DATA_CFG
from simulator import ProceduralTrack, sense_track
from models import rate_encode
import snntorch as snn
from snntorch import surrogate


# ============================================================
# Continuous Benchmark Suite
# ------------------------------------------------------------
# Run from inside neuromorphic_robot/
#   source .venv/bin/activate
#   python continuous_benchmark.py
#
# What it does:
# - trains ANN and SNN continuous controllers
# - evaluates across many tracks and corruption settings
# - saves raw CSV, summary CSV, plots, and model weights
# ============================================================

SEED = 91
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path("outputs/continuous_benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# Config
# ============================================================
class Cfg:
    batch_size = 96
    epochs_ann = 20
    epochs_snn = 20
    lr = 1e-3
    hidden_dim = 96
    snn_steps = 30
    beta = 0.92

    n_train_tracks = 30
    n_test_tracks = 18
    samples_per_track = 500

    speed = 0.15
    max_turn_rate = 0.12
    max_steps = 360
    failure_distance = 1.8
    delay_steps = 0


CFG = Cfg()


# ============================================================
# Helpers
# ============================================================
def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))



def unpack_sensor_info(sensor_info, robot_x=None, robot_y=None):
    if isinstance(sensor_info, dict):
        return (
            np.asarray(sensor_info["values"], dtype=np.float32),
            sensor_info["starts"],
            sensor_info["ends"],
            sensor_info["hits"],
        )

    sensors = np.asarray(sensor_info, dtype=np.float32)
    n = len(sensors)
    if robot_x is None:
        robot_x = 0.0
    if robot_y is None:
        robot_y = 0.0
    starts = [(robot_x, robot_y)] * n
    ends = [(robot_x, robot_y)] * n
    hits = [(robot_x, robot_y)] * n
    return sensors, starts, ends, hits



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

    out = []
    for key, vals in grouped.items():
        row = {k: v for k, v in zip(group_keys, key)}
        row["n_episodes"] = len(vals)
        row["success_rate"] = float(np.mean([v["success"] for v in vals]))
        row["mean_lateral_error"] = float(np.mean([v["mean_lateral_error"] for v in vals]))
        row["std_lateral_error"] = float(np.std([v["mean_lateral_error"] for v in vals]))
        row["mean_steps"] = float(np.mean([v["steps"] for v in vals]))
        row["mean_activity"] = float(np.mean([v["mean_activity"] for v in vals]))
        row["mean_abs_steering"] = float(np.mean([v["mean_abs_steering"] for v in vals]))
        out.append(row)
    return out


# ============================================================
# Teacher and dataset
# ============================================================
def teacher_steering(track, x, y, theta):
    look_y = y + 0.9
    target_x = float(track.line_x(look_y))
    target_theta = float(track.tangent_theta(look_y))
    lateral_error = target_x - x
    heading_error = wrap_angle(target_theta - theta)
    raw = 0.95 * lateral_error + 0.75 * heading_error
    return float(np.clip(np.tanh(raw), -1.0, 1.0))



def generate_continuous_dataset(track_seeds, samples_per_track):
    xs = []
    ys = []

    for seed in track_seeds:
        track = ProceduralTrack(seed=seed, y_max=DATA_CFG.track_y_max)
        rng = np.random.default_rng(seed + 3000)

        for _ in range(samples_per_track):
            y = rng.uniform(0.2, DATA_CFG.track_y_max - 2.0)
            center_x = float(track.line_x(y))
            tangent = float(track.tangent_theta(y))
            x = center_x + rng.normal(0.0, 0.65)
            theta = tangent + rng.normal(0.0, 0.40)

            sensor_info = sense_track(track, x=x, y=y, theta=theta)
            sensors, _, _, _ = unpack_sensor_info(sensor_info, x, y)
            steer = teacher_steering(track, x, y, theta)

            xs.append(sensors)
            ys.append([steer])

    x = torch.tensor(np.array(xs), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    return x, y


# ============================================================
# Models
# ============================================================
class ANNContinuousController(torch.nn.Module):
    def __init__(self, input_dim=9, hidden_dim=96):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class SNNContinuousController(torch.nn.Module):
    def __init__(self, input_dim=9, hidden_dim=96, beta=0.92):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.readout = torch.nn.Linear(hidden_dim, 1)

    def forward(self, spike_input):
        mem1 = self.lif1.init_leaky()
        spk1_rec = []
        for t in range(spike_input.size(0)):
            cur1 = self.fc1(spike_input[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_rec.append(spk1)
        spk1_rec = torch.stack(spk1_rec)
        features = spk1_rec.mean(dim=0)
        steer = torch.tanh(self.readout(features))
        return steer, spk1_rec


# ============================================================
# Training
# ============================================================
def train_ann(model, train_loader, test_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = {"train_loss": [], "test_loss": []}

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item() * yb.size(0)
            n += yb.size(0)
        return total_loss / n

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
            n += yb.size(0)

        train_loss = total_loss / n
        test_loss = evaluate(test_loader)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        print(f"[ANN-CONT] Epoch {epoch+1:02d}/{epochs} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f}")

    return history



def train_snn(model, train_loader, test_loader, epochs, lr, num_steps):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = {"train_loss": [], "test_loss": []}

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            spk_in = rate_encode(xb, num_steps).to(device)
            pred, _ = model(spk_in)
            loss = loss_fn(pred, yb)
            total_loss += loss.item() * yb.size(0)
            n += yb.size(0)
        return total_loss / n

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            spk_in = rate_encode(xb, num_steps).to(device)
            pred, _ = model(spk_in)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
            n += yb.size(0)

        train_loss = total_loss / n
        test_loss = evaluate(test_loader)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        print(f"[SNN-CONT] Epoch {epoch+1:02d}/{epochs} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f}")

    return history


# ============================================================
# Policies
# ============================================================
class ANNContinuousPolicy:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def act(self, sensor_values):
        self.model.eval()
        x_np = np.asarray(sensor_values, dtype=np.float32)[None, :]
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        steer = self.model(x).item()
        return float(np.clip(steer, -1.0, 1.0)), abs(float(steer))


class SNNContinuousPolicy:
    def __init__(self, model, num_steps):
        self.model = model
        self.num_steps = num_steps

    @torch.no_grad()
    def act(self, sensor_values):
        self.model.eval()
        x_np = np.asarray(sensor_values, dtype=np.float32)[None, :]
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        spk_in = rate_encode(x, self.num_steps).to(device)
        steer, spk_hidden = self.model(spk_in)
        spike_activity = float(spk_hidden.sum().item())
        return float(np.clip(steer.item(), -1.0, 1.0)), spike_activity


# ============================================================
# Continuous simulator
# ============================================================
class ContinuousRobot:
    def __init__(self, x, y, theta, speed, max_turn_rate):
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed
        self.max_turn_rate = max_turn_rate

    def step(self, steering):
        steering = float(np.clip(steering, -1.0, 1.0))
        self.theta += steering * self.max_turn_rate
        self.x += self.speed * np.cos(self.theta)
        self.y += self.speed * np.sin(self.theta)



def corrupt_sensors(sensors, noise_std=0.0, dropout_prob=0.0, dead_sensor_index=-1):
    x = np.array(sensors, dtype=np.float32).copy()
    if noise_std > 0:
        x += np.random.normal(0.0, noise_std, size=x.shape)
    if dropout_prob > 0:
        mask = np.random.rand(*x.shape) < dropout_prob
        x[mask] = 0.0
    if 0 <= dead_sensor_index < len(x):
        x[dead_sensor_index] = 0.0
    return np.clip(x, 0.0, 1.0)



def run_episode(track, policy, noise_std=0.0, delay_steps=0, dropout_prob=0.0, dead_sensor_index=-1):
    start_y = 0.5
    start_x = float(track.line_x(start_y)) + 0.25
    start_theta = float(track.tangent_theta(start_y))
    robot = ContinuousRobot(start_x, start_y, start_theta, CFG.speed, CFG.max_turn_rate)

    sensor_buffer = []
    lateral_errors = []
    steer_values = []
    activities = []
    failure = False

    for _ in range(CFG.max_steps):
        sensor_info = sense_track(track, robot.x, robot.y, robot.theta)
        sensors, _, _, _ = unpack_sensor_info(sensor_info, robot.x, robot.y)
        sensors = corrupt_sensors(sensors, noise_std=noise_std, dropout_prob=dropout_prob, dead_sensor_index=dead_sensor_index)

        sensor_buffer.append(sensors)
        if delay_steps > 0 and len(sensor_buffer) > delay_steps:
            used_sensors = sensor_buffer[-(delay_steps + 1)]
        else:
            used_sensors = sensors

        steering, activity = policy.act(used_sensors)
        robot.step(steering)

        track_x = float(track.line_x(robot.y))
        lateral_error = abs(robot.x - track_x)

        lateral_errors.append(lateral_error)
        steer_values.append(steering)
        activities.append(activity)

        if lateral_error > CFG.failure_distance:
            failure = True
            break
        if robot.y >= track.y_max:
            break

    metrics = {
        "steps": len(lateral_errors),
        "success": int((not failure) and (robot.y >= track.y_max)),
        "final_y": float(robot.y),
        "mean_lateral_error": float(np.mean(lateral_errors)) if lateral_errors else np.nan,
        "max_lateral_error": float(np.max(lateral_errors)) if lateral_errors else np.nan,
        "mean_abs_steering": float(np.mean(np.abs(steer_values))) if steer_values else np.nan,
        "mean_activity": float(np.mean(activities)) if activities else np.nan,
    }
    return metrics


# ============================================================
# Plotting
# ============================================================
def plot_training_curves(ann_hist, snn_hist):
    plt.figure(figsize=(8, 4.5))
    plt.plot(ann_hist["test_loss"], linewidth=2, label="ANN test loss")
    plt.plot(snn_hist["test_loss"], linewidth=2, label="SNN test loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Continuous Controllers: Test Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "continuous_test_loss.png", dpi=180)
    plt.close()



def plot_success_vs_noise(summary_rows):
    plt.figure(figsize=(8, 4.5))
    for policy in sorted(set(r["policy"] for r in summary_rows)):
        subset = [r for r in summary_rows if r["policy"] == policy and r["delay_steps"] == 0 and r["dropout_prob"] == 0.0 and r["dead_sensor_index"] == -1]
        subset = sorted(subset, key=lambda r: r["noise_std"])
        xs = [r["noise_std"] for r in subset]
        ys = [r["success_rate"] for r in subset]
        plt.plot(xs, ys, marker="o", linewidth=2, label=policy)
    plt.xlabel("Noise std")
    plt.ylabel("Success rate")
    plt.title("Continuous Benchmark: Success vs Noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "success_vs_noise.png", dpi=180)
    plt.close()



def plot_error_vs_noise(summary_rows):
    plt.figure(figsize=(8, 4.5))
    for policy in sorted(set(r["policy"] for r in summary_rows)):
        subset = [r for r in summary_rows if r["policy"] == policy and r["delay_steps"] == 0 and r["dropout_prob"] == 0.0 and r["dead_sensor_index"] == -1]
        subset = sorted(subset, key=lambda r: r["noise_std"])
        xs = [r["noise_std"] for r in subset]
        ys = [r["mean_lateral_error"] for r in subset]
        plt.plot(xs, ys, marker="o", linewidth=2, label=policy)
    plt.xlabel("Noise std")
    plt.ylabel("Mean lateral error")
    plt.title("Continuous Benchmark: Error vs Noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "error_vs_noise.png", dpi=180)
    plt.close()



def plot_steps_vs_condition(summary_rows):
    plt.figure(figsize=(11, 4.8))
    policies = sorted(set(r["policy"] for r in summary_rows))
    conditions = sorted(set(r["condition"] for r in summary_rows))
    x = np.arange(len(conditions))
    width = 0.35

    for i, policy in enumerate(policies):
        vals = []
        for cond in conditions:
            subset = [r for r in summary_rows if r["policy"] == policy and r["condition"] == cond]
            vals.append(float(np.mean([s["mean_steps"] for s in subset])) if subset else np.nan)
        plt.bar(x + (i - 0.5) * width, vals, width=width, label=policy)

    plt.xticks(x, conditions, rotation=20, ha="right")
    plt.ylabel("Mean steps")
    plt.title("Continuous Benchmark: Mean Steps by Condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_steps_by_condition.png", dpi=180)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    train_track_seeds = list(range(100, 100 + CFG.n_train_tracks))
    test_track_seeds = list(range(1000, 1000 + CFG.n_test_tracks))

    print("Generating continuous dataset...")
    x_train, y_train = generate_continuous_dataset(train_track_seeds, CFG.samples_per_track)
    x_test, y_test = generate_continuous_dataset(test_track_seeds, CFG.samples_per_track // 2)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=CFG.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=CFG.batch_size, shuffle=False)

    print("\nTraining ANN continuous controller...")
    ann_model = ANNContinuousController(input_dim=DATA_CFG.num_sensors, hidden_dim=CFG.hidden_dim).to(device)
    ann_hist = train_ann(ann_model, train_loader, test_loader, CFG.epochs_ann, CFG.lr)

    print("\nTraining SNN continuous controller...")
    snn_model = SNNContinuousController(input_dim=DATA_CFG.num_sensors, hidden_dim=CFG.hidden_dim, beta=CFG.beta).to(device)
    snn_hist = train_snn(snn_model, train_loader, test_loader, CFG.epochs_snn, CFG.lr, CFG.snn_steps)

    ann_policy = ANNContinuousPolicy(ann_model)
    snn_policy = SNNContinuousPolicy(snn_model, CFG.snn_steps)

    plot_training_curves(ann_hist, snn_hist)

    conditions = [
        ("noise_0.00", {"noise_std": 0.00, "delay_steps": 0, "dropout_prob": 0.0, "dead_sensor_index": -1}),
        ("noise_0.04", {"noise_std": 0.04, "delay_steps": 0, "dropout_prob": 0.0, "dead_sensor_index": -1}),
        ("noise_0.08", {"noise_std": 0.08, "delay_steps": 0, "dropout_prob": 0.0, "dead_sensor_index": -1}),
        ("noise_0.12", {"noise_std": 0.12, "delay_steps": 0, "dropout_prob": 0.0, "dead_sensor_index": -1}),
        ("delay_2", {"noise_std": 0.00, "delay_steps": 2, "dropout_prob": 0.0, "dead_sensor_index": -1}),
        ("delay_4", {"noise_std": 0.00, "delay_steps": 4, "dropout_prob": 0.0, "dead_sensor_index": -1}),
        ("dropout_0.10", {"noise_std": 0.00, "delay_steps": 0, "dropout_prob": 0.10, "dead_sensor_index": -1}),
        ("dropout_0.20", {"noise_std": 0.00, "delay_steps": 0, "dropout_prob": 0.20, "dead_sensor_index": -1}),
        ("dead_center", {"noise_std": 0.00, "delay_steps": 0, "dropout_prob": 0.0, "dead_sensor_index": DATA_CFG.num_sensors // 2}),
    ]

    raw_rows = []

    for cond_name, cond_cfg in conditions:
        print(f"\nEvaluating condition: {cond_name}")
        for seed in test_track_seeds:
            track = ProceduralTrack(seed=seed, y_max=DATA_CFG.track_y_max)
            ann_metrics = run_episode(track, ann_policy, **cond_cfg)
            snn_metrics = run_episode(track, snn_policy, **cond_cfg)

            raw_rows.append({
                "policy": "ANN",
                "condition": cond_name,
                "track_seed": seed,
                **cond_cfg,
                **ann_metrics,
            })
            raw_rows.append({
                "policy": "SNN",
                "condition": cond_name,
                "track_seed": seed,
                **cond_cfg,
                **snn_metrics,
            })

    summary_rows = summarize_rows(
        raw_rows,
        ["policy", "condition", "noise_std", "delay_steps", "dropout_prob", "dead_sensor_index"],
    )

    save_csv(raw_rows, OUT_DIR / "continuous_benchmark_raw.csv")
    save_csv(summary_rows, OUT_DIR / "continuous_benchmark_summary.csv")

    plot_success_vs_noise(summary_rows)
    plot_error_vs_noise(summary_rows)
    plot_steps_vs_condition(summary_rows)

    torch.save(ann_model.state_dict(), OUT_DIR / "ann_continuous_benchmark.pt")
    torch.save(snn_model.state_dict(), OUT_DIR / "snn_continuous_benchmark.pt")

    print("\nSaved files:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f" - {p.name}")

    best_rows = sorted(
        summary_rows,
        key=lambda r: (r["success_rate"], -r["mean_lateral_error"], r["mean_steps"]),
        reverse=True,
    )
    print("\nTop summary row:")
    print(best_rows[0])


if __name__ == "__main__":
    main()
