import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import DATA_CFG
from simulator import ProceduralTrack, sense_track
from models import rate_encode
import snntorch as snn
from snntorch import surrogate


# ============================================================
# Continuous Steering Extension
# ------------------------------------------------------------
# Goal:
#   Learn continuous steering instead of 3 discrete actions.
#
# Output:
#   steering value in [-1, 1]
#   negative -> turn right
#   positive -> turn left
#
# Run:
#   source .venv/bin/activate
#   python continuous_steering.py
# ============================================================

SEED = 77
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path("outputs/continuous_steering")
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
    n_test_tracks = 12
    samples_per_track = 500

    speed = 0.15
    max_turn_rate = 0.12
    max_steps = 360
    failure_distance = 1.8


CFG = Cfg()


# ============================================================
# Helpers
# ============================================================
def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))



def unpack_sensor_info(sensor_info, robot_x=None, robot_y=None):
    """
    Compatibility helper:
    - If simulator.sense_track returns a dict, use it directly.
    - If it returns only a numpy array, fabricate minimal geometry so plots still run.
    """
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


# ============================================================
# Teacher for continuous steering
# ============================================================
def teacher_steering(track, x, y, theta):
    """
    Continuous control target in [-1, 1].
    Combines lateral and heading error.
    """
    look_y = y + 0.9
    target_x = float(track.line_x(look_y))
    target_theta = float(track.tangent_theta(look_y))

    lateral_error = target_x - x
    heading_error = wrap_angle(target_theta - theta)

    raw = 0.95 * lateral_error + 0.75 * heading_error
    steer = np.tanh(raw)
    return float(np.clip(steer, -1.0, 1.0))


# ============================================================
# Dataset
# ============================================================
def generate_continuous_dataset(track_seeds, samples_per_track):
    xs = []
    ys = []

    for seed in track_seeds:
        track = ProceduralTrack(seed=seed, y_max=DATA_CFG.track_y_max)
        rng = np.random.default_rng(seed + 2000)

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
class ANNContinuousController(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class SNNContinuousController(nn.Module):
    """
    Hidden LIF layer + linear readout from accumulated spike features.
    This is better suited for regression than a spiking 3-class head.
    """
    def __init__(self, input_dim=9, hidden_dim=96, beta=0.92):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, spike_input):
        mem1 = self.lif1.init_leaky()
        spk1_rec = []

        for t in range(spike_input.size(0)):
            cur1 = self.fc1(spike_input[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_rec.append(spk1)

        spk1_rec = torch.stack(spk1_rec)   # [T, B, H]
        features = spk1_rec.mean(dim=0)    # [B, H]
        steer = torch.tanh(self.readout(features))
        return steer, spk1_rec


# ============================================================
# Train loops
# ============================================================
def train_ann(model, train_loader, test_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
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
        print(f"[ANN-REG] Epoch {epoch+1:02d}/{epochs} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f}")

    return history



def train_snn(model, train_loader, test_loader, epochs, lr, num_steps):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
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
        print(f"[SNN-REG] Epoch {epoch+1:02d}/{epochs} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f}")

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
# Continuous robot simulator
# ============================================================
class ContinuousRobot:
    def __init__(self, x, y, theta, speed, max_turn_rate):
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed
        self.max_turn_rate = max_turn_rate
        self.path_x = [x]
        self.path_y = [y]
        self.theta_hist = [theta]

    def step(self, steering):
        steering = float(np.clip(steering, -1.0, 1.0))
        self.theta += steering * self.max_turn_rate
        self.x += self.speed * np.cos(self.theta)
        self.y += self.speed * np.sin(self.theta)
        self.path_x.append(self.x)
        self.path_y.append(self.y)
        self.theta_hist.append(self.theta)



def run_episode(track, policy, noise_std=0.0, record=True):
    start_y = 0.5
    start_x = float(track.line_x(start_y)) + 0.25
    start_theta = float(track.tangent_theta(start_y))
    robot = ContinuousRobot(start_x, start_y, start_theta, CFG.speed, CFG.max_turn_rate)

    hist = {
        "x": [],
        "y": [],
        "theta": [],
        "track_x": [],
        "lateral_error": [],
        "steering": [],
        "activity": [],
        "sensors": [],
        "sensor_starts": [],
        "sensor_ends": [],
        "sensor_hits": [],
    }

    failure = False

    for _ in range(CFG.max_steps):
        sensor_info = sense_track(track, robot.x, robot.y, robot.theta)
        sensors, sensor_starts, sensor_ends, sensor_hits = unpack_sensor_info(sensor_info, robot.x, robot.y)

        if noise_std > 0:
            sensors = np.clip(sensors + np.random.normal(0.0, noise_std, size=sensors.shape), 0.0, 1.0)

        steering, activity = policy.act(sensors)
        robot.step(steering)

        track_x = float(track.line_x(robot.y))
        lateral_error = abs(robot.x - track_x)

        if record:
            hist["x"].append(robot.x)
            hist["y"].append(robot.y)
            hist["theta"].append(robot.theta)
            hist["track_x"].append(track_x)
            hist["lateral_error"].append(lateral_error)
            hist["steering"].append(steering)
            hist["activity"].append(activity)
            hist["sensors"].append(sensors.copy())
            hist["sensor_starts"].append(sensor_starts)
            hist["sensor_ends"].append(sensor_ends)
            hist["sensor_hits"].append(sensor_hits)

        if lateral_error > CFG.failure_distance:
            failure = True
            break
        if robot.y >= track.y_max:
            break

    metrics = {
        "steps": len(hist["y"]),
        "success": int((not failure) and (robot.y >= track.y_max)),
        "final_y": robot.y,
        "mean_lateral_error": float(np.mean(hist["lateral_error"])) if hist["lateral_error"] else np.nan,
        "max_lateral_error": float(np.max(hist["lateral_error"])) if hist["lateral_error"] else np.nan,
        "mean_abs_steering": float(np.mean(np.abs(hist["steering"]))) if hist["steering"] else np.nan,
        "mean_activity": float(np.mean(hist["activity"])) if hist["activity"] else np.nan,
    }
    return hist, metrics


# ============================================================
# Plots
# ============================================================
def plot_training(ann_hist, snn_hist):
    plt.figure(figsize=(8, 4))
    plt.plot(ann_hist["train_loss"], label="ANN train")
    plt.plot(ann_hist["test_loss"], label="ANN test")
    plt.plot(snn_hist["train_loss"], label="SNN train")
    plt.plot(snn_hist["test_loss"], label="SNN test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Continuous Steering Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "continuous_training_loss.png", dpi=180)
    plt.close()



def plot_trajectory(track, ann_hist, snn_hist, suffix):
    y_line = np.linspace(0.0, track.y_max, 800)
    x_line = track.line_x(y_line)

    plt.figure(figsize=(7, 10))
    plt.plot(x_line, y_line, linestyle="--", linewidth=2, label="track centerline")
    plt.plot(ann_hist["x"], ann_hist["y"], linewidth=2, label="ANN continuous")
    plt.plot(snn_hist["x"], snn_hist["y"], linewidth=2, label="SNN continuous")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Continuous Steering Trajectory ({suffix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"continuous_trajectory_{suffix}.png", dpi=180)
    plt.close()



def plot_steering(ann_hist, snn_hist, suffix):
    plt.figure(figsize=(8, 4))
    plt.plot(ann_hist["steering"], linewidth=2, label="ANN steering")
    plt.plot(snn_hist["steering"], linewidth=2, label="SNN steering")
    plt.xlabel("Step")
    plt.ylabel("Steering")
    plt.title(f"Continuous Steering Signal ({suffix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"continuous_steering_signal_{suffix}.png", dpi=180)
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

    sensor_lines = [ax.plot([], [], linewidth=1.3, alpha=0.9)[0] for _ in range(len(hist["sensor_starts"][0]))]
    hit_points = [ax.plot([], [], marker="o", markersize=3)[0] for _ in range(len(hist["sensor_hits"][0]))]

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", alpha=0.88))
    ax.legend(loc="lower right")

    def update(frame):
        x = hist["x"][frame]
        y = hist["y"][frame]
        theta = hist["theta"][frame]

        path_plot.set_data(hist["x"][:frame+1], hist["y"][:frame+1])
        robot_patch.center = (x, y)
        heading_plot.set_data([x, x + 0.45 * np.cos(theta)], [y, y + 0.45 * np.sin(theta)])

        sensor_vals = hist["sensors"][frame]
        starts = hist["sensor_starts"][frame]
        ends = hist["sensor_ends"][frame]
        hits = hist["sensor_hits"][frame]

        for i, (line, start, end, hit) in enumerate(zip(sensor_lines, starts, ends, hits)):
            line.set_data([start[0], end[0]], [start[1], end[1]])
            alpha = 0.2 + 0.8 * float(sensor_vals[i])
            line.set_alpha(alpha)
            hit_points[i].set_data([hit[0]], [hit[1]])
            hit_points[i].set_alpha(alpha)

        txt.set_text(
            f"step: {frame}\n"
            f"steering: {hist['steering'][frame]:.3f}\n"
            f"error: {hist['lateral_error'][frame]:.3f}\n"
            f"activity: {hist['activity'][frame]:.2f}"
        )
        artists = [path_plot, heading_plot, robot_patch, txt]
        artists.extend(sensor_lines)
        artists.extend(hit_points)
        return tuple(artists)

    anim = FuncAnimation(fig, update, frames=len(hist["x"]), interval=55, repeat=False)
    anim.save(OUT_DIR / filename, writer="pillow", fps=18)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    train_track_seeds = list(range(100, 100 + CFG.n_train_tracks))
    test_track_seeds = list(range(1000, 1000 + CFG.n_test_tracks))

    print("Generating continuous-steering dataset...")
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

    plot_training(ann_hist, snn_hist)

    demo_track = ProceduralTrack(seed=3030, y_max=DATA_CFG.track_y_max)
    ann_demo_hist, ann_demo_metrics = run_episode(demo_track, ann_policy, noise_std=0.0)
    snn_demo_hist, snn_demo_metrics = run_episode(demo_track, snn_policy, noise_std=0.0)

    plot_trajectory(demo_track, ann_demo_hist, snn_demo_hist, "clean")
    plot_steering(ann_demo_hist, snn_demo_hist, "clean")

    ann_noise_hist, ann_noise_metrics = run_episode(demo_track, ann_policy, noise_std=0.08)
    snn_noise_hist, snn_noise_metrics = run_episode(demo_track, snn_policy, noise_std=0.08)

    plot_trajectory(demo_track, ann_noise_hist, snn_noise_hist, "noise")
    plot_steering(ann_noise_hist, snn_noise_hist, "noise")

    make_animation(demo_track, snn_demo_hist, "continuous_snn_clean.gif", "SNN Continuous Steering - Clean")
    make_animation(demo_track, snn_noise_hist, "continuous_snn_noise.gif", "SNN Continuous Steering - Noise")

    torch.save(ann_model.state_dict(), OUT_DIR / "ann_continuous.pt")
    torch.save(snn_model.state_dict(), OUT_DIR / "snn_continuous.pt")

    print("\nContinuous steering metrics:")
    print(f"ANN clean: {ann_demo_metrics}")
    print(f"SNN clean: {snn_demo_metrics}")
    print(f"ANN noise: {ann_noise_metrics}")
    print(f"SNN noise: {snn_noise_metrics}")

    print("\nSaved files:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f" - {p.name}")


if __name__ == "__main__":
    main()
