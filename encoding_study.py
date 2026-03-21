import csv
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import DATA_CFG, TRAIN_CFG, SimConfig
from dataset import generate_supervised_dataset
from models import SNNController
from train import train_snn
from evaluate import evaluate_policy


# ============================================================
# Encoding Study for SNN Path-Following
# ------------------------------------------------------------
# Run from inside neuromorphic_robot/
#   source .venv/bin/activate
#   python encoding_study.py
#
# What it does:
# - compares three spike encodings:
#     1. rate coding
#     2. latency coding
#     3. population coding
# - trains one SNN per encoding
# - evaluates on multiple simulator corruption settings
# - saves CSV summaries and plots in outputs/encoding_study/
# ============================================================

SEED = 33
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path("outputs/encoding_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
# CSV helpers
# ------------------------------------------------------------
def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Encoding functions
# ------------------------------------------------------------
def rate_encode(x, num_steps):
    x_rep = x.unsqueeze(0).repeat(num_steps, 1, 1)
    return torch.bernoulli(x_rep)



def latency_encode(x, num_steps):
    """
    Earlier spike means stronger activation.
    One spike per feature channel, unless x is near zero.
    """
    x = torch.clamp(x, 0.0, 1.0)
    batch_size, features = x.shape
    spikes = torch.zeros((num_steps, batch_size, features), dtype=torch.float32, device=x.device)

    # strong input -> early spike, weak input -> late spike
    spike_times = ((1.0 - x) * (num_steps - 1)).round().long()
    active_mask = x > 0.05

    for b in range(batch_size):
        for f in range(features):
            if active_mask[b, f]:
                t = spike_times[b, f].item()
                spikes[t, b, f] = 1.0
    return spikes



def population_encode(x, num_steps, num_centers=3, sigma=0.18):
    """
    Expands each scalar feature into a small population of neurons.
    Each population neuron has a preferred center in [0,1].
    Population response is then rate-coded over time.
    """
    x = torch.clamp(x, 0.0, 1.0)
    centers = torch.linspace(0.0, 1.0, num_centers, device=x.device)
    batch_size, features = x.shape

    expanded = []
    for c in centers:
        response = torch.exp(-((x - c) ** 2) / (2 * sigma ** 2))
        expanded.append(response)
    pop = torch.cat(expanded, dim=1)  # [B, F * num_centers]

    pop_rep = pop.unsqueeze(0).repeat(num_steps, 1, 1)
    return torch.bernoulli(pop_rep)


# ------------------------------------------------------------
# Policy wrappers for custom encodings
# ------------------------------------------------------------
class EncodedSNNPolicy:
    def __init__(self, model, device, num_steps, encoding_name, pop_centers=3):
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.encoding_name = encoding_name
        self.pop_centers = pop_centers

    def encode(self, x):
        if self.encoding_name == "rate":
            return rate_encode(x, self.num_steps)
        if self.encoding_name == "latency":
            return latency_encode(x, self.num_steps)
        if self.encoding_name == "population":
            return population_encode(x, self.num_steps, num_centers=self.pop_centers)
        raise ValueError(f"Unknown encoding: {self.encoding_name}")

    @torch.no_grad()
    def act(self, sensor_values):
        self.model.eval()
        x_np = np.asarray(sensor_values, dtype=np.float32)[None, :]
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        spk_in = self.encode(x).to(self.device)
        spk_out, _ = self.model(spk_in)
        logits = spk_out.sum(dim=0)
        action = int(logits.argmax(dim=1).item())
        scores = logits.squeeze(0).detach().cpu().numpy()
        spike_count = float(spk_out.sum().item())
        return action, scores, spike_count


# ------------------------------------------------------------
# Custom train loop for arbitrary encoding function
# ------------------------------------------------------------
def train_snn_with_encoder(model, train_loader, test_loader, epochs, lr, num_steps, encoding_name, pop_centers=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = {"train_loss": [], "test_loss": [], "test_acc": []}

    def encode(x):
        if encoding_name == "rate":
            return rate_encode(x, num_steps)
        if encoding_name == "latency":
            return latency_encode(x, num_steps)
        if encoding_name == "population":
            return population_encode(x, num_steps, num_centers=pop_centers)
        raise ValueError(f"Unknown encoding: {encoding_name}")

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total = 0
        correct = 0
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            spk_in = encode(xb).to(device)
            spk_out, _ = model(spk_in)
            logits = spk_out.sum(dim=0)
            loss = loss_fn(logits, yb)
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            total_loss += loss.item() * yb.size(0)
        return total_loss / total, correct / total

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            spk_in = encode(xb).to(device)
            spk_out, _ = model(spk_in)
            logits = spk_out.sum(dim=0)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            seen += yb.size(0)

        train_loss = running_loss / seen
        test_loss, test_acc = evaluate(test_loader)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(
            f"[{encoding_name.upper()}] Epoch {epoch+1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc*100:.2f}%"
        )

    return history


# ------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------
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
def plot_accuracy_curves(histories, filename):
    plt.figure(figsize=(8, 4.5))
    for name, hist in histories.items():
        plt.plot(hist["test_acc"], linewidth=2, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("Encoding Study: SNN Test Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()



def plot_success_vs_noise(summary_rows, filename):
    plt.figure(figsize=(8, 4.5))
    encodings = sorted(set(r["encoding"] for r in summary_rows))
    for enc in encodings:
        subset = [r for r in summary_rows if r["encoding"] == enc and r["delay_steps"] == 0 and r["dropout_prob"] == 0.0 and r["dead_sensor_index"] == -1]
        subset = sorted(subset, key=lambda x: x["noise_std"])
        xs = [r["noise_std"] for r in subset]
        ys = [r["success_rate"] for r in subset]
        plt.plot(xs, ys, marker="o", linewidth=2, label=enc)
    plt.xlabel("Noise std")
    plt.ylabel("Success rate")
    plt.title("Encoding Study: Success vs Noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()



def plot_error_vs_noise(summary_rows, filename):
    plt.figure(figsize=(8, 4.5))
    encodings = sorted(set(r["encoding"] for r in summary_rows))
    for enc in encodings:
        subset = [r for r in summary_rows if r["encoding"] == enc and r["delay_steps"] == 0 and r["dropout_prob"] == 0.0 and r["dead_sensor_index"] == -1]
        subset = sorted(subset, key=lambda x: x["noise_std"])
        xs = [r["noise_std"] for r in subset]
        ys = [r["mean_lateral_error"] for r in subset]
        plt.plot(xs, ys, marker="o", linewidth=2, label=enc)
    plt.xlabel("Noise std")
    plt.ylabel("Mean lateral error")
    plt.title("Encoding Study: Tracking Error vs Noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()



def plot_activity_vs_noise(summary_rows, filename):
    plt.figure(figsize=(8, 4.5))
    encodings = sorted(set(r["encoding"] for r in summary_rows))
    for enc in encodings:
        subset = [r for r in summary_rows if r["encoding"] == enc and r["delay_steps"] == 0 and r["dropout_prob"] == 0.0 and r["dead_sensor_index"] == -1]
        subset = sorted(subset, key=lambda x: x["noise_std"])
        xs = [r["noise_std"] for r in subset]
        ys = [r["mean_activity"] for r in subset]
        plt.plot(xs, ys, marker="o", linewidth=2, label=enc)
    plt.xlabel("Noise std")
    plt.ylabel("Mean spike activity")
    plt.title("Encoding Study: Activity vs Noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    train_track_seeds = list(range(100, 100 + DATA_CFG.n_train_tracks))
    test_track_seeds = list(range(1000, 1000 + DATA_CFG.n_test_tracks))

    print("Generating datasets...")
    x_train, y_train = generate_supervised_dataset(train_track_seeds, DATA_CFG.samples_per_track)
    x_test, y_test = generate_supervised_dataset(test_track_seeds, DATA_CFG.samples_per_track // 2)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=TRAIN_CFG.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=TRAIN_CFG.batch_size, shuffle=False)

    encoding_specs = [
        {"name": "rate", "input_dim": DATA_CFG.num_sensors, "pop_centers": 1},
        {"name": "latency", "input_dim": DATA_CFG.num_sensors, "pop_centers": 1},
        {"name": "population", "input_dim": DATA_CFG.num_sensors * 3, "pop_centers": 3},
    ]

    histories = {}
    policies = {}
    raw_rows = []

    conditions = [
        ("noise_0.00", SimConfig(noise_std=0.00)),
        ("noise_0.04", SimConfig(noise_std=0.04)),
        ("noise_0.08", SimConfig(noise_std=0.08)),
        ("noise_0.12", SimConfig(noise_std=0.12)),
        ("delay_2", SimConfig(delay_steps=2)),
        ("dropout_0.20", SimConfig(sensor_dropout_prob=0.20)),
        ("dead_center", SimConfig(dead_sensor_index=DATA_CFG.num_sensors // 2)),
    ]

    for spec in encoding_specs:
        print(f"\n=== Training encoding: {spec['name']} ===")
        model = SNNController(
            input_dim=spec["input_dim"],
            hidden_dim=TRAIN_CFG.hidden_dim,
            beta=TRAIN_CFG.beta,
        ).to(device)

        history = train_snn_with_encoder(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=TRAIN_CFG.snn_epochs,
            lr=TRAIN_CFG.lr,
            num_steps=TRAIN_CFG.snn_steps,
            encoding_name=spec["name"],
            pop_centers=spec["pop_centers"],
        )
        histories[spec["name"]] = history

        policy = EncodedSNNPolicy(
            model=model,
            device=device,
            num_steps=TRAIN_CFG.snn_steps,
            encoding_name=spec["name"],
            pop_centers=spec["pop_centers"],
        )
        policies[spec["name"]] = policy

        torch.save(model.state_dict(), OUT_DIR / f"snn_{spec['name']}.pt")

        for cond_name, sim_cfg in conditions:
            print(f"Evaluating {spec['name']} | {cond_name}")
            rows = evaluate_policy(spec["name"], policy, test_track_seeds, sim_cfg, DATA_CFG)
            for r in rows:
                r["encoding"] = spec["name"]
                r["condition"] = cond_name
                r["snn_steps"] = TRAIN_CFG.snn_steps
                r["hidden_dim"] = TRAIN_CFG.hidden_dim
            raw_rows.extend(rows)

    summary_rows = summarize_rows(
        raw_rows,
        ["encoding", "condition", "noise_std", "delay_steps", "dropout_prob", "dead_sensor_index", "snn_steps", "hidden_dim"],
    )

    save_csv(raw_rows, OUT_DIR / "encoding_raw_metrics.csv")
    save_csv(summary_rows, OUT_DIR / "encoding_summary.csv")

    plot_accuracy_curves(histories, OUT_DIR / "encoding_test_accuracy.png")
    plot_success_vs_noise(summary_rows, OUT_DIR / "encoding_success_vs_noise.png")
    plot_error_vs_noise(summary_rows, OUT_DIR / "encoding_error_vs_noise.png")
    plot_activity_vs_noise(summary_rows, OUT_DIR / "encoding_activity_vs_noise.png")

    best_by_success = sorted(
        summary_rows,
        key=lambda r: (r["success_rate"], -r["mean_lateral_error"]),
        reverse=True,
    )[0]

    print("\nSaved files:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f" - {p.name}")

    print("\nBest encoding summary:")
    print(best_by_success)


if __name__ == "__main__":
    main()
