import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    plt.savefig(OUT_DIR / "training_accuracy_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(ann_hist["train_loss"], label="ANN train loss")
    plt.plot(snn_hist["train_loss"], label="SNN train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ANN vs SNN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_loss_comparison.png", dpi=160)
    plt.close()


def plot_episode(track, ann_hist, snn_hist, title_suffix="clean"):
    y_line = np.linspace(0.0, track.y_max, 800)
    x_line = track.line_x(y_line)

    plt.figure(figsize=(7, 10))
    plt.plot(x_line, y_line, linestyle="--", label="track centerline")
    plt.plot(ann_hist["x"], ann_hist["y"], label="ANN path")
    plt.plot(snn_hist["x"], snn_hist["y"], label="SNN path")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Trajectory Comparison ({title_suffix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"trajectory_comparison_{title_suffix}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(ann_hist["lateral_error"], label="ANN lateral error")
    plt.plot(snn_hist["lateral_error"], label="SNN lateral error")
    plt.xlabel("Step")
    plt.ylabel("|x - x_track|")
    plt.title(f"Lateral Error ({title_suffix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"lateral_error_{title_suffix}.png", dpi=160)
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
    plt.savefig(OUT_DIR / "success_rate_summary.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 4.5))
    for i, p in enumerate(policies):
        plt.bar(x + (i - 0.5) * width, error_vals[p], width=width, label=p)
    plt.xticks(x, cond_labels, rotation=20, ha="right")
    plt.ylabel("Mean lateral error")
    plt.title("Mean Lateral Error Across Conditions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "lateral_error_summary.png", dpi=160)
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
    ax.plot(x_line, y_line, linestyle="--", label="track centerline")
    path_plot, = ax.plot([], [], label="robot path")
    robot_dot, = ax.plot([], [], marker="o", markersize=8)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", alpha=0.85))
    ax.legend(loc="lower right")

    def update(frame):
        path_plot.set_data(hist["x"][:frame+1], hist["y"][:frame+1])
        robot_dot.set_data([hist["x"][frame]], [hist["y"][frame]])
        sensors = hist["sensors"][frame]
        act = ACTION_TO_NAME[hist["actions"][frame]]
        txt.set_text(
            f"step: {frame}\n"
            f"action: {act}\n"
            f"err: {hist['lateral_error'][frame]:.3f}\n"
            f"activity: {hist['activity'][frame]:.2f}\n"
            f"center sensor: {sensors[len(sensors)//2]:.2f}"
        )
        return path_plot, robot_dot, txt

    anim = FuncAnimation(fig, update, frames=len(hist["x"]), interval=55, repeat=False)
    anim.save(OUT_DIR / filename, writer="pillow", fps=18)
    plt.close(fig)
