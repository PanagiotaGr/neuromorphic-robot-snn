import csv
from pathlib import Path

from simulator import ProceduralTrack, run_episode


OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def evaluate_policy(policy_name, policy, track_seeds, sim_cfg, data_cfg):
    rows = []
    for seed in track_seeds:
        track = ProceduralTrack(seed=seed, y_max=data_cfg.track_y_max)
        _, _, metrics = run_episode(track, policy, sim_cfg, record=True)
        rows.append({
            "policy": policy_name,
            "track_seed": seed,
            "noise_std": sim_cfg.noise_std,
            "delay_steps": sim_cfg.delay_steps,
            "dropout_prob": sim_cfg.sensor_dropout_prob,
            "dead_sensor_index": sim_cfg.dead_sensor_index,
            **metrics,
        })
    return rows


def save_eval_csv(rows, filename="evaluation_metrics.csv"):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(OUT_DIR / filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
