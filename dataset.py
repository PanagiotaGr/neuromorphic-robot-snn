import math
import numpy as np
import torch

from config import DATA_CFG
from simulator import ProceduralTrack, sense_track


def teacher_action(track, x, y, theta):
    target_x = float(track.line_x(y + 0.8))
    lateral_error = target_x - x
    target_theta = float(track.tangent_theta(y + 0.8))
    heading_error = math.atan2(math.sin(target_theta - theta), math.cos(target_theta - theta))
    score = 0.90 * lateral_error + 0.55 * heading_error
    if score < -0.12:
        return 2
    if score > 0.12:
        return 0
    return 1


def generate_supervised_dataset(track_seeds, samples_per_track):
    xs, ys = [], []
    for seed in track_seeds:
        track = ProceduralTrack(seed=seed, y_max=DATA_CFG.track_y_max)
        rng = np.random.default_rng(seed + 1000)
        for _ in range(samples_per_track):
            y = rng.uniform(0.2, DATA_CFG.track_y_max - 2.0)
            center_x = float(track.line_x(y))
            tangent = float(track.tangent_theta(y))
            x = center_x + rng.normal(0.0, 0.55)
            theta = tangent + rng.normal(0.0, 0.35)
            sensors = sense_track(track, x=x, y=y, theta=theta)
            action = teacher_action(track, x, y, theta)
            xs.append(sensors)
            ys.append(action)
    x = torch.tensor(np.array(xs), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.long)
    return x, y
