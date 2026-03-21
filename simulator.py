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
        self.path_x = [x]
        self.path_y = [y]

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


def sense_track(track, x, y, theta, sigma=None, lookahead=None):
    if sigma is None:
        sigma = DATA_CFG.line_sigma
    if lookahead is None:
        lookahead = DATA_CFG.lookahead

    vals = []
    for ang in SENSOR_ANGLES:
        phi = theta + ang
        sx = x + lookahead * math.cos(phi)
        sy = y + lookahead * math.sin(phi)
        target_x = float(track.line_x(sy))
        dist = abs(sx - target_x)
        val = math.exp(-(dist ** 2) / (2 * sigma ** 2))
        vals.append(val)
    return np.clip(np.array(vals, dtype=np.float32), 0.0, 1.0)


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
    hist = {"x": [], "y": [], "track_x": [], "actions": [], "scores": [], "sensors": [], "lateral_error": [], "activity": []}
    failure = False

    for step in range(sim_cfg.max_steps):
        true_sensors = sense_track(track, robot.x, robot.y, robot.theta)
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
            hist["track_x"].append(track_x)
            hist["actions"].append(action)
            hist["scores"].append(np.array(scores))
            hist["sensors"].append(used_sensors.copy())
            hist["lateral_error"].append(lateral_error)
            hist["activity"].append(activity)

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
