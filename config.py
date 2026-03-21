from dataclasses import dataclass


@dataclass
class DataConfig:
    num_sensors: int = 9
    n_train_tracks: int = 30
    n_test_tracks: int = 12
    samples_per_track: int = 450
    lookahead: float = 0.85
    sensor_arc_deg: float = 90.0
    track_y_max: float = 40.0
    line_sigma: float = 0.28


@dataclass
class TrainConfig:
    batch_size: int = 96
    ann_epochs: int = 18
    snn_epochs: int = 18
    lr: float = 1e-3
    snn_steps: int = 25
    hidden_dim: int = 96
    beta: float = 0.92


@dataclass
class SimConfig:
    speed: float = 0.16
    turn_rate: float = 0.11
    max_steps: int = 340
    failure_distance: float = 1.55
    delay_steps: int = 0
    noise_std: float = 0.0
    sensor_dropout_prob: float = 0.0
    dead_sensor_index: int = -1


DATA_CFG = DataConfig()
TRAIN_CFG = TrainConfig()
