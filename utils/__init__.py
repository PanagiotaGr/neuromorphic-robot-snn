"""Utility modules for neuromorphic robot SNN project"""
from .checkpoint import save_checkpoint, load_checkpoint, CheckpointManager, get_best_checkpoint
from .early_stopping import EarlyStopping
from .scheduler import get_scheduler
from .energy import (
    estimate_snn_energy, estimate_ann_energy, compute_model_stats,
    compare_energy_efficiency, estimate_inference_time
)
from .statistics import (
    compute_all_statistics, print_statistics_summary,
    paired_ttest, wilcoxon_test, mcnemar_test, bootstrap_confidence_interval
)

__all__ = [
    'save_checkpoint', 'load_checkpoint', 'CheckpointManager', 'get_best_checkpoint',
    'EarlyStopping', 'get_scheduler',
    'estimate_snn_energy', 'estimate_ann_energy', 'compute_model_stats',
    'compare_energy_efficiency', 'estimate_inference_time',
    'compute_all_statistics', 'print_statistics_summary',
    'paired_ttest', 'wilcoxon_test', 'mcnemar_test', 'bootstrap_confidence_interval'
]
