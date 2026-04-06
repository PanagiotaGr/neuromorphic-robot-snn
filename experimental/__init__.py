"""Experimental novel features test suite"""
from .novel_experiments import (
    experiment_bio_inspired_navigation,
    experiment_multi_timescale_snn,
    experiment_predictive_coding,
    experiment_stdp_plasticity,
    experiment_integration_test
)
from .comprehensive_experiments import (
    run_all_experiments,
    main as run_comprehensive
)

__all__ = [
    'experiment_bio_inspired_navigation',
    'experiment_multi_timescale_snn',
    'experiment_predictive_coding',
    'experiment_stdp_plasticity',
    'experiment_integration_test',
    'run_all_experiments',
    'run_comprehensive'
]
