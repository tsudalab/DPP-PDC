"""
DPP-PDC: Determinantal Point Process for Phase Diagram Construction
====================================================================

A Bayesian active learning framework for efficient phase diagram construction
using Determinantal Point Processes (DPPs) for batch diversity control.

Basic Usage
-----------
Command line:
    $ dpp-pdc run --data datasets/Cu-Mg_Zn_850K.csv --algorithm DPP-PDC
    $ dpp-pdc run --config configs/default.toml

Python API:
    >>> from dpp_pdc import ExperimentConfig
    >>> from dpp_pdc.run_experiments import run_single_temperature_experiment
    >>> 
    >>> config = ExperimentConfig()
    >>> config.set_algorithms(["RS", "PDC", "DPP-PDC"])
    >>> result = run_single_temperature_experiment("850K", ...)

For more information, see the documentation at:
https://github.com/yourusername/dpp-pdc
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from dpp_pdc.config import ExperimentConfig
from dpp_pdc.data_preprocessing import standardize_data, prepare_initial_samples
from dpp_pdc.uncertainty_strategies import (
    compute_pdc_scores,
    compute_ts_scores,
    compute_random_scores,
)
from dpp_pdc.active_learning import (
    active_learning_with_pymc,
    active_learning_traditional,
)

__all__ = [
    "ExperimentConfig",
    "standardize_data",
    "prepare_initial_samples",
    "compute_pdc_scores",
    "compute_ts_scores",
    "compute_random_scores",
    "active_learning_with_pymc",
    "active_learning_traditional",
]
