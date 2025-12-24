# DPP-PDC: Bayesian Diversity Control for Phase Diagram Construction

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Bayesian active learning framework for efficient phase diagram determination using uncertainty-weighted Determinantal Point Processes (UwDPP).

## Overview

Machine learning methods are increasingly used in experimental design in phase diagram determination. Some methods perform batch design, where multiple points are sampled from the design space. In this case, it is essential to diversify samples to avoid performing almost identical experiments, and control the diversity level appropriately. Manual diversity control is unintuitive and may require additional trial-and-error in prior to the experiments are started. We propose a Bayesian model called determinantal point process for phase diagram construction (DPP-PDC) that can perform batch design and automatic diversity control simultaneously. Central to this model is the uncertainty-weighted determinantal point process that samples a set of points with high uncertainty under diversity control. Experiments with Cu-Mg-Zn ternary system demonstrate that DPP-PDC can actively control the sample diversity to achieve high efficiency.

## Installation

### Prerequisites

- Python 3.12 or higher
- We recommend using a virtual environment

### Option 1: Using conda (Recommended)

```bash
# Create a new environment
conda create -n dpp-pdc python=3.12 -y

# Activate the environment
conda activate dpp-pdc

# Clone the repository
git clone https://github.com/tsudalab/DPP-PDC.git
cd DPP-PDC

# Install the package
pip install -e .
```

### Option 2: Using venv

```bash
# Clone the repository
git clone https://github.com/tsudalab/DPP-PDC.git
cd DPP-PDC

# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate
# Activate (Windows)
venv\Scripts\activate

# Install the package
pip install -e .
```

### Verify Installation

```bash
# Check CLI is available
dpp-pdc --help

# Or test import
python -c 'import dpp_pdc; print("Installation successful!")'
```

## Quick Start

### 1. Command Line Interface (Recommended)

Run a simple experiment:

```bash
dpp-pdc run --temperatures 500K --algorithm DPP-PDC --batch-size 10
```

Run the full benchmark from the paper:

```bash
dpp-pdc run --data datasets/Cu-Mg_Zn_all.csv  --algorithms RS PDC K-Medoids DPP-PDC  --batch-sizes 5 8 10 12  --n-runs 10  --max-sampling 300
```

### 2. Using Configuration Files

Generate a configuration template:

```bash
dpp-pdc init-config --output configs/my_config.toml
```

Edit the configuration file, then run:

```bash
dpp-pdc run --config configs/my_config.toml
```

### 3. Python API

```python
from dpp_pdc import standardize_data
from dpp_pdc.run_experiments import run_single_temperature_experiment

# Load and preprocess data
scaler, X_std, y_encoded = standardize_data("datasets/Cu-Mg_Zn_500K.csv")

# Run experiment
results = run_single_temperature_experiment(
    temperature="500K",
    file_path_template="datasets/Cu-Mg_Zn_{temperature}.csv",
    algorithms=["RS", "PDC", "DPP-PDC"],
    batch_sizes=[10, 12],
    n_runs=5,
    max_sampling=300
)
```

## CLI Reference

### `dpp-pdc run`

Run active learning experiments.

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--config` | `-c` | Path to TOML configuration file | - |
| `--data` | `-d` | Path to dataset CSV file | - |
| `--algorithms` | `-a` | Algorithms to compare | `RS PDC K-Medoids DPP-PDC` |
| `--batch-sizes` | `-b` | Batch sizes to test | `10` |
| `--n-runs` | `-n` | Number of independent runs | `10` |
| `--n-initial` | - | Initial labeled samples | `10` |
| `--max-sampling` | - | Maximum total samples | auto |
| `--output-dir` | `-o` | Output directory | `results` |
| `--verbose` | `-v` | Enable verbose output | `False` |

### `dpp-pdc init-config`

Generate a configuration file template.
```bash
dpp-pdc init-config --output configs/my_config.toml
```

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| **RS** | Random Sampling (baseline) |
| **PDC** | Phase Diagram Construction using uncertainty sampling |
| **K-Medoids** | Two-step approach: select top 30% uncertain points, then apply K-Medoids clustering |
| **DPP-PDC** | Our method: uncertainty-weighted DPP with Bayesian diversity control |

## Dataset Format

Input CSV files should have the following format:

### Single Temperature Dataset
```csv
phase_name,Zn,Mg,Cu
FCC,0.1,0.2,0.7
BCC,0.3,0.3,0.4
HCP+LIQUID,0.5,0.4,0.1
...
```

### Complete Dataset (with Temperature)
```csv
phase_name,Zn,Mg,Cu,T
FCC,0.1,0.2,0.7,850
BCC,0.3,0.3,0.4,850
HCP+LIQUID,0.5,0.4,0.1,650
...
```

**Format requirements:**
- First column: phase label (categorical)
- Composition columns (e.g., Zn, Mg, Cu) should sum to 1.0
- Temperature column `T` is required for the complete dataset
- The complete dataset used in the paper contains 71 distinct phase labels across temperatures from 500K to 1500K

## Output

Results are saved to the `results_pymc_{temperature}/` directory.

### Complete Dataset (`Cu-Mg-Zn_complete.csv`)
```
results_pymc_complete/
├── combined_grid_complete.png/pdf       # Phase discovery curves + σ² evolution (all batch sizes)
├── auc_bar_complete.png/pdf             # AUC bar chart comparison
├── auc_table_complete.csv               # AUC values for all algorithms and batch sizes
├── metrics_curves_complete.png/pdf      # Classification metrics (Accuracy, Precision, Recall, F1, mAP)
├── phase_discovery_batch5_complete.png  # Phase discovery curve for batch size 5
├── phase_discovery_batch8_complete.png
├── phase_discovery_batch10_complete.png
├── phase_discovery_batch12_complete.png
├── RS_batch5_results.csv                # Phase counts per iteration
├── RS_batch8_results.csv
├── ...
├── DPP-PDC_batch5_results.csv
├── DPP-PDC_batch5_sigma.csv             # σ² values per iteration
├── DPP-PDC_batch8_sigma.csv
├── ...
├── RS_batch5_metrics.csv                # Classification metrics per iteration
├── PDC_batch5_metrics.csv
└── DPP-PDC_batch5_metrics.csv
```

### Single Temperature Dataset (e.g., `Cu-Mg_Zn_850K.csv`)
```
results_pymc_850K/
├── combined_grid_850K.png/pdf           # Phase discovery curves + σ² evolution
├── auc_bar_850K.png/pdf                 # AUC bar chart comparison
├── auc_table_850K.csv                   # AUC values
├── metrics_curves_850K.png/pdf          # Classification metrics curves
├── phase_discovery_batch10_850K.png     # Phase discovery curve
├── ternary_DPP-PDC_batch10_850K.png     # Ternary diagram with sampling points
├── RS_batch10_results.csv
├── PDC_batch10_results.csv
├── DPP-PDC_batch10_results.csv
├── DPP-PDC_batch10_sigma.csv
├── RS_batch10_metrics.csv
├── PDC_batch10_metrics.csv
└── DPP-PDC_batch10_metrics.csv
```

**Output files description:**

| File | Description |
|------|-------------|
| `combined_grid_*.png/pdf` | Combined plot: phase discovery curves (top) and σ² evolution (bottom) |
| `auc_bar_*.png/pdf` | Bar chart comparing AUC across algorithms and batch sizes |
| `auc_table_*.csv` | Area under the phase discovery curve (numerical values) |
| `metrics_curves_*.png/pdf` | Classification metrics over iterations (Accuracy, Precision, Recall, F1, mAP) |
| `phase_discovery_batch*_*.png` | Phase discovery curve for specific batch size |
| `ternary_*_*.png` | Ternary composition diagram with sampling evolution (single temperature only) |
| `{algorithm}_batch{size}_results.csv` | Number of discovered phases per iteration |
| `{algorithm}_batch{size}_sigma.csv` | σ² values per iteration (DPP methods only) |
| `{algorithm}_batch{size}_metrics.csv` | Classification metrics per iteration |

## Configuration File Format

Example `experiment.toml`:
```toml
[experiment]
# Temperature conditions to evaluate (ignored if using --data with complete dataset)
temperatures = ["500K", "650K", "1000K", "1050K"]

# Algorithms to compare
algorithms = ["RS", "PDC", "K-Medoids", "DPP-PDC"]

# Batch sizes for each sampling iteration
batch_sizes = [5, 8, 10, 12]

# Number of independent runs per configuration
n_runs = 10

# Number of initial random samples
n_initial_sampling = 10

# Maximum total sampling points (comment out for auto-calculation)
# max_sampling = 300

[data]
# Template for dataset file paths (use {temperature} as placeholder)
file_path_template = "datasets/Cu-Mg_Zn_{temperature}.csv"

[output]
output_dir = "results"
save_plots = true
verbose = false

[bayesian]
# Prior parameters for noise variance (log-normal distribution)
prior_mu = -4.0
prior_sigma = 4.0

# MCMC sampling parameters
mcmc_samples = 1000
mcmc_tune = 1000
mcmc_chains = 1
target_accept = 0.9
```

## Reproducing Paper Results

To reproduce the results from the paper:

```bash
# Full experiment
dpp-pdc run --data datasets/Cu-Mg_Zn_all.csv  --algorithms RS PDC K-Medoids DPP-PDC  --batch-sizes 5 8 10 12  --n-runs 10  --max-sampling 300  --output-dir paper_results
```

## Project Structure
```
DPP-PDC/
├── pyproject.toml              # Package configuration
├── README.md                   # This file
├── configs/
│   └── experiment.toml         # Example configuration
├── datasets/
│   ├── Cu-Mg_Zn_500K.csv       # Single temperature slices
│   ├── Cu-Mg_Zn_550K.csv       # (500K to 1500K, 50K intervals)
│   ├── ...
│   ├── Cu-Mg_Zn_1500K.csv
│   └── Cu-Mg_Zn_all.csv        # Complete dataset with all temperatures
├── dpp_pdc/
│   ├── __init__.py
│   ├── cli.py                  # Command-line interface
│   ├── config.py               # Configuration management
│   ├── active_learning.py      # Core active learning loop
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── uncertainty_strategies.py  # RS, PDC, K-Medoids sampling
│   ├── pymc_dpp.py             # Bayesian DPP sampling with PyMC
│   ├── visualization.py        # Plotting functions
│   ├── sampling_distribution.py   # Sampling visualization demo
│   └── run_experiments.py      # Experiment runner
└── examples/
    └── quickstart.ipynb        # Jupyter notebook tutorial
```

## Citation

If you use this code in your research, please cite:

```bibtex

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments


