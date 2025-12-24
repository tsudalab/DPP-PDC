# DPP-PDC: Determinantal Point Process for Phase Diagram Construction

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Bayesian active learning framework for efficient phase diagram construction using Determinantal Point Processes (DPPs) for batch diversity control.

## Abstract

Phase diagrams are crucial for understanding material systems and developing novel materials. However, constructing phase diagrams often requires extensive experimental data, consuming significant time and resources. This work proposes **DPP-PDC**, which introduces the repulsive property of Determinantal Point Processes into traditional phase diagram construction methods, naturally increasing the diversity of sampled sets while maintaining uncertainty-aware sampling.

**Key Features:**
- Bayesian inference of diversity control parameter via MCMC
- Automatic exploration-exploitation balance
- Significant improvement in complex phase spaces
- Comparable or better performance in simple phase spaces

## Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/dpp-pdc.git
cd dpp-pdc

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package with dependencies
pip install -e .
```

### Dependencies

The package requires Python 3.12+ and the following main dependencies:
- PyMC >= 5.20.1 (for Bayesian inference)
- scikit-learn >= 1.6.1
- numpy, pandas, matplotlib, scipy

All dependencies are automatically installed with `pip install -e .`

## Quick Start

### Option 1: Command Line Interface (Recommended)

```bash
# Run a simple experiment
dpp-pdc run --data datasets/Cu-Mg_Zn_850K.csv --algorithm DPP-PDC --batch-size 10

# Run with multiple algorithms for comparison
dpp-pdc run --temperatures 850K --algorithms RS PDC KMedoids-PDC DPP-PDC --batch-sizes 10 12

# Run with a configuration file
dpp-pdc run --config configs/default.toml

# Generate a configuration file template
dpp-pdc init-config --output my_experiment.toml

# Visualize sampling evolution
dpp-pdc visualize --data datasets/Cu-Mg_Zn_850K.csv --algorithm DPP-PDC
```

### Option 2: Configuration File

Create a TOML configuration file for reproducible experiments:

```toml
# experiment.toml
[experiment]
temperatures = ["650K", "850K", "1050K"]
algorithms = ["RS", "PDC", "KMedoids-PDC", "DPP-PDC"]
batch_sizes = [5, 8, 10, 12]
n_runs = 10

[data]
file_path_template = "datasets/Cu-Mg_Zn_{temperature}.csv"

[output]
output_dir = "results"
```

Then run:
```bash
dpp-pdc run --config experiment.toml
```

### Option 3: Python API

```python
from dpp_pdc import ExperimentConfig
from dpp_pdc.run_experiments import run_single_temperature_experiment

# Create and customize configuration
config = ExperimentConfig()
config.set_temperatures(["850K"])
config.set_algorithms(["RS", "PDC", "KMedoids-PDC", "DPP-PDC"])
config.set_batch_sizes([10, 12])

# Run experiment
result = run_single_temperature_experiment(
    temperature="850K",
    file_path_template="datasets/Cu-Mg_Zn_{temperature}.csv",
    algorithms=config.algorithms,
    batch_sizes=config.batch_sizes,
    n_runs=10
)
```

## CLI Reference

```
dpp-pdc --help

Commands:
  run          Run active learning experiments
  visualize    Visualize sampling distribution evolution
  init-config  Generate a default configuration file

Run Options:
  -c, --config        Path to TOML configuration file
  -d, --data          Path to dataset CSV file
  -t, --temperatures  Temperature conditions (e.g., 500K 850K)
  -a, --algorithms    Algorithms to evaluate (RS, PDC, KMedoids-PDC, DPP-PDC)
  -b, --batch-sizes   Batch sizes for sampling
  -n, --n-runs        Number of independent runs
  -o, --output-dir    Directory to save results
```

## Dataset Format

Input CSV files should have the following format:

```csv
phase_name,Zn,Mg,Cu
FCC,0.1,0.2,0.7
BCC,0.3,0.3,0.4
...
```

- First column: phase label
- Remaining columns: composition features

Example datasets for the Cu-Mg-Zn ternary system are provided in `datasets/`.

## Output

Results are saved to the specified output directory:

```
results/
├── phase_discovery_batch10_850K.png    # Phase discovery curves
├── combined_grid_850K.png              # Combined grid plot (phase + sigma)
├── auc_bar_850K.png                    # AUC comparison bar chart
├── auc_table_850K.csv                  # AUC values table
├── DPP-PDC_batch10_850K_sigma_evolution.png  # σ² evolution
├── RS_batch10_results.csv              # Raw results
├── PDC_batch10_results.csv
├── KMedoids-PDC_batch10_results.csv
└── DPP-PDC_batch10_results.csv
```

## Project Structure

```
dpp-pdc/
├── pyproject.toml          # Package configuration
├── README.md
├── LICENSE
├── configs/
│   └── default.toml        # Default configuration
├── datasets/               # Example datasets
├── dpp_pdc/
│   ├── __init__.py
│   ├── cli.py              # Command line interface
│   ├── config.py           # Configuration management
│   ├── active_learning.py  # Main algorithms
│   ├── data_preprocessing.py
│   ├── uncertainty_strategies.py
│   ├── pymc_dpp.py         # Bayesian DPP sampling
│   ├── visualization.py
│   └── run_experiments.py
└── examples/
    └── quickstart.ipynb    # Jupyter notebook tutorial
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2024dpppdc,
  title={DPP-PDC: Bayesian Diversity Control for Active Learning in Phase Diagram Construction},
  author={Your Name},
  journal={Digital Discovery},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon the phase diagram construction method proposed by Tamura et al. (2022).
