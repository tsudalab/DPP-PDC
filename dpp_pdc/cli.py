"""
Command Line Interface for DPP-PDC.
Provides easy-to-use commands for running experiments without editing source code.
"""

import argparse
import sys
from pathlib import Path


def create_parser():
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="dpp-pdc",
        description="DPP-PDC: Determinantal Point Process for Phase Diagram Construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single experiment
  dpp-pdc run --temperatures 500K --algorithm DPP-PDC --batch-size 10

  # Run multiple temperature experiments
  dpp-pdc run --temperatures 500K 650K 850K --algorithms RS PDC DPP-PDC

  # Run experiments with a config file
  dpp-pdc run --config configs/experiment.toml  

  # Run full benchmark on complete dataset (reproduces paper results)
  dpp-pdc run --data datasets/Cu-Mg-Zn_complete.csv -a RS PDC KMedoids-PDC DPP-PDC -b 5 8 10 12 -n 10 --max-sampling 300

  # Visualize sampling distribution
  dpp-pdc visualize --data datasets/Cu-Mg_Zn_850K.csv --algorithm DPP-PDC

  # Generate default config file
  dpp-pdc init-config --output configs/my_config.toml
        """,
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ==================== run subcommand ====================
    run_parser = subparsers.add_parser(
        "run", 
        help="Run active learning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file option (highest priority)
    run_parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to TOML configuration file (overrides other arguments)",
    )
    
    # Data options
    data_group = run_parser.add_argument_group("Data Options")
    data_group.add_argument(
        "-d", "--data",
        type=str,
        help="Path to dataset CSV file",
    )
    data_group.add_argument(
        "-t", "--temperatures",
        nargs="+",
        default=["850K"],
        help="Temperature conditions to run (default: 850K)",
    )
    data_group.add_argument(
        "--data-template",
        type=str,
        default="datasets/Cu-Mg_Zn_{temperature}.csv",
        help="Template for dataset file paths (use {temperature} placeholder)",
    )
    
    # Algorithm options
    algo_group = run_parser.add_argument_group("Algorithm Options")
    algo_group.add_argument(
        "-a", "--algorithms",
        nargs="+",
        choices=["RS", "PDC", "K-Medoids", "DPP-PDC"],
        default=["RS", "PDC", "K-Medoids", "DPP-PDC"],
        help="Algorithms to evaluate (default: RS PDC K-Medoids DPP-PDC)",
    )
    algo_group.add_argument(
        "-b", "--batch-sizes",
        nargs="+",
        type=int,
        default=[10],
        help="Batch sizes for sampling (default: 10)",
    )
    algo_group.add_argument(
        "-n", "--n-runs",
        type=int,
        default=10,
        help="Number of runs per configuration (default: 10)",
    )
    algo_group.add_argument(
        "--n-initial",
        type=int,
        default=10,
        help="Number of initial sampling points (default: 10)",
    )
    algo_group.add_argument(
        "--max-sampling",
        type=int,
        default=None,
        help="Maximum total sampling points (default: auto-calculate based on dataset size)",
    )
    
    # Output options
    output_group = run_parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o", "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)",
    )
    output_group.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Save visualization plots (default: True)",
    )
    output_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # ==================== visualize subcommand ====================
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Visualize sampling distribution evolution",
    )
    viz_parser.add_argument(
        "-d", "--data",
        type=str,
        required=True,
        help="Path to dataset CSV file",
    )
    viz_parser.add_argument(
        "-a", "--algorithm",
        type=str,
        choices=["RS", "PDC", "DPP-PDC"],
        default="DPP-PDC",
        help="Algorithm to visualize (default: DPP-PDC)",
    )
    viz_parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=10,
        help="Batch size (default: 10)",
    )
    viz_parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=5,
        help="Number of iterations to visualize (default: 5)",
    )
    viz_parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="visualization_output",
        help="Output directory for plots (default: visualization_output)",
    )

    # ==================== init-config subcommand ====================
    config_parser = subparsers.add_parser(
        "init-config",
        help="Generate a default configuration file",
    )
    config_parser.add_argument(
        "-o", "--output",
        type=str,
        default="config.toml",
        help="Output path for config file (default: config.toml)",
    )

    return parser


def load_config(config_path: str) -> dict:
    """Load configuration from TOML file."""
    import sys
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def run_experiments(args):
    """Execute the run command."""
    # Import here to avoid slow startup for help messages
    from dpp_pdc.config import ExperimentConfig
    from dpp_pdc.run_experiments import (
        run_single_temperature_experiment,
        run_multiple_temperature_experiments,
    )
    
    # Load config file if provided
    if args.config:
        config_dict = load_config(args.config)
        config = ExperimentConfig.from_dict(config_dict)
        print(f"Loaded configuration from: {args.config}")
    else:
        # Build config from command line arguments
        config = ExperimentConfig()
        config.temperatures = args.temperatures
        config.algorithms = args.algorithms
        config.batch_sizes = args.batch_sizes
        config.n_runs = args.n_runs
        config.n_initial_sampling = args.n_initial
        config.max_sampling=args.max_sampling,
        config.output_dir = args.output_dir
        config.file_path_template = args.data_template
        
        # If single data file provided, override template
        if args.data:
            config.file_path_template = args.data
            config.temperatures = ["complete"]
    
    # Print configuration summary
    print("=" * 60)
    print("DPP-PDC: Active Learning Phase Discovery")
    print("=" * 60)
    print(f"Temperatures: {config.temperatures}")
    print(f"Algorithms: {config.algorithms}")
    print(f"Batch sizes: {config.batch_sizes}")
    print(f"Runs per configuration: {config.n_runs}")
    print(f"Output directory: {config.output_dir}")
    print("=" * 60)
    
    # Run experiments
    if len(config.temperatures) == 1:
        result = run_single_temperature_experiment(
            temperature=config.temperatures[0],
            file_path_template=config.file_path_template,
            algorithms=config.algorithms,
            batch_sizes=config.batch_sizes,
            n_runs=config.n_runs,
            max_sampling=args.max_sampling,
        )
        print(f"\nExperiment completed! Results saved to: {result['result_dir']}")
    else:
        results = run_multiple_temperature_experiments(
            temperatures=config.temperatures,
            file_path_template=config.file_path_template,
            algorithms=config.algorithms,
            batch_sizes=config.batch_sizes,
            n_runs=config.n_runs,
            max_sampling=args.max_sampling,
        )
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        for result in results:
            print(f"Temperature {result['temperature']}: {result['phases_nums']} phases")


def run_visualization(args):
    """Execute the visualize command."""
    from dpp_pdc.sampling_distribution import run_sampling_demo
    
    run_sampling_demo(
        file_path=args.data,
        algorithm=args.algorithm,
        batch_size=args.batch_size,
        max_iterations=args.iterations,
        temperature="complete",
    )


def generate_config(args):
    """Generate a default configuration file."""
    default_config = '''# DPP-PDC Configuration File
# See documentation for detailed parameter descriptions

[experiment]
# Temperature conditions to evaluate
temperatures = ["500K", "650K", "1000K", "1050K"]

# Algorithms to compare
# - RS: Random Sampling (baseline)
# - PDC: Phase Diagram Construction with uncertainty sampling
# - K-Medoids: K-medoids clustering on top uncertain points
# - DPP-PDC: Our proposed method with Bayesian DPP
algorithms = ["RS", "PDC", "K-Medoids", "DPP-PDC"]

# Batch sizes for each sampling iteration
batch_sizes = [5, 8, 10, 12]

# Number of independent runs per configuration
n_runs = 10

# Number of initial random samples
n_initial_sampling = 10

# Maximum samplings per experiment
# Set to a number to limit iterations, or comment out for auto-calculation
# max_sampling = 300

[data]
# Template for dataset file paths
# Use {temperature} as placeholder for temperature value
file_path_template = "datasets/Cu-Mg_Zn_{temperature}.csv"

[output]
# Directory for saving results
output_dir = "results"

# Save visualization plots
save_plots = true

# Verbose output
verbose = false

[bayesian]
# Prior parameters for noise variance
prior_mu = -4.0
prior_sigma = 4.0

# MCMC sampling parameters
mcmc_samples = 1000
mcmc_tune = 1000
mcmc_chains = 1
target_accept = 0.9
'''
    
    output_path = Path(args.output)
    output_path.write_text(default_config)
    print(f"Configuration file generated: {output_path}")
    print("Edit this file to customize your experiment settings.")


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "run":
        run_experiments(args)
    elif args.command == "visualize":
        run_visualization(args)
    elif args.command == "init-config":
        generate_config(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
