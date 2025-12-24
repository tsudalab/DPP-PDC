"""
Experiment runner module for active learning phase discovery.
Coordinates the execution of experiments across different algorithms and configurations.
"""

import os
import numpy as np
import pandas as pd
import functools
import warnings
import matplotlib.pyplot as plt

from dpp_pdc.data_preprocessing import standardize_data, prepare_initial_samples
from dpp_pdc.active_learning import active_learning_with_pymc, active_learning_traditional
from dpp_pdc.visualization import (
    plot_combined_results,
    plot_empty_ternary_diagram,
    save_results_to_csv,
    save_pymc_results_to_csv,
    save_metrics_to_csv,
    compute_auc_from_results,
    plot_auc_bar_chart,
    plot_combined_grid,
    plot_metrics_curves,
)

# Enable immediate output flushing
print = functools.partial(print, flush=True)

# Suppress all warnings
warnings.filterwarnings("ignore")


def run_experiment_with_pymc(file_path, temperature, algorithms, batch_sizes, n_runs, phases_nums, result_dir, max_sampling=None):
    """
    Main function to run experiments using improved PyMC Bayesian framework.
    Supports all algorithms.
    
    Args:
        file_path: Path to data file
        temperature: Temperature condition
        algorithms: List of algorithms to test
        batch_sizes: List of batch sizes
        n_runs: Number of runs per configuration
        phases_nums: Total number of phases
        result_dir: Directory to save results
        max_sampling: Maximum samplings (None = auto-calculate)
        
    Returns:
        results: List of experimental results [mean, std, algorithm, batch_size]
        pymc_results: Dictionary of PyMC noise variance results {'mean': [...], 'std': [...]}
        metrics_results: Dictionary of classification metrics results
    """
    scaler, X_std, y_encoded, data = standardize_data(file_path)
    total_samples = X_std.shape[0]
    results = []
    
    pymc_results = {}
    metrics_results = {}

    # Set experiment parameters
    n_initial_sampling = 10
    
    # Check if ternary system (3 components + 1 phase_name = 4 columns)
    is_ternary = len(data.columns) == 4
    
    # Pre-generate initial sampling points for each batch size and run
    initial_samples = prepare_initial_samples(X_std, y_encoded, batch_sizes, n_runs, n_initial_sampling)
    
    for batch_size in batch_sizes:
        # Calculate max_iterations from max_sampling
        if max_sampling is not None:
            current_max_iterations = max_sampling // batch_size
        else:
            current_max_iterations = (len(X_std) // batch_size) + 1

        # Run experiments for each algorithm
        for algorithm in algorithms:
            print(f'\nEvaluating algorithm: {algorithm}, batch size: {batch_size}')
            phases_discovered_all_runs = []
            metrics_all_runs = []
            
            if algorithm.startswith('DPP-'):
                noise_variance_history_all_runs = []

            for run in range(n_runs):
                # Get pre-generated initial sampling points
                labeled_indices = initial_samples[(batch_size, run)]
                
                print(f"Run {run+1}/{n_runs}, initial phases discovered: {len(np.unique(y_encoded[labeled_indices]))}")
                
                # Choose different active learning methods based on algorithm type
                if algorithm.startswith('DPP-'):
                    # Only plot ternary diagram for 3-component systems
                    if run == 0 and is_ternary:
                        plot_empty_ternary_diagram(algorithm, batch_size, temperature, result_dir, data)

                    # Use PyMC Bayesian framework method
                    phases_discovered, noise_variance_history, selected_points_history, metrics_history = active_learning_with_pymc(
                        data, X_std, y_encoded,
                        algorithm,
                        n_initial_sampling,
                        current_max_iterations,
                        batch_size,
                        labeled_indices, 
                        phases_nums
                    )
                    
                    noise_variance_history_all_runs.append(noise_variance_history)
                    metrics_all_runs.append(metrics_history)

                else:
                    # For non-DPP algorithms, use traditional active learning methods
                    phases_discovered, metrics_history = active_learning_traditional(
                        data, X_std, y_encoded,
                        algorithm,
                        n_initial_sampling,
                        current_max_iterations,
                        batch_size,
                        labeled_indices, 
                        phases_nums
                    )
                    metrics_all_runs.append(metrics_history)
                    
                phases_discovered_all_runs.append(phases_discovered)
            
            # Process phase count results
            padded_lists = []
            for sublist in phases_discovered_all_runs:
                expected_length = current_max_iterations + 1 
                if len(sublist) < expected_length:
                    padded_list = sublist + [sublist[-1]] * (expected_length - len(sublist))
                else:
                    padded_list = sublist[:expected_length]
                padded_lists.append(padded_list)
            
            # Calculate mean and std
            mean_results = np.mean(np.array(padded_lists), axis=0)
            std_results = np.std(np.array(padded_lists), axis=0)
            results.append([mean_results, std_results, algorithm, batch_size])

            # Process metrics results
            metrics_key = f"{algorithm}_batch{batch_size}"
            metrics_results[metrics_key] = process_metrics_history(metrics_all_runs, current_max_iterations)

            if algorithm.startswith('DPP-'):
                
                pymc_padded_lists = []

                for noise_vars in noise_variance_history_all_runs:
                    if len(noise_vars) < current_max_iterations:
                        padded_list = noise_vars + [noise_vars[-1]] * (current_max_iterations - len(noise_vars))
                    else:
                        padded_list = noise_vars[:current_max_iterations]
                    pymc_padded_lists.append(padded_list)
                
                if pymc_padded_lists:
                    mean_pymc_results = np.mean(np.array(pymc_padded_lists), axis=0)
                    std_pymc_results = np.std(np.array(pymc_padded_lists), axis=0)
                    
                    result_key = f"{algorithm}_batch{batch_size}"
                    pymc_results[result_key] = {'mean': mean_pymc_results, 'std': std_pymc_results}
                    
                    print(f"PyMC results collected for {result_key}: {len(mean_pymc_results)} iterations")
                else:
                    print(f"Warning: No PyMC data for {algorithm}_batch{batch_size}")
    
    # Save to CSV
    if pymc_results:
        save_pymc_results_to_csv(pymc_results, result_dir, algorithms, batch_sizes)
    
    # Save metrics to CSV
    if metrics_results:
        save_metrics_to_csv(metrics_results, result_dir, algorithms, batch_sizes)
    
    return results, pymc_results, metrics_results


def process_metrics_history(metrics_all_runs, max_iterations):
    """
    Process metrics history from multiple runs into mean and std values.
    
    Args:
        metrics_all_runs: List of metrics_history from each run
        max_iterations: Maximum number of iterations
        
    Returns:
        Dictionary with mean and std values for each metric
    """
    metric_names = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'mAP']
    processed = {name: {'mean': [], 'std': []} for name in metric_names}
    
    expected_length = max_iterations + 1  # +1 for initial
    
    for metric_name in metric_names:
        all_values = []
        for run_metrics in metrics_all_runs:
            values = []
            for m in run_metrics:
                if m is not None and metric_name in m:
                    val = m[metric_name]
                    values.append(val if val is not None else np.nan)
                else:
                    values.append(np.nan)
            
            # Pad if necessary
            if len(values) < expected_length:
                values = values + [values[-1] if values else np.nan] * (expected_length - len(values))
            else:
                values = values[:expected_length]
            
            all_values.append(values)
        
        # Compute mean and std across runs
        all_values = np.array(all_values)
        mean_values = np.nanmean(all_values, axis=0)
        std_values = np.nanstd(all_values, axis=0)
        processed[metric_name]['mean'] = mean_values.tolist()
        processed[metric_name]['std'] = std_values.tolist()
    
    return processed


def run_single_temperature_experiment(temperature, file_path_template, algorithms, batch_sizes, n_runs, max_sampling=None):
    """
    Run experiments for a single temperature condition.
    
    Args:
        temperature: Temperature condition (e.g., '850K')
        file_path_template: Template for file path with temperature placeholder
        algorithms: List of algorithms to test
        batch_sizes: List of batch sizes
        n_runs: Number of runs per configuration
        max_sampling: Maximum samplings (None = auto-calculate)
        
    Returns:
        Dictionary containing results and metadata
    """
    print(f"\n{'='*50}")
    print(f"Starting experiments for temperature {temperature}")
    print(f"{'='*50}")
    
    # Construct file path
    file_path = file_path_template.format(temperature=temperature)
    
    # Read data and calculate phase count
    data = pd.read_csv(file_path)
    phases_nums = len(data['phase_name'].unique())
    print(f"Total phases in dataset: {phases_nums}")
    
    # Create results folder for current temperature
    result_dir = f'results_pymc_{temperature}'
    os.makedirs(result_dir, exist_ok=True)

    # Run experiments
    results, pymc_results, metrics_results = run_experiment_with_pymc(
        file_path, temperature, algorithms, batch_sizes, n_runs, phases_nums, result_dir, max_sampling
    )
    
    # Plot comparison results for each batch size
    for batch_size in batch_sizes:
        # Plot phase discovery comparison
        fig_phases = plot_combined_results(results, batch_size, phases_nums, temperature)
        fig_phases.savefig(f'{result_dir}/phase_discovery_batch{batch_size}_{temperature}.png', 
                          bbox_inches='tight', dpi=300)
        plt.close(fig_phases)
    
    # Save raw data to CSV
    save_results_to_csv(results, result_dir, algorithms, batch_sizes)
    
    # === AUC Bar Chart ===
    auc_df = compute_auc_from_results(results, batch_sizes, algorithms)
    auc_df.to_csv(f'{result_dir}/auc_table_{temperature}.csv', index=False)
    print(f"AUC table saved: auc_table_{temperature}.csv")
    
    fig_auc = plot_auc_bar_chart(auc_df, title=temperature, 
                                  output_path=f'{result_dir}/auc_bar_{temperature}')
    plt.close(fig_auc)
    
    # === Combined Grid Plot (Phase Discovery + Sigma) ===
    fig_grid = plot_combined_grid(
        results=results,
        sigma_results=pymc_results,
        batch_sizes=batch_sizes,
        algorithms=algorithms,
        title=temperature,
        output_path=f'{result_dir}/combined_grid_{temperature}'
    )
    plt.close(fig_grid)
    
    # === Classification Metrics Curves ===
    fig_metrics = plot_metrics_curves(
        metrics_results=metrics_results,
        batch_sizes=batch_sizes,
        algorithms=algorithms,
        output_path=f'{result_dir}/metrics_curves_{temperature}'
    )
    plt.close(fig_metrics)
    
    print(f"Experiments for temperature {temperature} completed, results saved to {result_dir} folder")
    
    return {
        'temperature': temperature,
        'results': results,
        'phases_nums': phases_nums,
        'result_dir': result_dir,
        'metrics_results': metrics_results
    }


def run_multiple_temperature_experiments(temperatures, file_path_template, algorithms, batch_sizes, n_runs, max_sampling=None):
    """
    Run experiments for multiple temperature conditions.
    
    Args:
        temperatures: List of temperature conditions
        file_path_template: Template for file path with temperature placeholder
        algorithms: List of algorithms to test
        batch_sizes: List of batch sizes
        n_runs: Number of runs per configuration
        max_sampling: Maximum samplings (None = auto-calculate)
        
    Returns:
        List of experiment results for each temperature
    """
    all_results = []
    
    for temperature in temperatures:
        try:
            result = run_single_temperature_experiment(
                temperature, file_path_template, algorithms, batch_sizes, n_runs, max_sampling
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error processing temperature {temperature}: {e}")
            continue
    
    print("\nAll experiments completed!")
    return all_results
