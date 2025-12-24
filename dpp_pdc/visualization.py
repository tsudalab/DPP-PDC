"""
Visualization module for active learning results and ternary phase diagrams.
Handles plotting of sampling evolution, phase discovery curves, and ternary diagrams.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import MultiPolygon


def plot_combined_results(results, batch_size, phases_nums, temperature):
    """
    Plot phase discovery results for different algorithms on the same figure for comparison.
    
    Args:
        results: List containing all algorithm results [mean, std, algorithm, batch_size]
        batch_size: Batch size
        phases_nums: Total phases
        temperature: Temperature condition
        
    Returns:
        matplotlib figure object
    """
    plt.figure(figsize=(12, 8))
    
    # Algorithm color mapping
    algorithm_colors = {
        'RS': 'gray',
        'PDC': 'purple',
        'TS': 'blue',
        'DPP-PDC': 'green',
        'DPP-TS': 'red',
        'K-Medoids': 'orange'
    }
    
    # Filter data for current batch size (index 3 is batch_size)
    batch_data = [result for result in results if result[3] == batch_size]
    
    # Plot results for each algorithm
    for result in batch_data:
        mean_values = np.array(result[0])
        std_values = np.array(result[1])
        algorithm = result[2]
        iterations = [10 + batch_size * i for i in range(len(mean_values))]
        
        color = algorithm_colors.get(algorithm, 'black')
        plt.plot(iterations, mean_values, 
                color=color, 
                label=algorithm, 
                linewidth=2)
        plt.fill_between(iterations, mean_values - std_values, mean_values + std_values,
                        color=color, alpha=0.15)
    
    # Set plot properties
    plt.title(f'{temperature}, Batch Size: {batch_size}', fontsize=16)
    plt.xlabel('Number of sampled candidate points', fontsize=14)
    plt.ylabel('Number of discovered phases', fontsize=14)
    plt.ylim(0, phases_nums)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    return plt.gcf()


def plot_ternary_phase_diagram(data, sampled_indices=None, previous_indices=None):
    """
    Plot ternary phase diagram with precise phase boundaries.
    
    Args:
        data: Original DataFrame containing phase data
        sampled_indices: Indices of newly sampled points
        previous_indices: Indices of previously sampled points
        
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Convert ternary compositions to cartesian for Voronoi calculation
    points = data[['Zn', 'Mg', 'Cu']].values
    x, y = ternary2cart(points)
    
    # Add points outside the triangle to ensure bounded Voronoi regions
    margin = 1
    extra_points = []
    for i in range(-margin, margin+1):
        for j in range(-margin, margin+1):
            p = [i*120, j*120]
            if p[0] != 0 or p[1] != 0:
                extra_points.append(p)
    
    # Combine original and extra points for Voronoi calculation
    vor_points = np.vstack((np.column_stack((x, y)), extra_points))
    vor = Voronoi(vor_points)
    
    # Define triangle boundary
    triangle = np.array([[0, 0], [100, 0], [50, 86.6]])
    triangle_boundary = ShapelyPolygon(triangle)
    triangle_path = Polygon(triangle, fill=False, color='black', linewidth=2)
    ax.add_patch(triangle_path)
    
    # Plot phase regions using Voronoi cells
    unique_phases = data['phase_name'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_phases)))
    phase_colors = dict(zip(unique_phases, colors))
    
    # Process each Voronoi region
    for i, phase in enumerate(data['phase_name']):
        if i >= len(vor.point_region):
            continue
            
        region_idx = vor.point_region[i]
        if region_idx == -1:
            continue
            
        region = vor.regions[region_idx]
        if -1 in region:
            continue
            
        polygon = [vor.vertices[j] for j in region]
        if len(polygon) < 3:
            continue
            
        # Create Shapely polygon and clip with triangle boundary
        try:
            polygon = ShapelyPolygon(polygon)
            clipped_polygon = polygon.intersection(triangle_boundary)
            
            if isinstance(clipped_polygon, MultiPolygon):
                for p in clipped_polygon.geoms:
                    poly = Polygon(p.exterior.coords, 
                                 facecolor=phase_colors[phase], 
                                 alpha=0.3, 
                                 edgecolor='none')
                    ax.add_patch(poly)
            elif isinstance(clipped_polygon, ShapelyPolygon):
                poly = Polygon(clipped_polygon.exterior.coords, 
                             facecolor=phase_colors[phase], 
                             alpha=0.3, 
                             edgecolor='none')
                ax.add_patch(poly)
        except:
            continue
    
    # Plot previously sampled points (if any)
    if previous_indices is not None and len(previous_indices) > 0:
        for idx in previous_indices:
            sample_point = data.iloc[idx]
            x, y = ternary2cart(np.array([[sample_point['Zn'], 
                                         sample_point['Mg'], 
                                         sample_point['Cu']]]))
            phase = sample_point['phase_name']
            # Use light color for previously sampled points
            ax.scatter(x, y, color=phase_colors[phase],
                      marker='o', s=20, linewidth=1.5, alpha=0.3, edgecolor='gray')
    
    # Plot newly sampled points
    if sampled_indices is not None and len(sampled_indices) > 0:
        for idx in sampled_indices:
            sample_point = data.iloc[idx]
            x, y = ternary2cart(np.array([[sample_point['Zn'], 
                                         sample_point['Mg'], 
                                         sample_point['Cu']]]))
            phase = sample_point['phase_name']
            # Use bold color for newly sampled points
            ax.scatter(x, y, color=phase_colors[phase],
                      marker='o', s=30, linewidth=2, alpha=1.0, edgecolor='black')
    
    # Customize plot
    plt.axis('equal')
    plt.xlim(-10, 110)
    plt.ylim(-10, 95)
    ax.axis('off')
    
    # Add labels
    plt.text(-4, 0, 'Cu', ha='center', va='center', fontsize=20)
    plt.text(50, 90, 'Zn', ha='center', va='center', fontsize=20)
    plt.text(105, 0, 'Mg', ha='center', va='center', fontsize=20)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.2, 0.5), loc='center left', fontsize=8)
    
    total_points = len(previous_indices or []) + len(sampled_indices or [])
    plt.title(f'New Points: {len(sampled_indices or [])}, Total Sampled: {total_points}', fontsize=14, pad=20)
    
    return fig, ax


def ternary2cart(points):
    """
    Convert ternary coordinates to cartesian coordinates.
    
    Args:
        points: Array of ternary coordinates
        
    Returns:
        x, y: Cartesian coordinates
    """
    x = 0.5 * points[:,1] + points[:,2]
    y = 0.866 * points[:,1]
    return x, y


def plot_empty_ternary_diagram(algorithm, batch_size, temperature, result_dir, data):
    """
    Plot an empty ternary diagram without any sampling points.
    
    Args:
        algorithm: Algorithm name
        batch_size: Batch size
        temperature: Temperature condition
        result_dir: Directory to save results
        data: Original data
    """
    fig, ax = plot_ternary_phase_diagram(data, sampled_indices=None, previous_indices=None)
    filename = f"{algorithm}_batch{batch_size}_{temperature}_sampling_round0_empty.png"
    save_path = os.path.join(result_dir, filename)
    plt.title("Phase Diagram - Cu-Mg-Zn_{}".format(temperature), fontsize=14, pad=20)
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()


def save_results_to_csv(results, result_dir, algorithms, batch_sizes, include_sampling_points=True):
    """
    Save experimental results to CSV files.
    
    Args:
        results: List of experimental results [mean, std, algorithm, batch_size]
        result_dir: Directory to save results
        algorithms: List of algorithms
        batch_sizes: List of batch sizes
        include_sampling_points: Whether to include sampling points column
    """
    for batch_size in batch_sizes:
        for algorithm in algorithms:
            result_data = None
            std_data = None
            for r in results:
                if r[2] == algorithm and r[3] == batch_size:
                    result_data = r[0]
                    std_data = r[1]
                    break
            
            if result_data is not None:
                iterations = range(1, len(result_data) + 1)
                
                df_dict = {
                    'Iteration': iterations,
                    'Phases_Discovered_Mean': np.round(result_data, 2),
                    'Phases_Discovered_Std': np.round(std_data, 2) if std_data is not None else [0] * len(result_data)
                }
                
                # Only add sampling points for ternary datasets
                if include_sampling_points:
                    sampling_points = [10 + batch_size * (i-1) for i in iterations]
                    df_dict['Sampling_Points'] = sampling_points
                
                df_results = pd.DataFrame(df_dict)
                df_results.to_csv(
                    f'{result_dir}/{algorithm}_batch{batch_size}_results.csv', 
                    index=False
                )


def save_pymc_results_to_csv(pymc_results, result_dir, algorithms, batch_sizes):
    """
    Save PyMC noise variance results to CSV files.
    
    Args:
        pymc_results: Dictionary containing PyMC results {'mean': [...], 'std': [...]}
        result_dir: Directory to save results
        algorithms: List of algorithms
        batch_sizes: List of batch sizes
    """
    for batch_size in batch_sizes:
        for algorithm in algorithms:
            
            if not algorithm.startswith('DPP-'):
                continue
                
            result_key = f"{algorithm}_batch{batch_size}"
            if result_key not in pymc_results:
                continue
            
            pymc_data = pymc_results[result_key]
            if pymc_data is None:
                continue
            
            # Handle both old format (array) and new format (dict with mean/std)
            if isinstance(pymc_data, dict):
                mean_data = pymc_data.get('mean', [])
                std_data = pymc_data.get('std', [])
            else:
                mean_data = pymc_data
                std_data = [0] * len(mean_data)
            
            if len(mean_data) == 0:
                continue
            
            iterations = range(1, len(mean_data) + 1)
            sampling_points = [10 + batch_size * (i-1) for i in iterations]
            
            df_pymc = pd.DataFrame({
                'Iteration': iterations,
                'Sampling_Points': sampling_points,
                'Noise_Variance_Mean': mean_data,
                'Noise_Variance_Std': std_data
            })
            
            filename = f'{algorithm}_batch{batch_size}_pymc_results.csv'
            save_path = os.path.join(result_dir, filename)
            df_pymc.to_csv(save_path, index=False)
            print(f"PyMC results saved: {filename}")


def save_metrics_to_csv(metrics_results, result_dir, algorithms, batch_sizes):
    """
    Save classification metrics results to CSV files.
    
    Args:
        metrics_results: Dictionary containing metrics results
        result_dir: Directory to save results
        algorithms: List of algorithms
        batch_sizes: List of batch sizes
    """
    for batch_size in batch_sizes:
        for algorithm in algorithms:
            result_key = f"{algorithm}_batch{batch_size}"
            if result_key not in metrics_results:
                continue
            
            metrics_data = metrics_results[result_key]
            if metrics_data is None:
                continue
            
            # Get length from first metric
            first_metric = list(metrics_data.values())[0]
            if isinstance(first_metric, dict):
                n_iterations = len(first_metric.get('mean', []))
            else:
                n_iterations = len(first_metric)
            
            if n_iterations == 0:
                continue
            
            iterations = range(1, n_iterations + 1)
            sampling_points = [10 + batch_size * (i-1) for i in iterations]
            
            df_dict = {
                'Iteration': iterations,
                'Sampling_Points': sampling_points,
            }
            
            for metric_name in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'mAP']:
                metric_data = metrics_data.get(metric_name, {})
                if isinstance(metric_data, dict):
                    df_dict[f'{metric_name}_mean'] = metric_data.get('mean', [np.nan] * n_iterations)
                    df_dict[f'{metric_name}_std'] = metric_data.get('std', [np.nan] * n_iterations)
                else:
                    df_dict[f'{metric_name}_mean'] = metric_data if metric_data else [np.nan] * n_iterations
                    df_dict[f'{metric_name}_std'] = [np.nan] * n_iterations
            
            df_metrics = pd.DataFrame(df_dict)
            
            filename = f'{algorithm}_batch{batch_size}_metrics.csv'
            save_path = os.path.join(result_dir, filename)
            df_metrics.to_csv(save_path, index=False)
            print(f"Metrics saved: {filename}")


def plot_metrics_curves(metrics_results, batch_sizes, algorithms, output_path=None, n_initial=10, dpi=300):
    """
    Plot classification metrics curves for all algorithms with +/-std bands.
    
    Args:
        metrics_results: Dictionary with metrics data
        batch_sizes: List of batch sizes
        algorithms: List of algorithm names
        output_path: Path to save figure
        n_initial: Number of initial sampling points
        dpi: Resolution for saved figure
        
    Returns:
        matplotlib figure object
    """
    from matplotlib.lines import Line2D
    
    # Style configuration
    style = {
        'lw': 1.6,
        'alpha_band': 0.15,
        'grid_ls': '--',
        'grid_lw': 0.5,
        'grid_alpha': 0.45,
        'col_title_fs': 12,
        'xlabel_fs': 12,
        'ylabel_fs': 11,
        'legend_fs': 10,
        'tick_fs': 9,
    }
    colors = {'RS': 'gray', 'PDC': 'purple', 'DPP-PDC': 'green', 'K-Medoids': 'orange'}
    
    # Metrics to plot
    metric_names = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    n_cols = len(batch_sizes)
    n_rows = len(metric_names)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows))
    
    # Handle single column/row case
    if n_cols == 1 and n_rows == 1:
        axes = np.array([[axes]])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.98, hspace=0.35, wspace=0.25)
    
    # Plot each metric
    for row, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
        for col, bs in enumerate(batch_sizes):
            ax = axes[row, col]
            
            for alg in algorithms:
                key = f"{alg}_batch{bs}"
                if key in metrics_results and metrics_results[key] is not None:
                    metric_data = metrics_results[key].get(metric_name, {})
                    if isinstance(metric_data, dict) and 'mean' in metric_data:
                        mean_values = np.array(metric_data['mean'])
                        std_values = np.array(metric_data.get('std', np.zeros_like(mean_values)))
                        if len(mean_values) > 0:
                            x = np.array([n_initial + bs * i for i in range(len(mean_values))])
                            c = colors.get(alg, 'black')
                            ax.plot(x, mean_values, linewidth=style['lw'], color=c)
                            ax.fill_between(x, mean_values - std_values, mean_values + std_values,
                                          color=c, alpha=style['alpha_band'])
            
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle=style['grid_ls'], linewidth=style['grid_lw'], alpha=style['grid_alpha'])
            ax.tick_params(labelsize=style['tick_fs'])
            
            # Column titles (only top row)
            if row == 0:
                ax.set_title(f"Batch size = {bs}", fontsize=style['col_title_fs'], pad=8)
            
            # Row labels (only first column)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=style['ylabel_fs'], fontweight='bold')
    
    # X-axis label
    fig.supxlabel("Total sampling points", fontsize=style['xlabel_fs'], fontweight='bold', y=0.02)
    
    # Legend
    handles = [
        Line2D([0], [0], color=colors.get('RS', 'gray'), lw=style['lw'] * 1.5, label='RS'),
        Line2D([0], [0], color=colors.get('PDC', 'purple'), lw=style['lw'] * 1.5, label='PDC'),
        Line2D([0], [0], color=colors.get('K-Medoids', 'orange'), lw=style['lw'] * 1.5, label='K-Medoids'),
        Line2D([0], [0], color=colors.get('DPP-PDC', 'green'), lw=style['lw'] * 1.5, label='DPP-PDC'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
               fontsize=style['legend_fs'], bbox_to_anchor=(0.5, 0.98))
    
    # Save if path provided
    if output_path:
        base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        for ext in ('png', 'pdf'):
            save_path = f"{base_path}.{ext}"
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"Metrics curves saved: {save_path}")
    
    return fig


# =============================================================================
# AUC Computation and Bar Chart Plotting
# =============================================================================

def compute_auc(x, y):
    """
    Compute Area Under Curve using trapezoidal rule.
    
    Args:
        x: X-axis values (e.g., sampling points)
        y: Y-axis values (e.g., discovered phases)
        
    Returns:
        AUC value
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # NumPy 2.0+ renamed trapz to trapezoid
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    else:
        return np.trapz(y, x)


def compute_normalized_auc(x, y, y_max=None):
    """
    Compute normalized AUC (nAUC) in range [0, 1].
    
    Args:
        x: X-axis values
        y: Y-axis values
        y_max: Maximum theoretical y value (default: max of y)
        
    Returns:
        Normalized AUC value
    """
    auc = compute_auc(x, y)
    x_span = float(np.max(x) - np.min(x))
    if y_max is None:
        y_max = float(np.max(y))
    denom = x_span * y_max
    return auc / denom if denom > 0 else np.nan


def compute_auc_from_results(results, batch_sizes, algorithms, n_initial=10):
    """
    Compute AUC and nAUC for all algorithm-batch combinations from experiment results.
    
    Args:
        results: List of [mean_results, std_results, algorithm, batch_size] from experiments
        batch_sizes: List of batch sizes
        algorithms: List of algorithm names
        n_initial: Number of initial sampling points
        
    Returns:
        DataFrame with columns: batch_size, algorithm, AUC, nAUC
    """
    rows = []
    
    # First pass: find y_max per batch size (for normalization)
    y_max_by_batch = {}
    for bs in batch_sizes:
        finals = []
        for result in results:
            if result[3] == bs:  # index 3 is batch_size
                finals.append(result[0][-1])  # index 0 is mean_values
        y_max_by_batch[bs] = float(np.max(finals)) if finals else None
    
    # Second pass: compute AUC for each combination
    for result in results:
        mean_values = result[0]
        algorithm = result[2]  # index 2 is algorithm
        batch_size = result[3]  # index 3 is batch_size
        
        # X-axis: total sampling points
        x = np.array([n_initial + batch_size * i for i in range(len(mean_values))])
        y = np.array(mean_values)
        
        auc = compute_auc(x, y)
        nauc = compute_normalized_auc(x, y, y_max=y_max_by_batch.get(batch_size))
        
        rows.append({
            'batch_size': batch_size,
            'algorithm': algorithm,
            'AUC': auc,
            'nAUC': nauc
        })
    
    return pd.DataFrame(rows)


def plot_auc_bar_chart(auc_df, title="", output_path=None, dpi=1000):
    """
    Plot grouped bar chart of AUC values.
    
    Args:
        auc_df: DataFrame with columns: batch_size, algorithm, AUC
        title: Plot title (not used, kept for compatibility)
        output_path: Path to save figure (optional)
        dpi: Resolution for saved figure
        
    Returns:
        matplotlib figure object
    """
    # Style configuration
    colors = {'RS': 'gray', 'PDC': 'purple', 'DPP-PDC': 'green', 'K-Medoids': 'orange'}
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Get unique batch sizes and algorithms (fixed order)
    batch_sizes = sorted(auc_df['batch_size'].unique())
    algorithms = [alg for alg in ['RS', 'PDC', 'K-Medoids', 'DPP-PDC'] if alg in auc_df['algorithm'].values]
    
    # Plot grouped bars
    x = np.arange(len(batch_sizes))
    n_algorithms = len(algorithms)
    width = 0.8 / n_algorithms
    
    for i, alg in enumerate(algorithms):
        vals = []
        for bs in batch_sizes:
            mask = (auc_df['batch_size'] == bs) & (auc_df['algorithm'] == alg)
            vals.append(auc_df.loc[mask, 'AUC'].values[0] if mask.any() else np.nan)
        ax.bar(x + (i - n_algorithms/2 + 0.5) * width, vals, width=width, label=alg, color=colors.get(alg, 'black'))
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=10)
    ax.set_xlabel('Batch size', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    
    # Legend at top, all in one row
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='upper center', ncol=4, 
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save if path provided
    if output_path:
        base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        for ext in ('png', 'pdf'):
            save_path = f"{base_path}.{ext}"
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"AUC bar chart saved: {save_path}")
    
    return fig


# =============================================================================
# Combined Grid Plot (Phase Discovery + Sigma Evolution)
# =============================================================================

def plot_combined_grid(results, sigma_results, batch_sizes, algorithms, 
                       title="", output_path=None, n_initial=10, dpi=1000):
    """
    Plot combined grid: Row 1 = Phase discovery, Row 2 = Sigma evolution.
    Number of columns = number of batch sizes (dynamic).
    With +/-std shaded bands.
    
    Args:
        results: List of [mean_results, std_results, algorithm, batch_size] for phase discovery
        sigma_results: Dict {key: {'mean': [...], 'std': [...]}} for sigma evolution
        batch_sizes: List of batch sizes to plot
        algorithms: List of algorithm names
        title: Main title (not used, kept for compatibility)
        output_path: Path to save figure (optional)
        n_initial: Number of initial sampling points
        dpi: Resolution for saved figure
        
    Returns:
        matplotlib figure object
    """
    from matplotlib.lines import Line2D
    
    # Style configuration
    style = {
        'lw': 1.6,
        'alpha_band': 0.15,
        'grid_ls': '--',
        'grid_lw': 0.5,
        'grid_alpha': 0.45,
        'title_fs': 16,
        'col_title_fs': 13,
        'xlabel_fs': 13,
        'ylabel_fs': 13,
        'legend_fs': 13,
        'tick_fs': 10,
    }
    colors = {'RS': 'gray', 'PDC': 'purple', 'DPP-PDC': 'green', 'K-Medoids': 'orange'}
    sigma_color = 'blue'
    
    n_cols = len(batch_sizes)
    fig_width = 3 * n_cols  # ~3 inches per column
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_width, 6))
    
    # Handle single column case
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.98, hspace=0.35, wspace=0.25)
    
    # Collect Y-axis ranges for consistent scaling (considering std)
    phase_y_all = []
    sigma_y_all = []
    
    # First pass: collect Y ranges
    for result in results:
        mean_values = np.array(result[0])
        std_values = np.array(result[1])
        phase_y_all.extend(mean_values + std_values)
        phase_y_all.extend(mean_values - std_values)
    
    if sigma_results:
        for key, sigma_data in sigma_results.items():
            if isinstance(sigma_data, dict) and 'mean' in sigma_data:
                sigma_mean = np.array(sigma_data['mean'])
                sigma_std = np.array(sigma_data.get('std', np.zeros_like(sigma_mean)))
                if len(sigma_mean) > 0:
                    sigma_y_all.extend(sigma_mean + sigma_std)
                    sigma_y_all.extend(sigma_mean - sigma_std)
            elif sigma_data is not None and len(sigma_data) > 0:
                # Old format compatibility
                sigma_y_all.extend(sigma_data)
    
    # Determine Y limits with padding
    if phase_y_all:
        phase_ymin, phase_ymax = min(phase_y_all), max(phase_y_all)
        phase_margin = (phase_ymax - phase_ymin) * 0.05
        phase_ylim = (max(0, phase_ymin - phase_margin), phase_ymax + phase_margin)
    else:
        phase_ylim = (0, 50)
    
    if sigma_y_all:
        sigma_ymin, sigma_ymax = min(sigma_y_all), max(sigma_y_all)
        sigma_margin = (sigma_ymax - sigma_ymin) * 0.05
        sigma_ylim = (max(0, sigma_ymin - sigma_margin), sigma_ymax + sigma_margin)
    else:
        sigma_ylim = (1e-5, 1.5e-5)
    
    # Second pass: plot
    for j, bs in enumerate(batch_sizes):
        ax_phase = axes[0, j]
        ax_sigma = axes[1, j]
        
        # --- Row 1: Phase discovery with std bands ---
        for result in results:
            if result[3] != bs:  # index 3 is batch_size
                continue
            mean_values = np.array(result[0])
            std_values = np.array(result[1])
            algorithm = result[2]  # index 2 is algorithm
            
            x = np.array([n_initial + bs * i for i in range(len(mean_values))])
            c = colors.get(algorithm, 'black')
            ax_phase.plot(x, mean_values, linewidth=style['lw'], color=c)
            ax_phase.fill_between(x, mean_values - std_values, mean_values + std_values,
                                  color=c, alpha=style['alpha_band'])
        
        ax_phase.set_ylim(phase_ylim)
        ax_phase.grid(True, linestyle=style['grid_ls'], linewidth=style['grid_lw'], alpha=style['grid_alpha'])
        ax_phase.tick_params(labelsize=style['tick_fs'])
        ax_phase.set_title(f"Batch size = {bs}", fontsize=style['col_title_fs'], pad=8)
        
        if j == 0:
            ax_phase.set_ylabel("Discovered phases", fontsize=style['ylabel_fs'], fontweight='bold')
        
        # --- Row 2: Sigma evolution with std bands ---
        for alg in algorithms:
            if not alg.startswith('DPP-'):
                continue
            key = f"{alg}_batch{bs}"
            if sigma_results and key in sigma_results:
                sigma_data = sigma_results[key]
                if isinstance(sigma_data, dict) and 'mean' in sigma_data:
                    sigma_mean = np.array(sigma_data['mean'])
                    sigma_std = np.array(sigma_data.get('std', np.zeros_like(sigma_mean)))
                    if len(sigma_mean) > 0:
                        x = np.array([n_initial + bs * i for i in range(len(sigma_mean))])
                        ax_sigma.plot(x, sigma_mean, linewidth=style['lw'], color=sigma_color)
                        ax_sigma.fill_between(x, sigma_mean - sigma_std, sigma_mean + sigma_std,
                                              color=sigma_color, alpha=style['alpha_band'])
                elif sigma_data is not None and len(sigma_data) > 0:
                    # Old format compatibility
                    x = np.array([n_initial + bs * i for i in range(len(sigma_data))])
                    ax_sigma.plot(x, sigma_data, linewidth=style['lw'], color=sigma_color)
        
        ax_sigma.set_ylim(sigma_ylim)
        ax_sigma.grid(True, linestyle=style['grid_ls'], linewidth=style['grid_lw'], alpha=style['grid_alpha'])
        ax_sigma.tick_params(labelsize=style['tick_fs'])
        
        if j == 0:
            ax_sigma.set_ylabel(r"$\sigma^2$", fontsize=style['ylabel_fs'], fontweight='bold')
    
    # X-axis label
    fig.supxlabel("Total sampling points", fontsize=style['xlabel_fs'], fontweight='bold', y=0.02)
    
    # Legend (fixed order: RS, PDC, K-Medoids, DPP-PDC)
    handles = [
        Line2D([0], [0], color=colors.get('RS', 'gray'), lw=style['lw'] * 1.5, label='RS'),
        Line2D([0], [0], color=colors.get('PDC', 'purple'), lw=style['lw'] * 1.5, label='PDC'),
        Line2D([0], [0], color=colors.get('K-Medoids', 'orange'), lw=style['lw'] * 1.5, label='K-Medoids'),
        Line2D([0], [0], color=colors.get('DPP-PDC', 'green'), lw=style['lw'] * 1.5, label='DPP-PDC'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
               fontsize=style['legend_fs'], bbox_to_anchor=(0.5, 0.98))
    
    # Save if path provided
    if output_path:
        base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        for ext in ('png', 'pdf'):
            save_path = f"{base_path}.{ext}"
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"Combined grid plot saved: {save_path}")
    
    return fig


def plot_combined_grid_with_std(result_dir, batch_sizes, algorithms, 
                                title="", output_path=None, dpi=1000):
    """
    Plot combined grid from CSV files with mean +/- std bands.
    Expects CSV files with columns: Iteration, Mean, Std
    
    Args:
        result_dir: Directory containing CSV result files
        batch_sizes: List of batch sizes
        algorithms: List of algorithm names
        title: Main title (not used, kept for compatibility)
        output_path: Path to save figure
        dpi: Resolution for saved figure
        
    Returns:
        matplotlib figure object
    """
    from matplotlib.lines import Line2D
    
    # Style configuration
    style = {
        'lw': 1.6,
        'alpha_band': 0.18,
        'grid_ls': '--',
        'grid_lw': 0.5,
        'grid_alpha': 0.45,
        'title_fs': 16,
        'col_title_fs': 13,
        'xlabel_fs': 13,
        'ylabel_fs': 13,
        'legend_fs': 13,
        'tick_fs': 10,
    }
    colors = {'RS': 'gray', 'PDC': 'purple', 'DPP-PDC': 'green', 'K-Medoids': 'orange'}
    sigma_color = 'blue'
    
    n_cols = len(batch_sizes)
    fig_width = 3 * n_cols
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_width, 6))
    
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.98, hspace=0.35, wspace=0.25)
    
    # Collect Y ranges
    phase_y_all = []
    sigma_y_all = []
    
    # First pass: collect ranges
    for bs in batch_sizes:
        for alg in algorithms:
            # Phase discovery
            csv_path = os.path.join(result_dir, f"{alg}_batch{bs}_phase_discovery_runs.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'Mean' in df.columns and 'Std' in df.columns:
                    phase_y_all.extend((df['Mean'] - df['Std']).tolist())
                    phase_y_all.extend((df['Mean'] + df['Std']).tolist())
            
            # Sigma
            if alg.startswith('DPP-'):
                sigma_path = os.path.join(result_dir, f"{alg}_batch{bs}_sigma_runs.csv")
                if os.path.exists(sigma_path):
                    df = pd.read_csv(sigma_path)
                    if 'Mean' in df.columns and 'Std' in df.columns:
                        sigma_y_all.extend((df['Mean'] - df['Std']).tolist())
                        sigma_y_all.extend((df['Mean'] + df['Std']).tolist())
    
    # Y limits
    if phase_y_all:
        phase_ymin, phase_ymax = min(phase_y_all), max(phase_y_all)
        phase_margin = (phase_ymax - phase_ymin) * 0.05
        phase_ylim = (max(0, phase_ymin - phase_margin), phase_ymax + phase_margin)
    else:
        phase_ylim = (0, 50)
    
    if sigma_y_all:
        sigma_ymin, sigma_ymax = min(sigma_y_all), max(sigma_y_all)
        sigma_margin = (sigma_ymax - sigma_ymin) * 0.05
        sigma_ylim = (max(0, sigma_ymin - sigma_margin), sigma_ymax + sigma_margin)
    else:
        sigma_ylim = (1e-5, 1.5e-5)
    
    # Second pass: plot
    for j, bs in enumerate(batch_sizes):
        ax_phase = axes[0, j]
        ax_sigma = axes[1, j]
        
        # --- Row 1: Phase discovery ---
        for alg in algorithms:
            csv_path = os.path.join(result_dir, f"{alg}_batch{bs}_phase_discovery_runs.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'Iteration' in df.columns and 'Mean' in df.columns:
                    x = df['Iteration'].values * bs
                    mean = df['Mean'].values
                    c = colors.get(alg, 'black')
                    ax_phase.plot(x, mean, linewidth=style['lw'], color=c)
                    if 'Std' in df.columns:
                        std = df['Std'].values
                        ax_phase.fill_between(x, mean - std, mean + std, color=c, alpha=style['alpha_band'])
        
        ax_phase.set_ylim(phase_ylim)
        ax_phase.grid(True, linestyle=style['grid_ls'], linewidth=style['grid_lw'], alpha=style['grid_alpha'])
        ax_phase.tick_params(labelsize=style['tick_fs'])
        ax_phase.set_title(f"Batch size = {bs}", fontsize=style['col_title_fs'], pad=8)
        
        if j == 0:
            ax_phase.set_ylabel("Discovered phases", fontsize=style['ylabel_fs'], fontweight='bold')
        
        # --- Row 2: Sigma ---
        for alg in algorithms:
            if not alg.startswith('DPP-'):
                continue
            sigma_path = os.path.join(result_dir, f"{alg}_batch{bs}_sigma_runs.csv")
            if os.path.exists(sigma_path):
                df = pd.read_csv(sigma_path)
                if 'Iteration' in df.columns and 'Mean' in df.columns:
                    x = df['Iteration'].values * bs
                    mean = df['Mean'].values
                    ax_sigma.plot(x, mean, linewidth=style['lw'], color=sigma_color)
                    if 'Std' in df.columns:
                        std = df['Std'].values
                        ax_sigma.fill_between(x, mean - std, mean + std, color=sigma_color, alpha=style['alpha_band'])
        
        ax_sigma.set_ylim(sigma_ylim)
        ax_sigma.grid(True, linestyle=style['grid_ls'], linewidth=style['grid_lw'], alpha=style['grid_alpha'])
        ax_sigma.tick_params(labelsize=style['tick_fs'])
        
        if j == 0:
            ax_sigma.set_ylabel(r"$\sigma^2$", fontsize=style['ylabel_fs'], fontweight='bold')
    
    # X-axis label
    fig.supxlabel("Total sampling points", fontsize=style['xlabel_fs'], fontweight='bold', y=0.02)
    
    # Legend (fixed order)
    handles = [
        Line2D([0], [0], color=colors.get('RS', 'gray'), lw=style['lw'] * 1.5, label='RS'),
        Line2D([0], [0], color=colors.get('PDC', 'purple'), lw=style['lw'] * 1.5, label='PDC'),
        Line2D([0], [0], color=colors.get('K-Medoids', 'orange'), lw=style['lw'] * 1.5, label='K-Medoids'),
        Line2D([0], [0], color=colors.get('DPP-PDC', 'green'), lw=style['lw'] * 1.5, label='DPP-PDC'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
               fontsize=style['legend_fs'], bbox_to_anchor=(0.5, 0.98))
    
    # Save
    if output_path:
        base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        for ext in ('png', 'pdf'):
            save_path = f"{base_path}.{ext}"
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"Combined grid plot saved: {save_path}")
    
    return fig
