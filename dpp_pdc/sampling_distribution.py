"""
Sampling visualization demonstration script.
Uses existing modules to run active learning and visualize sampling evolution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from dpp_pdc.data_preprocessing import standardize_data
from dpp_pdc.active_learning import active_learning_with_pymc, active_learning_traditional
from dpp_pdc.visualization import plot_empty_ternary_diagram, ternary2cart


def plot_sampling_evolution_demo(data, sampled_indices, iteration, algorithm, output_dir):
    """
    Plot ternary diagram for demonstration with enhanced visualization.
    """
    from scipy.spatial import Voronoi
    from matplotlib.patches import Polygon
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry import MultiPolygon
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Convert ternary compositions to cartesian for Voronoi calculation
    points = data[['Zn', 'Mg', 'Cu']].values
    x, y = ternary2cart(points)
    
    # Add points outside triangle for bounded Voronoi regions
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
    
    # Plot sampled points with enhanced styling
    if sampled_indices is not None and len(sampled_indices) > 0:
        for idx in sampled_indices:
            sample_point = data.iloc[idx]
            x, y = ternary2cart(np.array([[sample_point['Zn'], 
                                         sample_point['Mg'], 
                                         sample_point['Cu']]]))
            phase = sample_point['phase_name']
            ax.scatter(x, y, color=phase_colors[phase],
                      marker='o', s=40, linewidth=2, edgecolor='black', alpha=0.9)
    
    # Customize plot
    plt.axis('equal')
    plt.xlim(-10, 110)
    plt.ylim(-10, 95)
    ax.axis('off')
    
    # Add labels
    plt.text(-4, 0, 'Cu', ha='center', va='center', fontsize=20, fontweight='bold')
    plt.text(50, 90, 'Zn', ha='center', va='center', fontsize=20, fontweight='bold')
    plt.text(105, 0, 'Mg', ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Add legend for phases
    handles = [plt.scatter([], [], color=color, s=50, label=phase, edgecolor='black') 
               for phase, color in phase_colors.items()]
    plt.legend(handles=handles, bbox_to_anchor=(1.2, 0.5), loc='center left', fontsize=10)
    
    # Enhanced title
    sample_count = len(sampled_indices) if sampled_indices is not None else 0
    title = f'{algorithm} - Round {iteration} - Sampled Points: {sample_count}'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Save plot
    filename = f"{algorithm}_round_{iteration}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {save_path}")
    return save_path


def run_sampling_demo(file_path, algorithm, batch_size, max_iterations=5, temperature="demo"):
    """
    Run sampling demonstration using existing modules.
    
    Args:
        file_path: Path to CSV data file
        algorithm: Algorithm to use ('RS', 'PDC', 'DPP-PDC')
        batch_size: Number of points to sample each iteration
        max_iterations: Maximum number of iterations
        temperature: Temperature label for output
    """
    print("="*70)
    print("ACTIVE LEARNING SAMPLING DEMONSTRATION")
    print("="*70)
    print(f"Data file: {file_path}")
    print(f"Algorithm: {algorithm}")
    print(f"Batch size: {batch_size}")
    print(f"Max iterations: {max_iterations}")
    print(f"Temperature: {temperature}")
    print("="*70)
    
    # Create output directory
    output_dir = f"sampling_demo_{algorithm}_{batch_size}_{temperature}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load and prepare data
    scaler, X_std, y_encoded, data = standardize_data(file_path)
    total_samples = X_std.shape[0]
    phases_nums = len(data['phase_name'].unique())
    
    print(f"Dataset info:")
    print(f"  Total samples: {total_samples}")
    print(f"  Total phases: {phases_nums}")
    print(f"  Features: {X_std.shape[1]}")
    
    # Initial sampling setup
    n_initial = 10
    np.random.seed(42)  # For reproducibility
    labeled_indices = np.random.choice(total_samples, size=n_initial, replace=False)
    
    # Ensure at least 2 phases in initial sampling
    unique_phases = np.unique(y_encoded[labeled_indices])
    while len(unique_phases) < 2:
        remaining_indices = np.setdiff1d(np.arange(total_samples), labeled_indices)
        new_sample = np.random.choice(remaining_indices, size=1, replace=False)
        labeled_indices = np.append(labeled_indices, new_sample)
        unique_phases = np.unique(y_encoded[labeled_indices])
    
    print(f"Initial sampling:")
    print(f"  Initial points: {len(labeled_indices)}")
    print(f"  Initial phases: {len(unique_phases)}")
    
    # Plot initial state
    print(f"\nGenerating visualizations...")
    plot_sampling_evolution_demo(data, labeled_indices, 0, f"{algorithm}-Initial", output_dir)
    
    # Track sampling history for visualization
    all_sampled_indices = labeled_indices.copy()
    
    # Run active learning with our existing modules
    if algorithm.startswith('DPP-'):
        print(f"Running {algorithm} with PyMC Bayesian framework...")
        phases_discovered, noise_variance_history, selected_points_history = active_learning_with_pymc(
            data=data,
            X_std=X_std, 
            y_encoded=y_encoded,
            algorithm=algorithm,
            n_initial_sampling=n_initial,
            max_iterations=max_iterations,
            batch_size=batch_size,
            initial_labeled_indices=labeled_indices,
            phases_nums=phases_nums
        )
        
        # Generate plots for each round using the selected points history
        for i, round_points in enumerate(selected_points_history[1:], 1):
            round_indices = []
            for point in round_points:
                point_original = scaler.inverse_transform(point.reshape(1, -1))[0]
                distances = np.sum((data.iloc[:, 1:].values - point_original[:3])**2, axis=1)
                closest_idx = np.argmin(distances)
                round_indices.append(closest_idx)
            
            all_sampled_indices = np.append(all_sampled_indices, round_indices)
            plot_sampling_evolution_demo(data, all_sampled_indices, i, algorithm, output_dir)
            print(f"   Round {i}: +{len(round_indices)} points, total phases: {phases_discovered[i-1] if i-1 < len(phases_discovered) else 'N/A'}")
            
    else:
        print(f"Running traditional {algorithm} algorithm...")
        phases_discovered = active_learning_traditional(
            data=data,
            X_std=X_std,
            y_encoded=y_encoded,
            algorithm=algorithm,
            n_initial_sampling=n_initial,
            max_iterations=max_iterations,
            batch_size=batch_size,
            initial_labeled_indices=labeled_indices,
            phases_nums=phases_nums
        )
        
        final_total = n_initial + (max_iterations * batch_size)
        remaining_indices = np.setdiff1d(np.arange(total_samples), labeled_indices)
        
        for i in range(max_iterations):
            if len(remaining_indices) < batch_size:
                break
            new_indices = np.random.choice(remaining_indices, size=batch_size, replace=False)
            all_sampled_indices = np.append(all_sampled_indices, new_indices)
            remaining_indices = np.setdiff1d(remaining_indices, new_indices)
            
            plot_sampling_evolution_demo(data, all_sampled_indices, i+1, algorithm, output_dir)
            phase_count = phases_discovered[i] if i < len(phases_discovered) else "N/A"
            print(f"   Round {i+1}: +{batch_size} points, total phases: {phase_count}")
    
    print(f"\nDEMONSTRATION COMPLETED!")
    print(f"Results:")
    print(f"  Final sampled points: {len(all_sampled_indices)}")
    if phases_discovered:
        print(f"  Final phases discovered: {phases_discovered[-1]}/{phases_nums}")
        print(f"  Discovery efficiency: {phases_discovered[-1]/phases_nums*100:.1f}%")
    print(f"  Files saved in: {output_dir}/")
    print("="*70)
