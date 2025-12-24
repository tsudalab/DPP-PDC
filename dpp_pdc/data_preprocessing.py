"""
Data preprocessing module for active learning phase discovery.
Handles data loading, standardization, and basic preprocessing operations.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def standardize_data(file_path):
    """
    Load CSV file, extract features and labels, standardize features.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        scaler: StandardScaler object
        X_std: Standardized features
        y_encoded: Encoded labels
        data: Original DataFrame
    """
    data = pd.read_csv(file_path)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return scaler, X_std, y_encoded, data


def compute_rbf_kernel(X1, X2=None):
    """
    Compute RBF kernel matrix.
    
    Args:
        X1: First set of data points
        X2: Second set of data points (optional)
        
    Returns:
        K_t: RBF kernel matrix
    """
    from scipy.spatial.distance import cdist
    
    if X2 is None:
        X2 = X1
    
    # Compute squared Euclidean distances
    sq_dist = cdist(X1, X2, metric='sqeuclidean')
    
    # Set gamma based on feature dimensions
    gamma = 1.0 / X1.shape[1]
    
    # Compute RBF kernel matrix
    K_t = np.exp(-gamma * sq_dist)
    
    return K_t


def prepare_initial_samples(X_std, y_encoded, batch_sizes, n_runs, n_initial_sampling):
    """
    Pre-generate initial sampling points for all batch sizes and runs.
    
    Args:
        X_std: Standardized features
        y_encoded: Encoded labels
        batch_sizes: List of batch sizes
        n_runs: Number of runs per configuration
        n_initial_sampling: Number of initial samples
        
    Returns:
        initial_samples: Dictionary mapping (batch_size, run) to initial indices
    """
    total_samples = X_std.shape[0]
    initial_samples = {}
    
    for batch_size in batch_sizes:
        for run in range(n_runs):
            np.random.seed(42+run)
            
            # Initial sampling until at least two phases are discovered
            labeled_indices = np.random.choice(total_samples, size=n_initial_sampling, replace=False)
            discovered_phases = np.unique(y_encoded[labeled_indices])
            
            while len(discovered_phases) < 2:
                remaining_indices = np.setdiff1d(np.arange(total_samples), labeled_indices)
                new_samples = np.random.choice(remaining_indices, size=batch_size, replace=False)
                labeled_indices = np.append(labeled_indices, new_samples)
                discovered_phases = np.unique(y_encoded[labeled_indices])
            
            initial_samples[(batch_size, run)] = labeled_indices
    
    return initial_samples
