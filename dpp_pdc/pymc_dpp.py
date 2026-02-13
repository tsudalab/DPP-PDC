"""
Bayesian Determinantal Point Process (DPP) sampling module.
Implements PyMC-based Bayesian framework for DPP sampling in active learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from pytensor import tensor as pt
import pytensor.tensor.nlinalg as nlinalg


def bayesian_dpp_sampling(X_combined, unlabeled_indices, labeled_indices, us_scores, 
                         model, batch_size, algorithm, prior_mu, prior_sigma):
    """
    Bayesian DPP sampling using PyMC framework.
    
    Args:
        X_combined: Complete feature matrix
        unlabeled_indices: Unlabeled data indices
        labeled_indices: Labeled data indices
        us_scores: Uncertainty scores (PDC or TS scores) as likelihood
        model: Label propagation model
        batch_size: Batch size
        algorithm: Algorithm name
        prior_mu: Prior distribution mean
        prior_sigma: Prior distribution standard deviation
        
    Returns:
        chosen_indices: Selected sample indices
        optimal_noise_variance: Optimized noise variance
    """
    print("Performing Bayesian DPP sampling using PyMC...")
    
    # Initialize current batch (random selection)
    available_indices = unlabeled_indices

    # Create mapping for possible sample indices
    idx_mapping = {i: idx for i, idx in enumerate(available_indices)}
    
    with pm.Model() as dpp_model:
        # Define prior distribution for noise variance
        log_min = np.log(1e-5)  # Minimum log value
        log_max = np.log(10.0)   # Maximum log value
        log_noise_var = pm.TruncatedNormal(
            'log_noise_var', 
            mu=prior_mu, 
            sigma=prior_sigma,
            lower=log_min,
            upper=log_max
        )
        noise_variance = pm.Deterministic('noise_variance', pt.exp(log_noise_var))
        
        # Define prior distribution for sampling points (discrete distribution)
        sample_points = []
        for i in range(batch_size):
            # Use uncertainty scores as discrete distribution probabilities
            point_i = pm.Categorical(f'point_{i}', p=np.ones(len(available_indices))/len(available_indices))
            sample_points.append(point_i)
        
        # Define DPP prior distribution (using determinant as penalty term)
        def compute_dpp_score(points):
            # Use stack to combine PyTensor Symbol types
            idx_tensor = pt.stack(points)
            mapped_indices = pt.as_tensor_variable([idx_mapping[i] for i in range(len(available_indices))])
            actual_indices = mapped_indices[idx_tensor]

            # Extract corresponding data points
            X_all_tensor = pt.as_tensor_variable(X_combined)
            X_selected = X_all_tensor[actual_indices]

            # Compute kernel matrix
            sq_dist = pt.sum((X_selected[:, None, :] - X_selected[None, :, :]) ** 2, axis=2)
            gamma = 1.0 / X_selected.shape[1]
            K_selected = pt.exp(-gamma * sq_dist)

            dpp_matrix = pt.eye(X_selected.shape[0]) * (1 + 1e-6) + (1 / noise_variance) * K_selected
            det_score = nlinalg.det(dpp_matrix)

            return pt.log(det_score)
        
        # Define DPP prior
        dpp_prior = pm.Potential('dpp_prior', compute_dpp_score(sample_points))
        
        # Define uncertainty scores as likelihood
        def us_likelihood(points):
            idx_tensor = pt.stack(points)
            us_scores_tensor = pt.as_tensor_variable(us_scores)
            selected_scores = us_scores_tensor[idx_tensor]
            
            # Adjust likelihood intensity based on labeled data ratio
            labeled_ratio = len(labeled_indices) / len(X_combined)
            intensity = max(0.5, labeled_ratio)  # Increase influence with data accumulation
            
            # Apply intensity adjustment
            adjusted_scores = selected_scores ** intensity
            
            # Compute log likelihood
            return pt.sum(pt.log(adjusted_scores + 1e-10))
        
        # Add uncertainty scores as likelihood
        us_like = pm.Potential('us_likelihood', us_likelihood(sample_points))

        # MCMC sampling with modified parameters
        trace = pm.sample(
            1000, 
            tune=1000,  # Increase warm-up steps
            chains=1, 
            cores=1,
            target_accept=0.9,  # Acceptance rate target
            return_inferencedata=True
        )
    
    # Analyze sampling results
    print("MCMC sampling completed, analyzing results...")
    noise_var_samples = trace.posterior['noise_variance'].values.flatten()

    # Compute mean with minimum threshold
    raw_optimal_variance = float(np.mean(noise_var_samples))
    optimal_noise_variance = max(raw_optimal_variance, 1e-5)  # Ensure not smaller than 1e-5

    # Print warning if original value is too small
    if raw_optimal_variance < 1e-5:
        print(f"Warning: Raw optimal variance ({raw_optimal_variance:.2e}) too small, adjusted to {optimal_noise_variance:.2e}")
    else:
        print(f"Optimized noise variance: {optimal_noise_variance:.2e}")

    # Optional: Check noise variance distribution stability
    var_std = float(np.std(noise_var_samples))
    var_min = float(np.min(noise_var_samples))
    var_max = float(np.max(noise_var_samples))
    print(f"Noise variance distribution: min={var_min:.2e}, max={var_max:.2e}, std={var_std:.2e}")
    
    # Extract most frequent point combinations
    selected_points = []
    for i in range(batch_size):
        # Get posterior mode for each point position
        point_var = trace.posterior[f'point_{i}']
        point_values = point_var.values.flatten()
        most_common = np.bincount(point_values).argmax()
        selected_points.append(most_common)
    
    # Convert to actual indices
    chosen_indices = [idx_mapping[p] for p in selected_points]
    
    # Ensure no duplicate indices
    unique_indices = np.unique(chosen_indices)
    if len(unique_indices) < batch_size:
        remaining_needed = batch_size - len(unique_indices)
        remaining_indices = np.setdiff1d(available_indices, unique_indices)
        additional_indices = np.random.choice(remaining_indices, size=remaining_needed, replace=False)
        chosen_indices = np.concatenate([unique_indices, additional_indices])
    
    print(f"Selected {len(chosen_indices)} new sample points")

    return chosen_indices, optimal_noise_variance


def compute_dpp_kernel_matrix(X_selected, noise_variance):
    """
    Compute DPP kernel matrix for given points and noise variance.
    
    Args:
        X_selected: Selected data points
        noise_variance: Noise variance parameter
        
    Returns:
        DPP kernel matrix
    """
    # Compute squared distances
    sq_dist = np.sum((X_selected[:, None, :] - X_selected[None, :, :]) ** 2, axis=2)
    
    # Set gamma based on feature dimensions
    gamma = 1.0 / X_selected.shape[1]
    
    # Compute RBF kernel
    K_selected = np.exp(-gamma * sq_dist)
    
    # Compute DPP matrix
    dpp_matrix = np.eye(X_selected.shape[0]) * (1 + 1e-6) + (1 / noise_variance) * K_selected
    
    return dpp_matrix

