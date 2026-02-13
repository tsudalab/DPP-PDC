"""
Active learning algorithms module.
Implements both traditional and Bayesian DPP-based active learning methods.
"""

import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score
from sklearn.preprocessing import label_binarize

from dpp_pdc.uncertainty_strategies import (
    compute_pdc_scores,
    compute_ts_scores,
    compute_random_scores,
    kmedoids_uncertainty_sampling,
)
from dpp_pdc.pymc_dpp import bayesian_dpp_sampling


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Compute classification metrics for phase diagram evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional, for mAP)
        
    Returns:
        Dictionary containing accuracy, macro_precision, macro_recall, macro_f1, mAP
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Mean Average Precision (mAP)
    mAP = None
    if y_prob is not None:
        try:
            # Get unique classes
            classes = np.unique(y_true)
            if len(classes) > 1:
                # Binarize labels for multi-class mAP
                y_true_bin = label_binarize(y_true, classes=classes)
                
                # Handle binary case
                if y_true_bin.shape[1] == 1:
                    y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
                
                # Ensure y_prob has correct shape
                if y_prob.shape[1] >= len(classes):
                    # Select only columns corresponding to classes present in y_true
                    y_prob_subset = y_prob[:, :len(classes)]
                    mAP = average_precision_score(y_true_bin, y_prob_subset, average='macro')
        except Exception as e:
            # If mAP computation fails, leave as None
            pass
    
    return {
        'accuracy': accuracy,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'mAP': mAP
    }


def active_learning_with_pymc(data, X_std, y_encoded, 
                            algorithm,
                            n_initial_sampling, 
                            max_iterations, 
                            batch_size, 
                            initial_labeled_indices, 
                            phases_nums):
    """
    Active learning implementation using PyMC Bayesian framework.
    
    Args:
        data: Original DataFrame
        X_std: Standardized feature matrix
        y_encoded: Encoded labels
        algorithm: Algorithm to use ('DPP-PDC', 'DPP-TS')
        n_initial_sampling: Number of initial samples
        max_iterations: Maximum number of iterations
        batch_size: Number of samples to select per iteration
        initial_labeled_indices: Initial labeled point indices
        phases_nums: Total number of phases
        
    Returns:
        phases_discovered_all: List of discovered phases count per iteration
        noise_variance_history: History of noise variance values
        selected_points_history: History of selected points
        metrics_history: History of classification metrics per iteration
    """
    total_samples = X_std.shape[0]
    labeled_indices = initial_labeled_indices.copy()
    
    unlabeled_indices = np.setdiff1d(np.arange(total_samples), labeled_indices)
    y_combined = np.full(total_samples, -1)
    y_combined[labeled_indices] = y_encoded[labeled_indices]
    X_combined = X_std

    # Initialize label propagation model
    model = LabelPropagation()
    
    phases_discovered_all = []
    noise_variance_history = []
    selected_points_history = []
    metrics_history = []

    # Record initial sampling points
    initial_points = X_combined[labeled_indices]
    selected_points_history.append(initial_points)  # Round 0: initial sampling
    initial_phases_discovered = len(np.unique(y_combined[y_combined != -1]))
    phases_discovered_all.append(initial_phases_discovered)
    print(f'Initial phases discovered: {initial_phases_discovered}')

    # Initial metrics (before any active learning iteration)
    model.fit(X_combined[labeled_indices], y_combined[labeled_indices])
    y_pred_all = model.predict(X_combined)
    y_prob_all = model.predict_proba(X_combined)
    initial_metrics = compute_classification_metrics(y_encoded, y_pred_all, y_prob_all)
    metrics_history.append(initial_metrics)
    print(f'Initial metrics - Accuracy: {initial_metrics["accuracy"]:.4f}, F1: {initial_metrics["macro_f1"]:.4f}')

    prior_mu = -4  # Initial prior mean
    prior_sigma = 4  # Initial prior standard deviation

    for iteration in range(max_iterations):
        if len(unlabeled_indices) < batch_size:
            print(f"Stopping: insufficient unlabeled points ({len(unlabeled_indices)} < {batch_size})")
            phases_discovered_all.append(len(np.unique(y_combined[y_combined != -1])))
            # Append last metrics
            metrics_history.append(metrics_history[-1] if metrics_history else None)
            break

        print(f"\nStarting iteration: {iteration + 1}, labeled points: {len(labeled_indices)}")

        # Train model
        model.fit(X_combined[labeled_indices], y_combined[labeled_indices])
        
        # Compute uncertainty scores based on algorithm
        if algorithm == 'DPP-PDC':
            us_scores = compute_pdc_scores(model, X_combined[unlabeled_indices])
        else:  # DPP-TS
            us_scores = compute_ts_scores(model, X_combined[unlabeled_indices])
        
        chosen_indices, optimal_noise_variance = bayesian_dpp_sampling(
            X_combined, 
            unlabeled_indices, 
            labeled_indices,
            us_scores,
            model, 
            batch_size,
            algorithm,
            prior_mu,
            prior_sigma
        )
        noise_variance_history.append(optimal_noise_variance)
        selected_points_history.append(X_combined[chosen_indices])
        prior_mu = np.log(optimal_noise_variance)
        
        # Update indices and labels
        labeled_indices = np.append(labeled_indices, chosen_indices)
        unlabeled_indices = np.setdiff1d(unlabeled_indices, chosen_indices)
        y_combined[chosen_indices] = y_encoded[chosen_indices]

        # Calculate discovered phases count
        phases_discovered = len(np.unique(y_combined[y_combined != -1]))
        print(f'Phases discovered after iteration {iteration + 1}: {phases_discovered}')
        phases_discovered_all.append(phases_discovered)
        
        # Compute classification metrics
        # Re-fit model with updated labels to get predictions for all points
        model.fit(X_combined[labeled_indices], y_combined[labeled_indices])
        y_pred_all = model.predict(X_combined)
        y_prob_all = model.predict_proba(X_combined)
        iter_metrics = compute_classification_metrics(y_encoded, y_pred_all, y_prob_all)
        metrics_history.append(iter_metrics)
        print(f'Metrics - Accuracy: {iter_metrics["accuracy"]:.4f}, F1: {iter_metrics["macro_f1"]:.4f}')
    
    return phases_discovered_all, noise_variance_history, selected_points_history, metrics_history


def active_learning_traditional(data, X_std, y_encoded, 
                              algorithm,
                              n_initial_sampling, 
                              max_iterations, 
                              batch_size, 
                              initial_labeled_indices, 
                              phases_nums,
                              kmedoids_variant='PAM',
                              kmedoids_top_percentile=0.025):
    """
    Traditional active learning method implementation (RS, PDC, K-Medoids, etc.).
    
    Args:
        data: Original DataFrame
        X_std: Standardized feature matrix
        y_encoded: Encoded labels
        algorithm: Algorithm to use ('RS', 'PDC', 'TS', 'K-Medoids')
        n_initial_sampling: Number of initial samples
        max_iterations: Maximum iterations
        batch_size: Samples per iteration
        initial_labeled_indices: Initial labeled point indices
        phases_nums: Total phases
        kmedoids_variant: K-Medoids variant ('FPS' or 'PAM'), default 'PAM'
        kmedoids_top_percentile: Top percentile for K-Medoids candidate selection, default 0.025
        
    Returns:
        phases_discovered_all: List of discovered phases per iteration
        metrics_history: History of classification metrics per iteration
    """
    total_samples = X_std.shape[0]
    labeled_indices = initial_labeled_indices.copy()
    
    unlabeled_indices = np.setdiff1d(np.arange(total_samples), labeled_indices)
    y_combined = np.full(total_samples, -1)
    y_combined[labeled_indices] = y_encoded[labeled_indices]
    X_combined = X_std

    # Initialize label propagation model
    model = LabelPropagation()
    
    phases_discovered_all = []
    metrics_history = []
                                
    initial_phases_discovered = len(np.unique(y_combined[y_combined != -1]))
    phases_discovered_all.append(initial_phases_discovered)
    print(f'Initial phases discovered: {initial_phases_discovered}')
    
    # Initial metrics
    model.fit(X_combined[labeled_indices], y_combined[labeled_indices])
    y_pred_all = model.predict(X_combined)
    y_prob_all = model.predict_proba(X_combined)
    initial_metrics = compute_classification_metrics(y_encoded, y_pred_all, y_prob_all)
    metrics_history.append(initial_metrics)
    print(f'Initial metrics - Accuracy: {initial_metrics["accuracy"]:.4f}, F1: {initial_metrics["macro_f1"]:.4f}')

    for iteration in range(max_iterations):
        if len(unlabeled_indices) < batch_size:
            print(f"Stopping: insufficient unlabeled points ({len(unlabeled_indices)} < {batch_size})")
            phases_discovered_all.append(len(np.unique(y_combined[y_combined != -1])))
            metrics_history.append(metrics_history[-1] if metrics_history else None)
            break

        print(f"\nStarting iteration: {iteration + 1}, labeled points: {len(labeled_indices)}")

        # Train model
        model.fit(X_combined[labeled_indices], y_combined[labeled_indices])
        
        # Select new sample points
        if algorithm == 'RS':
            # Random sampling
            chosen_indices = np.random.choice(unlabeled_indices, size=batch_size, replace=False)
        elif algorithm == 'K-Medoids':
            # K-medoids sampling within top uncertain points
            # Step 1: Compute uncertainty scores using PDC
            uncertainty_scores = compute_pdc_scores(model, X_combined[unlabeled_indices])
            # Step 2: Apply k-medoids variant on top uncertain points
            chosen_indices = kmedoids_uncertainty_sampling(
                X_unlabeled=X_combined[unlabeled_indices],
                unlabeled_indices=unlabeled_indices,
                uncertainty_scores=uncertainty_scores,
                batch_size=batch_size,
                top_percentile=kmedoids_top_percentile,
                variant=kmedoids_variant,
                random_state=iteration
            )
        elif algorithm in ['PDC', 'TS']:
            # Compute strategy scores based on algorithm
            if algorithm == 'PDC':
                strategy_scores = compute_pdc_scores(model, X_combined[unlabeled_indices])
            else:  # TS
                strategy_scores = compute_ts_scores(model, X_combined[unlabeled_indices])
            
            # Ensure sufficient non-zero probabilities
            if np.count_nonzero(strategy_scores) < batch_size:
                print(f"Warning: non-zero probability count ({np.count_nonzero(strategy_scores)}) < batch size ({batch_size}), adding smoothing")
                # Add smoothing to ensure all probabilities are non-zero
                strategy_scores = strategy_scores + 1e-6
                strategy_scores = strategy_scores / np.sum(strategy_scores)
            
            try:
                chosen_indices = np.random.choice(
                    unlabeled_indices, 
                    size=batch_size, 
                    replace=False, 
                    p=strategy_scores
                )
            except ValueError as e:
                print(f"Sampling error: {e}")
                print(f"Strategy score stats: min={np.min(strategy_scores)}, max={np.max(strategy_scores)}, non-zero count={np.count_nonzero(strategy_scores)}")
                print("Falling back to random sampling")
                chosen_indices = np.random.choice(unlabeled_indices, size=batch_size, replace=False)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Update indices and labels
        labeled_indices = np.append(labeled_indices, chosen_indices)
        unlabeled_indices = np.setdiff1d(unlabeled_indices, chosen_indices)
        y_combined[chosen_indices] = y_encoded[chosen_indices]

        # Calculate discovered phases count
        phases_discovered = len(np.unique(y_combined[y_combined != -1]))
        print(f'Phases discovered after iteration {iteration + 1}: {phases_discovered}')
        phases_discovered_all.append(phases_discovered)
        
        # Compute classification metrics
        model.fit(X_combined[labeled_indices], y_combined[labeled_indices])
        y_pred_all = model.predict(X_combined)
        y_prob_all = model.predict_proba(X_combined)
        iter_metrics = compute_classification_metrics(y_encoded, y_pred_all, y_prob_all)
        metrics_history.append(iter_metrics)
        print(f'Metrics - Accuracy: {iter_metrics["accuracy"]:.4f}, F1: {iter_metrics["macro_f1"]:.4f}')
    
    return phases_discovered_all, metrics_history


def evaluate_phase_discovery(y_combined):
    """
    Evaluate the number of discovered phases.
    
    Args:
        y_combined: Combined labels array with -1 for unlabeled
        
    Returns:
        Number of discovered phases
    """
    labeled_mask = y_combined != -1
    if np.any(labeled_mask):
        return len(np.unique(y_combined[labeled_mask]))
    return 0
