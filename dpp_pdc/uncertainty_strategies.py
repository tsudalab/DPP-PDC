"""
Uncertainty estimation strategies for active learning.
Implements various uncertainty scoring methods including PDC and Thompson Sampling.
"""

import numpy as np
from scipy.spatial.distance import cdist


def compute_pdc_scores(model, X_unlabeled):
    """
    PDC strategy implementation based on label propagation.
    
    Args:
        model: Label propagation model
        X_unlabeled: Unlabeled data points
        
    Returns:
        Normalized uncertainty scores
    """
    # Get prediction probabilities
    y_prob = model.predict_proba(X_unlabeled)
    
    # Use Least Confidence strategy to compute uncertainty scores
    u_score_list = 1 - np.max(y_prob, axis=1)
    
    # Ensure no all-zero probability distributions
    if np.sum(u_score_list) == 0 or np.all(u_score_list < 1e-10):
        return np.ones(len(u_score_list)) / len(u_score_list)
    
    # Ensure no zero probabilities (add small epsilon value)
    u_score_list = u_score_list + 1e-10
    
    # Normalize to probability distribution
    return u_score_list / np.sum(u_score_list)


def compute_ts_scores(model, X_unlabeled):
    """
    Thompson Sampling implementation based on label propagation.
    
    Args:
        model: Label propagation model
        X_unlabeled: Unlabeled data points
        
    Returns:
        Normalized uncertainty scores
    """
    y_prob = model.predict_proba(X_unlabeled)
    
    # Thompson Sampling
    n_samples = 10
    samples = []
    for _ in range(n_samples):
        weights = np.random.dirichlet([2] * y_prob.shape[1])
        weighted_probs = y_prob * weights
        samples.append(weighted_probs)
    
    samples = np.array(samples)
    uncertainty_scores = 1 - np.max(np.mean(samples, axis=0), axis=1)
    
    # Ensure no all-zero probability distributions
    if np.sum(uncertainty_scores) == 0 or np.all(uncertainty_scores < 1e-10):
        return np.ones(len(uncertainty_scores)) / len(uncertainty_scores)
    
    # Ensure no zero probabilities (add small epsilon value)
    uncertainty_scores = uncertainty_scores + 1e-10
    
    # Normalize to probability distribution
    return uncertainty_scores / np.sum(uncertainty_scores)


def compute_random_scores(X_unlabeled):
    """
    Random sampling strategy (baseline).
    
    Args:
        X_unlabeled: Unlabeled data points
        
    Returns:
        Uniform probability distribution
    """
    n_points = len(X_unlabeled)
    return np.ones(n_points) / n_points


def kmedoids_uncertainty_sampling(
    X_unlabeled,
    unlabeled_indices,
    uncertainty_scores,
    batch_size,
    top_percentile=0.3,
    random_state=0,
    strict_top_percentile=True,
):
    """
    Two-step baseline:
    (1) take top x% most uncertain points
    (2) run PAM-style k-medoids on the candidate set
    """
    unlabeled_indices = np.asarray(unlabeled_indices)
    uncertainty_scores = np.asarray(uncertainty_scores)

    assert X_unlabeled.shape[0] == len(unlabeled_indices) == len(uncertainty_scores), \
        "X_unlabeled, unlabeled_indices, and uncertainty_scores must be aligned."

    n_unlabeled = len(unlabeled_indices)
    if n_unlabeled == 0:
        return np.array([], dtype=int)

    # treat NaN as very low uncertainty
    scores = np.nan_to_num(uncertainty_scores, nan=-np.inf)

    if strict_top_percentile:
        n_candidates = max(batch_size, int(n_unlabeled * top_percentile))
    else:
        n_candidates = max(batch_size * 2, int(n_unlabeled * top_percentile))
    n_candidates = min(n_candidates, n_unlabeled)

    # top uncertain points
    top_local = np.argsort(scores)[-n_candidates:]
    X_candidates = X_unlabeled[top_local]

    if len(top_local) <= batch_size:
        selected_local = top_local
    else:
        medoid_in_candidates = _pam_kmedoids_fit(
            X_candidates,
            n_clusters=batch_size,
            max_iter=100,
            random_state=random_state,
        )
        selected_local = top_local[medoid_in_candidates]

    return unlabeled_indices[selected_local][:batch_size]


def _pam_kmedoids_fit(X, n_clusters, max_iter=100, random_state=0):
    """
    PAM-style k-medoids with random initialization + assignment/update iterations.
    """
    n_samples = X.shape[0]
    if n_samples <= n_clusters:
        return np.arange(n_samples)

    distances = cdist(X, X, metric="euclidean")

    rng = np.random.default_rng(random_state)
    medoid_indices = rng.choice(n_samples, size=n_clusters, replace=False)

    for _ in range(max_iter):
        labels = np.argmin(distances[:, medoid_indices], axis=1)

        new_medoids = medoid_indices.copy()
        changed = False

        for k in range(n_clusters):
            cluster_indices = np.where(labels == k)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_distances = distances[np.ix_(cluster_indices, cluster_indices)]
            best = cluster_indices[np.argmin(cluster_distances.sum(axis=1))]

            if best != medoid_indices[k]:
                changed = True
                new_medoids[k] = best

        medoid_indices = new_medoids
        if not changed:
            break

    return medoid_indices
