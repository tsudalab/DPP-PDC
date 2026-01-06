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


# =============================================================================
# K-Medoids Variants
# =============================================================================

def kmedoids_uncertainty_sampling(
    X_unlabeled,
    unlabeled_indices,
    uncertainty_scores,
    batch_size,
    top_percentile=0.025,
    variant='PAM',
    random_state=0,
):
    """
    K-Medoids based uncertainty sampling with multiple variants.
    
    Args:
        X_unlabeled: Unlabeled data features
        unlabeled_indices: Global indices of unlabeled points
        uncertainty_scores: PDC uncertainty scores
        batch_size: Number of points to select
        top_percentile: Fraction of top uncertain points to consider (0.0-1.0)
        variant: 'FPS' (farthest point sampling) or 'PAM' (original k-medoids)
        random_state: Random seed for reproducibility
        
    Returns:
        Selected global indices
    """
    if variant == 'FPS':
        return _kmedoids_fps(
            X_unlabeled, unlabeled_indices, uncertainty_scores,
            batch_size, top_percentile, random_state
        )
    elif variant == 'PAM':
        return _kmedoids_pam(
            X_unlabeled, unlabeled_indices, uncertainty_scores,
            batch_size, top_percentile, random_state
        )
    else:
        raise ValueError(f"Unknown K-Medoids variant: {variant}. Use 'PAM' or 'FPS'.")


def _kmedoids_fps(
    X_unlabeled,
    unlabeled_indices,
    uncertainty_scores,
    batch_size,
    top_percentile=0.025,
    random_state=0,
):
    """
    Farthest Point Sampling within top uncertain candidates.
    
    Algorithm:
        1. Select top uncertain points as candidates
        2. Start with most uncertain point
        3. Iteratively add point farthest from current selection (max-min diversity)
    
    Reference:
        Gonzalez, T. F. (1985). Clustering to minimize the maximum intercluster distance.
        Similar to Core-Set approach: Sener & Savarese (2018), ICLR.
    """
    unlabeled_indices = np.asarray(unlabeled_indices)
    uncertainty_scores = np.asarray(uncertainty_scores)

    n_unlabeled = len(unlabeled_indices)
    if n_unlabeled == 0:
        return np.array([], dtype=int)

    scores = np.nan_to_num(uncertainty_scores, nan=-np.inf)

    # Select top uncertain candidates
    n_candidates = max(batch_size, int(n_unlabeled * top_percentile))
    n_candidates = min(n_candidates, n_unlabeled)

    top_local = np.argsort(scores)[-n_candidates:]
    X_candidates = X_unlabeled[top_local]
    scores_candidates = scores[top_local]

    if len(top_local) <= batch_size:
        return unlabeled_indices[top_local][:batch_size]

    # Compute pairwise distances
    distances = cdist(X_candidates, X_candidates, metric="euclidean")
    
    # Start with most uncertain point
    selected = [np.argmax(scores_candidates)]
    
    # Greedy farthest point selection
    for _ in range(batch_size - 1):
        # Compute min distance to current selection for each candidate
        min_dists = distances[:, selected].min(axis=1)
        # Exclude already selected
        min_dists[selected] = -np.inf
        # Select farthest point
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)
    
    selected_local = top_local[selected]
    return unlabeled_indices[selected_local][:batch_size]


def _kmedoids_pam(
    X_unlabeled,
    unlabeled_indices,
    uncertainty_scores,
    batch_size,
    top_percentile=0.025,
    random_state=0,
):
    """
    Original PAM-style K-Medoids on top uncertain candidates.
    
    Algorithm:
        1. Select top uncertain points as candidates
        2. Run PAM k-medoids clustering
        3. Return cluster medoids as selected points
    """
    unlabeled_indices = np.asarray(unlabeled_indices)
    uncertainty_scores = np.asarray(uncertainty_scores)

    n_unlabeled = len(unlabeled_indices)
    if n_unlabeled == 0:
        return np.array([], dtype=int)

    scores = np.nan_to_num(uncertainty_scores, nan=-np.inf)

    n_candidates = max(batch_size, int(n_unlabeled * top_percentile))
    n_candidates = min(n_candidates, n_unlabeled)

    top_local = np.argsort(scores)[-n_candidates:]
    X_candidates = X_unlabeled[top_local]

    if len(top_local) <= batch_size:
        return unlabeled_indices[top_local][:batch_size]

    medoid_indices = _pam_kmedoids_fit(
        X_candidates,
        n_clusters=batch_size,
        max_iter=100,
        random_state=random_state,
    )
    
    selected_local = top_local[medoid_indices]
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
