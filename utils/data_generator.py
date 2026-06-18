import numpy as np

def generate_ring_dataset(
    n_inner: int = 35,
    n_outer: int = 45,
    inner_radius_range: tuple[float, float] = (0.0, 1.0),
    outer_radius_range: tuple[float, float] = (1.6, 2.5),
    noise: float = 0.08,
    random_seed: int = 7,
):
    """
    Generate a 2D ring dataset for SVM kernel trick demonstration.
    
    Parameters:
    -----------
    n_inner : int
        Number of points in the inner circular class (label 0, blue).
    n_outer : int
        Number of points in the outer ring class (label 1, red).
    inner_radius_range : tuple of (float, float)
        Min and max radius bounds for the inner class.
    outer_radius_range : tuple of (float, float)
        Min and max radius bounds for the outer class.
    noise : float
        Standard deviation of Gaussian noise added to the points.
    random_seed : int
        Random seed for reproducibility.
        
    Returns:
    --------
    X : numpy.ndarray of shape (n_samples, 2)
        The 2D feature coordinates of the generated points.
    y : numpy.ndarray of shape (n_samples,)
        The binary labels (0 for inner, 1 for outer).
    """
    rng = np.random.default_rng(random_seed)
    
    # Generate inner points
    r_inner = rng.uniform(inner_radius_range[0], inner_radius_range[1], n_inner)
    theta_inner = rng.uniform(0, 2 * np.pi, n_inner)
    X_inner = np.column_stack((r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)))
    
    # Generate outer points
    r_outer = rng.uniform(outer_radius_range[0], outer_radius_range[1], n_outer)
    theta_outer = rng.uniform(0, 2 * np.pi, n_outer)
    X_outer = np.column_stack((r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)))
    
    # Add noise
    X_inner += rng.normal(0, noise, size=X_inner.shape)
    X_outer += rng.normal(0, noise, size=X_outer.shape)
    
    # Combine
    X = np.vstack((X_inner, X_outer))
    y = np.hstack((np.zeros(n_inner, dtype=int), np.ones(n_outer, dtype=int)))
    
    return X, y
