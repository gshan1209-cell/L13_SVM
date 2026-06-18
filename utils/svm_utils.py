import numpy as np
from sklearn.svm import SVC

def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "rbf",
    C: float = 10.0,
    gamma: float | str = 1.0,
    degree: int = 3
) -> SVC:
    """
    Train a scikit-learn Support Vector Classifier (SVC).
    
    Parameters:
    -----------
    X : numpy.ndarray of shape (n_samples, 2)
        Training vectors.
    y : numpy.ndarray of shape (n_samples,)
        Target values.
    kernel : str
        Specifies the kernel type to be used in the algorithm.
        Options: 'linear', 'poly', 'rbf', 'sigmoid'.
    C : float
        Regularization parameter.
    gamma : float or str
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    degree : int
        Degree of the polynomial kernel function ('poly').
        
    Returns:
    --------
    clf : sklearn.svm.SVC
        The trained SVM model.
    """
    # Create the model. If gamma is a float, check if it's positive.
    # We pass C, kernel, gamma, and degree. SVC handles ignoring irrelevant parameters.
    # Note: For 'linear' kernel, gamma is ignored, but passing it doesn't cause errors.
    
    # In some versions of sklearn, if gamma is not applicable, we can still pass it.
    # But just in case, we can clean up arguments or let SVC handle them.
    # Passing gamma as 'scale' or 'auto' works too.
    clf = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        random_state=42
    )
    clf.fit(X, y)
    return clf

def make_decision_grid(
    x_range: tuple[float, float] = (-3.0, 3.0),
    y_range: tuple[float, float] = (-3.0, 3.0),
    resolution: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a mesh grid for decision function evaluation.
    
    Parameters:
    -----------
    x_range : tuple of (float, float)
        Range of x values.
    y_range : tuple of (float, float)
        Range of y values.
    resolution : int
        Number of grid points along each axis.
        
    Returns:
    --------
    xx : numpy.ndarray of shape (resolution, resolution)
        X coordinates grid.
    yy : numpy.ndarray of shape (resolution, resolution)
        Y coordinates grid.
    grid_points : numpy.ndarray of shape (resolution * resolution, 2)
        Flattened 2D coordinate pairs.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    return xx, yy, grid_points

def compute_decision_surface(
    model: SVC,
    grid_points: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray
) -> np.ndarray:
    """
    Evaluate the model's decision function on the grid points and reshape.
    
    Parameters:
    -----------
    model : sklearn.svm.SVC
        Trained SVM model.
    grid_points : numpy.ndarray of shape (N, 2)
        Flattened grid points to predict.
    xx : numpy.ndarray
        Meshgrid X grid for shape reference.
    yy : numpy.ndarray
        Meshgrid Y grid for shape reference.
        
    Returns:
    --------
    Z : numpy.ndarray of shape xx.shape
        Decision values at each grid point.
    """
    # decision_function returns the distance to the separating hyperplane
    # for each point. For binary classification, shape is (n_samples,)
    Z = model.decision_function(grid_points)
    return Z.reshape(xx.shape)
