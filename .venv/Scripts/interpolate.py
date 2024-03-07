import numpy as np
from scipy.spatial.distance import pdist, squareform

def spherical_model(distances, range, sill):
    """Spherical semivariance model."""
    # For distances within the range, apply the spherical model formula
    semivariances = np.where(distances <= range,
                             sill * (1.5 * (distances / range) - 0.5 * (distances / range) ** 3),
                             sill)  # For distances beyond the range, the semivariance is equal to the sill
    return semivariances

def calculate_semivariance_matrix(points, range, sill):
    """Calculate the semivariance matrix for known points using a spherical model."""
    distances = squareform(pdist(points[:, :2]))  # Calculate pairwise distances between points
    semivariance_matrix = spherical_model(distances, range, sill)  # Apply the spherical model
    return semivariance_matrix

