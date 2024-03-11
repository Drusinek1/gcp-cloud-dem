import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os
import math


def stratified_sample(df, stratify_col='elevation_band', n_samples=1000):
    """
    Perform stratified sampling from the DataFrame.
    """
    return df.groupby(stratify_col, group_keys=False).apply(
        lambda x: x.sample(min(len(x), n_samples))
    )


def estimate_quadtree_partition_size(semivariogram_params, point_count):
    """
    Estimates an optimal quadtree partition size based on semivariogram range and point count.

    Parameters:
    - semivariogram_params: Dictionary of semivariogram parameters including 'range'.
    - point_count: Approximate number of points calculated for an area.

    Returns:
    - An estimated optimal partition size for quadtree.
    """
    # Extract the range from semivariogram parameters
    semivariogram_range = semivariogram_params['range']

    # Assuming points are distributed evenly, calculate the side length of an area
    # that contains 'point_count' number of points with 'semivariogram_range' as a diagonal.
    # This is an approximation to ensure partitions are larger than the semivariogram range.
    area_side_length = math.sqrt(point_count) * semivariogram_range

    return area_side_length


def calculate_semivariogram_parameters(band_df):
    """
    Calculate semivariogram parameters (nugget, sill, range) for a given elevation band.
    """
    X = band_df[['X', 'Y']].values
    Z = band_df['Z'].values
    distances = squareform(pdist(X))
    nugget = np.var(Z) * 0.1
    sill = np.var(Z)
    range_estimate = np.max(distances) * 0.5
    return {'nugget': nugget, 'sill': sill, 'range': range_estimate}


def process_parquet(file_path, num_bands=10, n_samples=100):
    df = pd.read_parquet(file_path)
    df['elevation_band'] = pd.qcut(df['Z'], q=num_bands, labels=range(num_bands))
    sampled_df = stratified_sample(df, 'elevation_band', n_samples=n_samples)

    parameters_by_band = {}
    for band in sampled_df['elevation_band'].unique():
        band_df = sampled_df[sampled_df['elevation_band'] == band]
        parameters_by_band[band] = calculate_semivariogram_parameters(band_df)

    return parameters_by_band


# Replace 'path_to_your_parquet_file.parquet' with your actual file path
file_path = os.environ.get['PARQUET_PATH']
num_bands = os.environ.get['BANDS']
n_samples = os.environ.get['N_SAMPLES']
semivariogram_parameters = process_parquet(file_path, num_bands=num_bands, n_samples=n_samples)

# semivariogram_parameters now contains the nugget, sill, and range for each elevation band


def calculate_point_count(M, N):
    """
    Calculates the approximate number of points needed to cover an M x N area.

    Parameters:
    - M, N: Dimensions of the area in meters.

    Returns:
    - Approximate number of points needed.
    """
    # Get the average point spacing from environment variable, convert to float
    point_spacing = float(os.environ.get('LIDAR_POINT_SPACING', '1'))  # Default to 1 meter if not set

    # Calculate the total area
    total_area = M * N

    # Calculate and return the number of points
    # Assuming a square grid layout for simplicity, adjust if the point pattern differs
    return math.ceil(total_area / (point_spacing ** 2))


def estimate_quadtree_partition_size_with_buffer(semivariogram_params, point_count):
    """
    Estimates an optimal quadtree partition size based on semivariogram range, point count,
    and includes a buffer equal to the semivariogram range to ensure all necessary points
    for kriging are within the partition.

    Parameters:
    - semivariogram_params: Dictionary of semivariogram parameters including 'range'.
    - point_count: Approximate number of points calculated for an area.

    Returns:
    - An estimated optimal partition size for quadtree, including spatial buffers.
    """
    semivariogram_range = semivariogram_params['range']

    # Calculate the side length of an area that needs to contain 'point_count' number of points,
    # then add a buffer equal to twice the semivariogram range (for each side of the partition)
    # to ensure all relevant points for kriging are included within a single partition.
    area_side_length = math.sqrt(point_count) * semivariogram_range

    # Including buffer: Add twice the range to account for buffer on both sides
    buffered_side_length = area_side_length + 2 * semivariogram_range

    return buffered_side_length



# Example dimensions and semivariogram parameters
M, N = 1000, 1000  # Example area dimensions in meters
semivariogram_params = {'nugget': 0.1, 'sill': 1.5, 'range': 150}  # Example semivariogram parameters

# Calculate the approximate number of points
point_count = calculate_point_count(M, N)

# Estimate the optimal quadtree partition size with buffer
optimal_partition_size_with_buffer = estimate_quadtree_partition_size_with_buffer(semivariogram_params, point_count)

print("Approximate number of points:", point_count)
print("Estimated optimal quadtree partition size with buffer:", optimal_partition_size_with_buffer)


