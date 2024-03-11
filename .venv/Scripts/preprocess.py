import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from skgstat import Variogram
from google.cloud import pubsub_v1
import json
import pylas
import os
import time
from kneed import KneeLocator
from google.api_core.exceptions import AlreadyExists


def analyze_pointcloud(parquet_file_path):
    # ----------- For local testing ONLY!------------------
    # Load the point cloud data from a Parquet file
    print("Reading Point Cloud parquet file")
    pointcloud_df = pq.read_table(parquet_file_path).to_pandas()

    # Determine dynamic sample size based on the number of pixels
    print("Choosing optimal sample size")

    total_pixels = len(pointcloud_df)
    max_sample_size = 1000  # Adjust as needed
    min_sample_size = 100   # Adjust as needed
    if total_pixels <= 10000:
        num_samples = min_sample_size
    elif total_pixels > 1000000:
        num_samples = max_sample_size
    else:
        num_samples = int(min_sample_size + (max_sample_size - min_sample_size) * (total_pixels - 10000) / (1000000 - 10000))
    print("Beginning stratifyed sampling")
    sampled_df = pointcloud_df.sample(n=min(1000, len(pointcloud_df)), random_state=0)
    k_range = range(1, 11)  # Example range from 1 to 10
    sse = elbow_method(sampled_df, k_range, features=['x', 'y'])
    optimal_k = find_optimal_clusters(sse, k_range)
    print(f"Optimal number of clusters: {optimal_k}")
    # Stratify Sample Spatially using KMeans
    print(f"KMeans clustering with {num_samples}")
    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(sampled_df[['x', 'y']])

    indices = [np.where(kmeans.labels_ == i)[0][0] for i in range(optimal_k)]
    print("Sampling points")
    sampled_points = pointcloud_df.iloc[indices]
    print("Calculating Semivariograms")

    # Calculate Semivariogram for the Sample
    coordinates = sampled_points[['x', 'y']].values
    values = sampled_points['z'].values  # Assuming 'z' is the value of interest
    V = Variogram(coordinates, values, model='spherical', normalize=False)
    print("Checking sill")
    sill = V.describe()['sill']

    print("Defining optimal window size")

    # Define window size based on sill
    window_size = sill * 2
    buffer_size = window_size * 0.1

    # Prepare results
    results = {
        "window_size": window_size,
        "buffer_size": buffer_size,
        "total_windows": num_samples,  # Assuming one window per sample cluster
        "sill": sill
    }

    return results

def analyze_and_publish_pointcloud(parquet_file_path, x_mesh, y_mesh, project_id, topic_id):
    # Initialize Pub/Sub publisher
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(os.environ("PROJECT_ID"), os.environ("TOPIC_ID"))

    # Load the point cloud data from a Parquet file
    pointcloud_df = pq.read_table(parquet_file_path).to_pandas()

    # Sample data if necessary
    sampled_df = pointcloud_df.sample(n=min(1000, len(pointcloud_df)), random_state=0)

    # Determine the optimal number of clusters
    k_range = range(1, 11)
    sse = elbow_method(sampled_df[['x', 'y']], k_range)
    optimal_k = find_optimal_clusters(sse, k_range)
    print(f"Optimal number of clusters: {optimal_k}")

    # Stratify Sample Spatially using KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(sampled_df[['x', 'y']])
    indices = [np.where(kmeans.labels_ == i)[0][0] for i in range(optimal_k)]

    # Select representative points
    sampled_points = pointcloud_df.iloc[indices]

    # Calculate Semivariogram for the sample
    coordinates = sampled_points[['x', 'y']].values
    values = sampled_points['z'].values
    V = Variogram(coordinates, values, model='spherical', normalize=False)

    sill = V.describe()['sill']

    # Define window and buffer size based on sill
    window_size = sill * 2
    buffer_size = window_size * 0.1

    # Prepare message with results
    message = {
        "window_size": window_size,
        "buffer_size": buffer_size,
        "total_windows": optimal_k,  # Use optimal_k for the number of clusters/windows
        "sill": sill
    }

    # Serialize message for publishing
    message_data = json.dumps(message).encode("utf-8")

    # Publish results to Pub/Sub
    future = publisher.publish(topic_path, message_data)
    future.result()  # Ensure the publish completes
    print(f"Published analysis results to {topic_path}")

def convert_las_to_parquet(input_las_path, output_parquet_path):
    # Read the LAS/LAZ file
    las = pylas.read(input_las_path)

    # Convert LAS data to a pandas DataFrame
    # Extracting relevant fields, add or remove fields as needed
    data = {
        'x': las.x,
        'y': las.y,
        'z': las.z,
        'intensity': las.intensity,
        'classification': las.classification,
        'return_number': las.return_number,
        'number_of_returns': las.number_of_returns,
        # Add any other LAS attributes you need
    }
    df = pd.DataFrame(data)

    # Write the DataFrame to a Parquet file
    df.to_parquet(output_parquet_path, index=False)

    print(f"Converted {input_las_path} to {output_parquet_path}")

from sklearn.cluster import KMeans

def elbow_method(data, k_range, features=['x', 'y']):
    """
    Apply the Elbow Method to determine the optimal number of clusters for KMeans clustering
    and return the sum of squared distances for each k without plotting.

    Parameters:
    - data: DataFrame containing the dataset to cluster.
    - k_range: Range of k values (number of clusters) to try.
    - features: List of column names to use for clustering.

    Returns:
    - sse: A list of sum of squared distances for each k in k_range.
    """
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data[features])
        sse.append(kmeans.inertia_)

    return sse


def find_optimal_clusters(sse, k_range):
    """
    Uses the kneedle algorithm to find the elbow point in the SSE curve,
    which suggests the optimal number of clusters.

    Parameters:
    - sse: A list of sum of squared distances for each k in k_range.
    - k_range: The range of k values that were tested.

    Returns:
    - The optimal number of clusters (k value) based on the elbow method.
    """
    # Convert k_range to a list if it's a range object
    k_list = list(k_range) if isinstance(k_range, range) else k_range

    # Initialize the KneeLocator to find the elbow point
    knee_locator = KneeLocator(k_list, sse, curve='convex', direction='decreasing')

    return knee_locator.elbow

def publish_message_to_pubsub(topic_path, message):
    publisher = pubsub_v1.PublisherClient()
    data = json.dumps(message).encode("utf-8")
    future = publisher.publish(topic_path, data)
    future.result()  # Ensure the publish completes
    print(f"Published message to {topic_path}")



if __name__ == "__main__":
    start_time = time.time()
    # CREATING TOPIC
    # Initialize a Publisher client
    publisher = pubsub_v1.PublisherClient()
    # Specify your Google Cloud Project ID
    project_id = os.environ.get("PROJECT_ID")
    topic_id = os.environ.get("TOPIC_ID")
    # Build the topic path
    topic_path = publisher.topic_path(project_id, topic_id)

    # Create the topic
    topic = publisher.create_topic(name=topic_path)

    print(f"Created topic: {topic.name}")
    # Switch working directory to project root ()
    os.chdir('C:/Users/drusi/PycharmProjects/Cloud-Krig/')
    # Specify the paths
    print(os.getcwd())
    input_laz_path = "data/USGS_LPC_TX_WestTexas_2018_D19_13REN585685.laz"
    output_parquet_path = "USGS_LPC_TX_WestTexas_2018_D19_13REN585685.parquet"

    # Step 1: Convert LAZ to Parquet (Assuming the function exists)
    print("Converting LAZ to Parquet...")
    convert_las_to_parquet(input_laz_path, output_parquet_path)

    # Step 2: Analyze the point cloud data from the Parquet file
    print("Analyzing point cloud data...")
    # Now call the function that analyzes and publishes the results
    analyze_and_publish_pointcloud(output_parquet_path, None, None, project_id, topic_id)

    # Record the end time and calculate the duration
    end_time = time.time()
    duration = end_time - start_time

    # Print the duration
    print(f"Total Time Taken: {duration:.2f} seconds")

