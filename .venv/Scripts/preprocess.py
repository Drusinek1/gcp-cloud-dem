import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from skgstat import Variogram
from google.cloud import pubsub_v1
import json

def publish_message_to_pubsub(topic_path, message):
    publisher = pubsub_v1.PublisherClient()
    data = json.dumps(message).encode("utf-8")
    future = publisher.publish(topic_path, data)
    future.result()  # Ensure the publish completes
    print(f"Published message to {topic_path}")

def analyze_and_publish_pointcloud(parquet_file_path, x_mesh, y_mesh, project_id, topic_id):
    # Configure Pub/Sub topic path
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)

    # Load the point cloud data from a Parquet file
    pointcloud_df = pq.read_table(parquet_file_path).to_pandas()

    # Determine dynamic sample size based on the number of pixels
    total_pixels = len(pointcloud_df)
    max_sample_size = 1000  # Adjust as needed
    min_sample_size = 100   # Adjust as needed
    if total_pixels <= 10000:
        num_samples = min_sample_size
    elif total_pixels > 1000000:
        num_samples = max_sample_size
    else:
        num_samples = int(min_sample_size + (max_sample_size - min_sample_size) * (total_pixels - 10000) / (1000000 - 10000))

    # Stratify Sample Spatially
    kmeans = KMeans(n_clusters=num_samples, random_state=0).fit(pointcloud_df[['x', 'y']])
    indices = [np.where(kmeans.labels_ == i)[0][0] for i in range(num_samples)]
    sampled_points = pointcloud_df.iloc[indices]

    # Calculate Semivariogram for the Sample
    coordinates = sampled_points[['x', 'y']].values
    values = sampled_points['z'].values  # Assuming 'z' is the value of interest
    V = Variogram(coordinates, values, model='spherical', normalize=False)
    sill = V.describe()['sill']

    # Define window size based on sill
    window_size = sill * 2
    buffer_size = window_size * 0.1

    # Prepare message with results
    message = {
        "window_size": window_size,
        "buffer_size": buffer_size,
        "total_windows": num_samples,  # Assuming one window per sample cluster
        "sill": sill
    }

    # Publish results to Pub/Sub
    publish_message_to_pubsub(topic_path, message)

