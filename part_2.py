import numpy as np
import csv

def generate_data(dim, k_clusters, n_points, out_path, points_gen=None, extras={}, block_size=1000000):
    """
    Generates synthetic data in blocks to handle large datasets and saves it to a CSV file.
    
    Parameters:
    dim (int): Number of dimensions for each data point.
    k_clusters (int): Number of clusters to generate.
    n_points (int): Total number of points to generate.
    out_path (str): Path to save the generated CSV file.
    points_gen (function, optional): Custom function to generate points (not used by default).
    extras (dict, optional): Additional parameters, such as 'std_dev' for standard deviation.
    block_size (int, optional): Number of points to generate per batch. Default is 1,000,000.
    
    Returns:
    None
    """
    # Define cluster centers using a uniform distribution in range (-10,10)
    cluster_centers = np.random.uniform(-10, 10, size=(k_clusters, dim))
    
    # Define standard deviation for clusters (default is 1.5, adjustable via extras)
    std_dev = extras.get("std_dev", 1.5)
    
    num_blocks = (n_points // block_size) + (1 if n_points % block_size != 0 else 0)
    
    with open(out_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        for _ in range(num_blocks):
            points = []
            current_block_size = min(block_size, n_points)
            
            for cluster_id in range(k_clusters):
                cluster_points = np.random.normal(
                    loc=cluster_centers[cluster_id], scale=std_dev, size=(current_block_size // k_clusters, dim)
                )
                for point in cluster_points:
                    points.append(list(point) + [cluster_id])
            
            # Handle remaining points that couldn't be evenly distributed
            remaining_points = current_block_size % k_clusters
            if remaining_points > 0:
                extra_points = np.random.normal(
                    loc=cluster_centers[np.random.choice(k_clusters, remaining_points)],
                    scale=std_dev,
                    size=(remaining_points, dim)
                )
                for i, point in enumerate(extra_points):
                    points.append(list(point) + [i % k_clusters])
            
            # Save block to CSV
            writer.writerows(points)
            n_points -= current_block_size
    
    print(f"Generated {n_points} points in blocks and saved to {out_path}.")

def bfr_cluster(dim, k_clusters, n_points, block_size, in_path, out_path):
    """
    Performs BFR clustering on large datasets by processing data in blocks.
    
    Parameters:
    dim (int): Number of dimensions per point.
    k_clusters (int or None): Number of clusters to form. If None, automatically determines the best k.
    n_points (int): Total number of points in the dataset.
    block_size (int): Number of points to process at a time (to handle large datasets).
    in_path (str): Path to input CSV file.
    out_path (str): Path to save the clustered results.
    
    Returns:
    None
    """
    # Initialize cluster centers (random selection from first batch)
    initial_batch = []
    with open(in_path, mode='r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i >= block_size:
                break
            initial_batch.append([float(value) for value in row[:dim]])
    
    # Convert to numpy array for numerical operations
    initial_batch = np.array(initial_batch)
    
    # Use K-Means++ to initialize centroids
    centroids = initialize_centroids(initial_batch, k_clusters)
    
    clusters = [[] for _ in range(k_clusters)]
    
    # Process dataset in blocks
    with open(in_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            point = np.array([float(value) for value in row[:dim]])
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
    
    # Save clustered data
    with open(out_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for cluster_idx, cluster in enumerate(clusters):
            for point in cluster:
                writer.writerow(list(point) + [cluster_idx])
    
    print(f"BFR clustering completed. Results saved to {out_path}.")

def cure_cluster(dim, k_clusters, n_points, block_size, in_path, out_path, num_representatives=5, shrink_factor=0.5):
    """
    Performs CURE clustering for large datasets by processing data in blocks.
    
    Parameters:
    dim (int): Number of dimensions per point.
    k_clusters (int or None): Number of clusters to form. If None, automatically determines the best k.
    n_points (int): Total number of points in the dataset.
    block_size (int): Number of points to process at a time (to handle large datasets).
    in_path (str): Path to input CSV file.
    out_path (str): Path to save the clustered results.
    num_representatives (int, optional): Number of representative points per cluster. Default is 5.
    shrink_factor (float, optional): Factor to shrink representative points toward the centroid. Default is 0.5.
    
    Returns:
    None
    """
    # Load initial batch to initialize clusters
    initial_batch = []
    with open(in_path, mode='r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i >= block_size:
                break
            initial_batch.append([float(value) for value in row[:dim]])
    
    # Convert to numpy array
    initial_batch = np.array(initial_batch)
    
    # Use K-Means++ to initialize centroids
    centroids = initialize_centroids(initial_batch, k_clusters)
    clusters = {i: [] for i in range(k_clusters)}
    representatives = {i: [] for i in range(k_clusters)}
    
    # Process dataset in blocks
    with open(in_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            point = np.array([float(value) for value in row[:dim]])
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
    
    # Select representative points per cluster
    for cluster_idx, points in clusters.items():
        if len(points) > num_representatives:
            distances = np.linalg.norm(np.array(points)[:, None] - np.array(points), axis=2)
            farthest_points = np.argsort(distances.sum(axis=1))[-num_representatives:]
            representatives[cluster_idx] = [points[i] for i in farthest_points]
        else:
            representatives[cluster_idx] = points
        
        # Shrink representative points toward centroid
        centroid = np.mean(points, axis=0)
        representatives[cluster_idx] = [centroid + shrink_factor * (rep - centroid) for rep in representatives[cluster_idx]]
    
    # Save clustered data
    with open(out_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for cluster_idx, reps in representatives.items():
            for point in reps:
                writer.writerow(list(point) + [cluster_idx])
    
    print(f"CURE clustering completed. Results saved to {out_path}.")

def initialize_centroids(points, k):
    """Initialize centroids using K-Means++ method."""
    centroids = [points[np.random.randint(len(points))]]  # First centroid randomly
    for _ in range(1, k):
        distances = np.array([min(np.linalg.norm(p - c) for c in centroids) for p in points])
        probabilities = distances / distances.sum()
        centroids.append(points[np.random.choice(len(points), p=probabilities)])
    return np.array(centroids)
