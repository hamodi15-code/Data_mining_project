import numpy as np
import csv

def generate_data(dim, k_clusters, n_points, out_path, block_size=1000000, std_dev=1.5):
    """
    Generates large synthetic data in blocks to handle memory efficiently.
    """
     # We initialize cluster centers using a uniform distribution to ensure randomness in data generation
    cluster_centers = np.random.uniform(-10, 10, size=(k_clusters, dim))
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
            
            remaining_points = current_block_size % k_clusters
            if remaining_points > 0:
                extra_points = np.random.normal(
                    loc=cluster_centers[np.random.choice(k_clusters, remaining_points)],
                    scale=std_dev,
                    size=(remaining_points, dim)
                )
                for i, point in enumerate(extra_points):
                    points.append(list(point) + [i % k_clusters])
            
            writer.writerows(points)
            n_points -= current_block_size
    
    return out_path


def bfr_cluster(dim, k_clusters, n_points, block_size, in_path, out_path):
    """
    Implements BFR clustering for large datasets by processing data in blocks.
    """
    initial_batch = []
    with open(in_path, mode='r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i >= block_size:
                break
            initial_batch.append([float(value) for value in row[:dim]])
    
    centroids = np.array(initial_batch[:k_clusters])
    clusters = [[] for _ in range(k_clusters)]
    
    with open(in_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            point = np.array([float(value) for value in row[:dim]])
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point.tolist())
    
    with open(out_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for cluster_idx, cluster in enumerate(clusters):
            for point in cluster:
                writer.writerow(point + [cluster_idx])
    
    return out_path


def cure_cluster(dim, k_clusters, n_points, block_size, in_path, out_path, num_representatives=5, shrink_factor=0.5):
    """
    Implements CURE clustering for large datasets by processing data in blocks.
    """
    initial_batch = []
    with open(in_path, mode='r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i >= block_size:
                break
            initial_batch.append([float(value) for value in row[:dim]])
    
    centroids = np.array(initial_batch[:k_clusters])
    clusters = {i: [] for i in range(k_clusters)}
    representatives = {i: [] for i in range(k_clusters)}
    
    with open(in_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            point = np.array([float(value) for value in row[:dim]])
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point.tolist())
    
    for cluster_idx, points in clusters.items():
        if len(points) > num_representatives:
            distances = np.linalg.norm(np.array(points)[:, None] - np.array(points), axis=2)
            farthest_points = np.argsort(distances.sum(axis=1))[-num_representatives:]
            representatives[cluster_idx] = [points[i] for i in farthest_points]
        else:
            representatives[cluster_idx] = points
        
        centroid = np.mean(points, axis=0)
        representatives[cluster_idx] = [centroid + shrink_factor * (rep - centroid) for rep in representatives[cluster_idx]]
    
    with open(out_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for cluster_idx, reps in representatives.items():
            for point in reps:
                writer.writerow(point + [cluster_idx])
    
    return out_path

def initialize_centroids(points, k):
    """Initialize centroids using K-Means++ method."""
    centroids = [points[np.random.randint(len(points))]]  # First centroid randomly
    for _ in range(1, k):
        distances = np.array([min(np.linalg.norm(p - c) for c in centroids) for p in points])
        probabilities = distances / distances.sum()
        centroids.append(points[np.random.choice(len(points), p=probabilities)])
    return np.array(centroids)
