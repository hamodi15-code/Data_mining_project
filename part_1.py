import numpy as np
import csv
from scipy.spatial.distance import euclidean

def generate_data(dim, k_clusters, n_points, out_path, points_gen=None, extras={}):
    """
    Generates synthetic data with normally distributed clusters and saves it to a CSV file.
    
    Parameters:
    dim (int): Number of dimensions for each data point.
    k_clusters (int): Number of clusters to generate.
    n_points (int): Total number of points to generate.
    out_path (str): Path to save the generated CSV file.
    points_gen (function, optional): Custom function to generate points (not used by default).
    extras (dict, optional): Additional parameters, such as 'std_dev' for standard deviation.
    
    Returns:
    list: A list of generated points with cluster labels.
    """
    # Define cluster centers using a uniform distribution in range (-10,10)
    cluster_centers = np.random.uniform(-10, 10, size=(k_clusters, dim))
    
    # Define standard deviation for clusters (default is 1.5, adjustable via extras)
    std_dev = extras.get("std_dev", 1.5)
    
    points = []
    
    # Generate points around each cluster center using a normal distribution
    for cluster_id in range(k_clusters):
        cluster_points = np.random.normal(loc=cluster_centers[cluster_id], 
                                          scale=std_dev, 
                                          size=(n_points // k_clusters, dim))
        for point in cluster_points:
            points.append(list(point) + [cluster_id])  # Append cluster ID to each point
    
    # Handle remaining points that don't divide evenly among clusters
    remaining_points = n_points % k_clusters
    if remaining_points > 0:
        extra_points = np.random.normal(loc=cluster_centers[np.random.choice(k_clusters, remaining_points)], 
                                        scale=std_dev, 
                                        size=(remaining_points, dim))
        for i, point in enumerate(extra_points):
            points.append(list(point) + [i % k_clusters])
    
    # Write the generated data to a CSV file
    with open(out_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for point in points:
            writer.writerow(point)
    
    return points  # Return the list of generated points

def load_data(in_path, dim, n_points=-1, points=[]):
    """
    Loads data points from a CSV file.
    
    Parameters:
    in_path (str): Path to the CSV file.
    dim (int): Number of dimensions per point.
    n_points (int, optional): Number of points to load (-1 to load all). Default is -1.
    points (list, optional): List to store the loaded points.
    
    Returns:
    list: A list of loaded points.
    """
    with open(in_path, mode='r') as file:
        reader = csv.reader(file)
        count = 0
        
        for row in reader:
            if len(row) < dim:  # Skip rows with insufficient data
                continue
            
            try:
                point = [float(value) for value in row[:dim]]  # Convert first 'dim' values to float
                points.append(point)
                count += 1
                
                if 0 < n_points == count:
                    break  # Stop reading if reached required number of points
            except ValueError:
                continue  # Skip rows that contain invalid values
    
    return points

def h_clustering(dim, k_clusters, points, dist=euclidean, clusts=[]):
    """
    Performs hierarchical bottom-up clustering using Euclidean distance.
    
    Parameters:
    dim (int): Number of dimensions per point.
    k_clusters (int or None): Number of clusters to form. If None, uses distance threshold.
    points (list): List of data points.
    dist (function, optional): Distance function (default: Euclidean distance).
    clusts (list, optional): List to store the resulting clusters.
    
    Returns:
    list: A list of clusters, where each cluster is a list of points.
    """
    # Initialize each point as its own cluster
    clusters = [[point] for point in points]
    
    # Define distance threshold if k_clusters is None
    distance_threshold = 5.0  # Can be adjusted
    
    while len(clusters) > (k_clusters if k_clusters else 1):
        min_distance = float("inf")
        merge_pair = (-1, -1)
        
        # Find the closest pair of clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = dist(np.mean(clusters[i], axis=0), np.mean(clusters[j], axis=0))
                if d < min_distance:
                    min_distance = d
                    merge_pair = (i, j)
        
        # Stop merging if using a threshold and the closest clusters are too far apart
        if k_clusters is None and min_distance > distance_threshold:
            break
        
        # Merge the two closest clusters
        i, j = merge_pair
        clusters[i].extend(clusters[j])
        del clusters[j]
    
    return clusters

def k_means(dim, k_clusters, n_points, points, clusts=[]):
    """
    Performs K-Means clustering with optimized centroid initialization (K-Means++).
    
    Parameters:
    dim (int): Number of dimensions per point.
    k_clusters (int or None): Number of clusters to form. If None, multiple runs are performed to find the best k.
    n_points (int): Number of data points.
    points (list): List of data points.
    clusts (list, optional): List to store the resulting clusters.
    
    Returns:
    list: A list of clusters, where each cluster is a list of points.
    """
    def initialize_centroids(points, k):
        """Initialize centroids using K-Means++ method."""
        centroids = [points[np.random.randint(len(points))]]  # First centroid randomly
        for _ in range(1, k):
            distances = np.array([min(np.linalg.norm(p - c) for c in centroids) for p in points])
            probabilities = distances / distances.sum()
            centroids.append(points[np.random.choice(len(points), p=probabilities)])
        return np.array(centroids)
    def run_k_means(k, points, max_iters=100, tolerance=1e-4):
        """Runs the K-Means clustering algorithm for a given k."""
        centroids = initialize_centroids(np.array(points), k)
        prev_centroids = np.zeros_like(centroids)
        clusters = [[] for _ in range(k)]
        
        for _ in range(max_iters):
            clusters = [[] for _ in range(k)]
            
            # Assign each point to the nearest centroid
            for point in points:
                distances = [np.linalg.norm(point - centroid) for centroid in centroids]
                cluster_idx = np.argmin(distances)
                clusters[cluster_idx].append(point)
            
            # Recalculate centroids
            for i in range(k):
                if clusters[i]:
                    centroids[i] = np.mean(clusters[i], axis=0)
            
            # Check for convergence
            if np.linalg.norm(centroids - prev_centroids) < tolerance:
                break
            prev_centroids = centroids.copy()
        
        score = sum(np.linalg.norm(point - centroids[np.argmin([np.linalg.norm(point - centroid) for centroid in centroids])]) for point in points)
        return clusters, score
    
    best_k = k_clusters
    if k_clusters is None:
        # Iterate over different values of k to find the best clustering
        best_score = float("inf")
        for k in range(2, min(10, len(points) // 2)):
            clusters, score = run_k_means(k, points)
            if score < best_score:
                best_k = k
                clusts = clusters
    else:
        clusts, _ = run_k_means(k_clusters, points)
    
    return clusts




def save_points(clusts, out_path, out_path_tagged):
    """
    Saves clustered points to CSV files, with and without cluster labels.
    
    Parameters:
    clusts (list): List of clusters, where each cluster is a list of points.
    out_path (str): Path to save the CSV file without cluster labels.
    out_path_tagged (str): Path to save the CSV file with cluster labels.
    
    Returns:
    None
    """
    # Save points without cluster labels
    with open(out_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for cluster in clusts:
            for point in cluster:
                writer.writerow(point)
    
    # Save points with cluster labels
    with open(out_path_tagged, mode='w', newline='') as file:
        writer = csv.writer(file)
        for cluster_id, cluster in enumerate(clusts):
            for point in cluster:
                writer.writerow(list(point) + [cluster_id])


load = load_data("data.csv", 2)
data  =k_means(2,4,100,load)
save_points(data, "output.csv", "output_tagged.csv")