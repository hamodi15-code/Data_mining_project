import numpy as np
import csv
from part_1 import generate_data, load_data, h_clustering, k_means, save_points
from part_2 import bfr_cluster, cure_cluster,generate_large_data

def process_clustering(method, dim, k_clusters, n_points, in_path, out_path, block_size=1000000):
    """
    General function to handle clustering using the selected method.
    """
    points = load_data(in_path, dim, n_points)
    
    if method == 'hierarchical':
        clusters = h_clustering(dim, k_clusters, points)
    elif method == 'kmeans':
        clusters = k_means(dim, k_clusters, n_points, points)
    elif method == 'bfr':
        clusters = bfr_cluster(dim, k_clusters, n_points, block_size, in_path, out_path)
        return clusters  # Already saved inside the function
    elif method == 'cure':
        clusters = cure_cluster(dim, k_clusters, n_points, block_size, in_path, out_path)
        return clusters  # Already saved inside the function
    else:
        raise ValueError("Invalid clustering method.")
    
    save_points(clusters, out_path, out_path.replace(".csv", "_tagged.csv"))
    return clusters


# Generate required datasets
# 1. Small dataset with 2 dimensions and 5 clusters (~1000 rows)
generate_data(2, 5, 1000, "small_data.csv")

# 2. Dataset for memory-based clustering with dim > 3, k > 4, dim + k > 10
generate_data(4, 7, 5000, "memory_data.csv")

generate_data(5, 6, 8000, "medium_data.csv")
generate_data(10, 8, 15000, "large_data.csv")

# Example Usage of clustering
process_clustering('hierarchical', 2, 5, 1000, "small_data.csv", "hierarchical_output.csv")
process_clustering('kmeans', 2, 5, 1000, "small_data.csv", "kmeans_output.csv")
process_clustering('hierarchical', 4, 7, 5000, "memory_data.csv", "hierarchical_memory_output.csv")
process_clustering('kmeans', 4, 7, 5000, "memory_data.csv", "kmeans_memory_output.csv")
process_clustering('bfr', 10, 8, 15000, "large_data.csv", "bfr_large_output.csv")
process_clustering('cure', 10, 8, 15000, "large_data.csv", "cure_large_output.csv")
