import numpy as np
import csv
from part_1 import load_data, h_clustering, k_means, save_points
from part_2 import bfr_cluster, cure_cluster, generate_large_data

def process_clustering(method, dim, k_clusters_list, n_points, in_path, out_path_pattern, block_size=1000000):
    """
    General function to handle clustering using the selected method for multiple values of k.
    """
    print(f"Running {method} clustering on {in_path} for multiple k values...")
    points = load_data(in_path, dim, n_points)
    
    for k_clusters in k_clusters_list:
        out_path = out_path_pattern.replace("{k}", str(k_clusters))
        
        if method == 'hierarchical':
            clusters = h_clustering(dim, k_clusters, points)
        elif method == 'kmeans':
            clusters = k_means(dim, k_clusters, n_points, points)
        elif method == 'bfr':
            clusters = bfr_cluster(dim, k_clusters, n_points, block_size, in_path, out_path)
            continue  # Already saved inside the function
        elif method == 'cure':
            clusters = cure_cluster(dim, k_clusters, n_points, block_size, in_path, out_path)
            continue  # Already saved inside the function
        else:
            raise ValueError("Invalid clustering method.")
        
        save_points(clusters, out_path, out_path.replace(".csv", "_tagged.csv"))
        print(f"Clustering completed: {out_path}")

def compare_cluster_results(original_file, clustered_file):
    """
    Compares the original dataset with the clustered dataset to evaluate clustering accuracy.
    """
    original_labels = []
    clustered_labels = []
    
    with open(original_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            original_labels.append(row[-1])
    
    with open(clustered_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            clustered_labels.append(row[-1])
    
    if len(original_labels) != len(clustered_labels):
        print("⚠️ Mismatch in number of rows between original and clustered data!")
        return None
    
    correct = sum(1 for i in range(len(original_labels)) if original_labels[i] == clustered_labels[i])
    accuracy = correct / len(original_labels)
    print(f"✅ Clustering Accuracy for {clustered_file}: {accuracy:.4f}")
    return accuracy

# Generate required datasets
print("Generating datasets...")
data_files = [
    (2, 5, 1000, "small_data.csv"),
    (4, 7, 5000, "memory_data.csv"),
    (5, 6, 8000, "medium_data.csv"),
    (10, 8, 15000, "large_data.csv"),
    (10, 8, 15000, "large_data2.csv")
]
for dim, k, n, filename in data_files:
    generate_large_data(dim, k, n, filename)
print("Dataset generation completed.")

# Run clustering on datasets efficiently
k_values = list(range(2, 9))  # k values from 2 to 8
clustering_tasks = [
    ('hierarchical', 2, k_values, 1000, "small_data.csv", "hierarchical_output{k}.csv"),
    ('kmeans', 2, k_values, 1000, "small_data.csv", "kmeans_output{k}.csv"),
    ('hierarchical', 4, k_values, 5000, "memory_data.csv", "hierarchical_memory_output{k}.csv"),
    ('kmeans', 4, k_values, 5000, "memory_data.csv", "kmeans_memory_output{k}.csv"),
    ('bfr', 10, k_values, 15000, "large_data.csv", "bfr_large_output{k}.csv"),
    ('cure', 10, k_values, 15000, "large_data2.csv", "cure_large_output{k}.csv")
]
for task in clustering_tasks:
    process_clustering(*task)

# Compare clustering results
data_comparisons = []
for k in k_values:
    data_comparisons.append(("small_data.csv", f"hierarchical_output{k}_tagged.csv"))
    data_comparisons.append(("small_data.csv", f"kmeans_output{k}_tagged.csv"))
    data_comparisons.append(("memory_data.csv", f"hierarchical_memory_output{k}.csv"))
    data_comparisons.append(("memory_data.csv", f"kmeans_memory_output{k}.csv"))
    data_comparisons.append(("large_data.csv", f"bfr_large_output{k}.csv"))
    data_comparisons.append(("large_data2.csv", f"cure_large_output{k}.csv"))

for original, clustered in data_comparisons:
    compare_cluster_results(original, clustered)

print("All clustering tasks and evaluations completed.")