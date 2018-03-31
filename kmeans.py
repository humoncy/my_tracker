import numpy as np
import matplotlib.pyplot as plt

# calculate Euclidean distance
def euclidean_distance(x1, x2):
    scale_x1 = x1 * 100
    scale_x2 = x2 * 100

    return np.sqrt(np.sum(np.power(scale_x1 - scale_x2, 2)))


# Initial centroids with random samples
def init_centroids(data, k):
    num_samples, dim = data.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = int(np.random.uniform(0, num_samples))
        centroids[i] = data[index]
    return centroids


# K-means clusters
def k_means(data, k):
    print("K-means clustering...")
    print("Data shape for K-means:", data.shape)
    if data.ndim == 1:
        raise Exception("Reshape your data either using array.reshape(-1, 1) if your data has a single feature "
                        "or array.reshape(1, -1) if it contains a single sample.")

    num_samples = data.shape[0]
    # First column stores which cluster this sample belongs to,
    # Second column stores the error between this sample and its centroid
    cluster_assignment = np.zeros((num_samples, 2))
    cluster_changed = True

    # Step 1: init centroids
    centroids = init_centroids(data, k)
    print("Centroids initialization:")
    print(centroids)
    
    # show_cluster(data, k, cluster_assignment[:, 0], centroids, title="K-means, initial centroids")

    num_iterations = 0
    while cluster_changed:
        cluster_changed = False
        # for each sample
        for j in range(num_samples):
            min_distance = 100000.0
            min_index = 0
            # for each centroid
            # Step 2: find the centroid who is closest
            for i in range(k):
                distance = euclidean_distance(data[j], centroids[i])
                if distance < min_distance:
                    min_distance = distance
                    min_index = i

            # Step 3: update its cluster
            if cluster_assignment[j, 0] != min_index:
                cluster_changed = True
                cluster_assignment[j] = min_index, np.power(min_distance, 2)

        # Step 4: update centroids
        for i in range(k):
            points_in_cluster = data[np.nonzero(cluster_assignment[:, 0] == i)[0]]
            if len(points_in_cluster) > 0:
                centroids[i] = np.mean(points_in_cluster, axis=0)

        title = "K-means, #iter:" + num_iterations.__str__()
        print(title)
        print(centroids)
        # show_cluster(data, k, cluster_assignment[:, 0], centroids, title=title)

        num_iterations += 1

    # show_cluster(data, k, cluster_assignment[:, 0], title="eigen_space")

    return centroids, cluster_assignment[:, 0]
