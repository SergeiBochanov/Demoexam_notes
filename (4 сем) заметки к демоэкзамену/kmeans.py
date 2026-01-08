import numpy as np

class my_KMeans:
    def __init__(self, k):
        self.n_clusters = k
        self.centroids = None
        self.clusters = None
        self.labels = None
    
    def fit(self, X):
        current_centroids = np.random.uniform(-2, 2, size=(self.n_clusters, X.shape[1]))

        while True:
            labels = []
            new_clusters = [[] for _ in range(self.n_clusters)]
            for point in X:
                min_dist = np.inf
                closest_cluster = 0
                for i in range(self.n_clusters):
                    dist = np.linalg.norm(current_centroids[i] - point)
                    if dist < min_dist:
                        min_dist = dist
                        closest_cluster = i
                new_clusters[closest_cluster].append(point)
                labels.append(closest_cluster)

            new_centroids = current_centroids.copy()
            for i in range(self.n_clusters):
                new_centroids[i] = np.mean(new_clusters[i], axis=0) if len(new_clusters[i]) > 0 else current_centroids[i]
            
            if np.array_equal(current_centroids, new_centroids):
                self.centroids = current_centroids
                self.clusters = new_clusters
                self.labels = labels
                break

            current_centroids = new_centroids
    
    def WSS(self):
        result = 0
        for i in range(self.n_clusters):
            for point in self.clusters[i]:
                result += (np.linalg.norm(self.centroids[i] - point))**2
        return result
