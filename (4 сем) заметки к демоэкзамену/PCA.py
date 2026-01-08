import numpy as np

class my_PCA:
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit_transform(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean)/std

        cov_matrix = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        transform_matrix = eigenvectors[:, :self.n_components]
        return np.dot(X, transform_matrix)
