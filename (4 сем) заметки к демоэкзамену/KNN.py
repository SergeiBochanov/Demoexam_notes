import numpy as np
from scipy.stats import mode

class KNN:
    def __init__(self, k = 5):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def get_class(self, point):
        distances = np.sqrt(np.sum((self.X_train - point)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest = self.y_train.iloc[k_indices]
        return mode(k_nearest)[0]

    def predict(self, X_test):
        return [self.get_class(point) for point in X_test]
