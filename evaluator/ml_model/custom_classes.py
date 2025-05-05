# evaluator/ml_model/custom_classes.py
import numpy as np

class MockScaler:
    """A simple mock scaler class that can be pickled."""
    def transform(self, X):
        # Just return the same data standardized 
        X = np.array(X)
        return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)