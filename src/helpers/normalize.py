import numpy as np

def normalize_data(X_train, X_val, X_test):
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1)) + 1e-8

    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std
    )
