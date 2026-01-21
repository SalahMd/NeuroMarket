import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class StockDataset(Dataset):
    def __init__(
        self,
        csv_path,
        window_size=30,
        target_column="Close",
        feature_columns=None
    ):
        """
        csv_path: مسار ملف CSV
        window_size: عدد الأيام في كل نافذة
        target_column: العمود الذي نحدد منه الاتجاه
        feature_columns: الأعمدة المستخدمة كميزات
        """

        self.window_size = window_size

        # قراءة البيانات
        chunks = pd.read_csv(
    csv_path,
    usecols=feature_columns,
    chunksize=50_000
)

        self.data = pd.concat(list(chunks)[:5])
        self.data = self.data.tail(100_000)



        if feature_columns is None:
            feature_columns = ["Open", "High", "Low", "Close", "Volume"]

        self.features = self.data[feature_columns].values
        self.targets = self.data[target_column].values

        self.X, self.y = self._create_windows()

    def _create_windows(self):
        X, y = [], []

        for i in range(len(self.features) - self.window_size - 1):
            window = self.features[i : i + self.window_size]

            next_price = self.targets[i + self.window_size]
            current_price = self.targets[i + self.window_size - 1]

            label = 1 if next_price > current_price else 0

            X.append(window)
            y.append(label)

        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

    def time_split(X, y, train_ratio=0.7, val_ratio=0.15):
        n = len(X)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train = X[:train_end]
        y_train = y[:train_end]

        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]

        X_test = X[val_end:]
        y_test = y[val_end:]

        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def normalize_data(X_train, X_val, X_test):
        mean = X_train.mean(axis=(0, 1))
        std = X_train.std(axis=(0, 1)) + 1e-8

        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

        return X_train, X_val, X_test

