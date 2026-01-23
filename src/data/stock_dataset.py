import torch
from torch.utils.data import Dataset
import numpy as np


class StockDataset(torch.utils.data.Dataset):
    def __init__(self, data, features, target, window_size=30, horizon=30):
        self.window_size = window_size
        self.horizon = horizon
        self.features = features
        self.target = target
        self.data = data

        self.sequences = self._create_sequences()

    def _create_sequences(self):
        sequences = []

        for _, group in self.data.groupby("Ticker"):

            feature_values = group[self.features].values
            prices = group[self.target].values

            max_i = len(group) - self.window_size - self.horizon
            if max_i <= 0:
                continue

            for i in range(max_i):
                X = feature_values[i : i + self.window_size]

                current_price = prices[i + self.window_size - 1]
                future_price = prices[i + self.window_size - 1 + self.horizon]

                y = 1 if future_price > current_price else 0

                sequences.append((X, y))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y = self.sequences[idx]
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )

