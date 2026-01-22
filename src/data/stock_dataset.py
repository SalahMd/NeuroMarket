import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, features, targets, window_size=30):
        self.window_size = window_size
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features) - self.window_size - 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]

        next_price = self.targets[idx + self.window_size]
        current_price = self.targets[idx + self.window_size - 1]

        y = 1.0 if next_price > current_price else 0.0

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
