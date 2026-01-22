import torch
from torch.utils.data import Dataset
import pandas as pd


class StockDataset(Dataset):
    def __init__(self, csv_path, window_size=30):
        self.window_size = window_size

        self.feature_columns = [
            "Open", "High", "Low", "Close",
            "Volume", "Dividends", "Stock Splits"
        ]

        self.target_column = "Close"

        chunks = pd.read_csv(
    csv_path,
    usecols=self.feature_columns,
    chunksize=200_000,
    engine="python"
)


        data = pd.concat(chunks)
        data = data.tail(5_000_000)

        self.features = data[self.feature_columns].values.astype("float32")
        self.targets = data[self.target_column].values.astype("float32")

    def __len__(self):
        return len(self.features) - self.window_size - 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]

        next_price = self.targets[idx + self.window_size]
        current_price = self.targets[idx + self.window_size - 1]

        y = 1.0 if next_price > current_price else 0.0

        return (
            torch.tensor(x),
            torch.tensor(y)
        )
