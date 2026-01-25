# src/data/stock_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class StockDataset(Dataset):
    def __init__(
        self,
        data,
        features,
        target,
        window_size=60,
        horizon=30,
        max_sequences=None,
        threshold=0.0  # بدون threshold - خذ كل البيانات
    ):
        self.features = features
        self.target = target
        self.window_size = window_size
        self.horizon = horizon

        self.data_by_ticker = {
            ticker: group.reset_index(drop=True) 
            for ticker, group in data.groupby("Ticker")
        }
        
        self.indices = []
        labels_count = {0: 0, 1: 0}
        
        for ticker, group in self.data_by_ticker.items():
            group_len = len(group)
            max_start = group_len - window_size - horizon
            if max_start <= 0:
                continue
                
            for i in range(max_start):
                current_price = group.loc[i + window_size - 1, target]
                future_price = group.loc[i + window_size - 1 + horizon, target]
                ret = (future_price - current_price) / current_price
                
                # بدون threshold - التصنيف المباشر
                label = 1 if ret > 0 else 0
                self.indices.append((ticker, i, label))
                labels_count[label] += 1

        if max_sequences is not None:
            import random
            random.shuffle(self.indices)
            self.indices = self.indices[:max_sequences]
        
        # طباعة التوازن
        total = labels_count[0] + labels_count[1]
        if total > 0:
            print(f"Dataset Balance: Up={labels_count[1]/total*100:.1f}% Down={labels_count[0]/total*100:.1f}%")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ticker, start, label = self.indices[idx]
        group = self.data_by_ticker[ticker]
        
        X = group.loc[start:start + self.window_size - 1, self.features].values
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(label, dtype=torch.long)