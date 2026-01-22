# from torch.utils.data import DataLoader, TensorDataset
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np
# from torchvision import datasets, transforms

# import train

# model = nn.Sequential(
# nn.Linear(2, 4),
# nn.ReLU(),
# nn.Linear(4, 4),
# nn.ReLU(),
# nn.Linear(4, 1),
# nn.Sigmoid()
# )
# x = np.random.uniform(low=-1, high=1, size=(200, 2))
# y = np.ones(len(x))
# y[x[:, 0] * x[:, 1] < 0] = 0
# num_epochs = 200
# n_train = 100
# x_train = torch.tensor(x[:n_train, :], dtype=torch.float32)
# y_train = torch.tensor(y[:n_train], dtype=torch.float32)
# train_ds = TensorDataset(x_train, y_train)
# batch_size = 200
# torch.manual_seed(1)
# train_dl = DataLoader(train_ds, batch_size, shuffle=True)
# x_valid = torch.tensor(x[n_train:, :], dtype=torch.float32)
# y_valid = torch.tensor(y[n_train:], dtype=torch.float32)
# history = train(model, num_epochs, train_dl, x_valid, y_valid)
# loss_fn = nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.015)
# fig = plt.figure(figsize=(16, 4))
# ax = fig.add_subplot(1, 2, 1)
# plt.plot(history[0], lw=4)
# plt.plot(history[1], lw=4)
# plt.legend(['Train loss', 'Validation loss'], fontsize=15)
# ax.set_xlabel('Epochs', size=15)
# ax = fig.add_subplot(1, 2, 2)
# plt.plot(history[2], lw=4)
# plt.plot(history[3], lw=4)
# plt.legend(['Train acc.', 'Validation acc.'], fontsize=15)
# ax.set_xlabel('Epochs', size=15)
# import torch

# from models.cnn import CNNBlock
# from models.lstm import LSTMBlock


# x = torch.randn(8, 30, 5)

# cnn = CNNBlock(in_features=5)
# cnn_out = cnn(x)

# lstm = LSTMBlock(input_size=32)
# lstm_out = lstm(cnn_out)

# print(cnn_out.shape)
# print(lstm_out.shape)

# from src.data.stock_dataset import StockDataset


# dataset = StockDataset(
#     csv_path="data/raw/train.csv",
#     window_size=30
# )


# x, y = dataset[0]

# print(x.shape)  # (30, 5)
# print(y)        # 0 أو 1
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.stock_classifier import StockClassifier
from src.train.trainer import Trainer
from src.train.eval import Evaluator
from src.data.stock_dataset import StockDataset
from src.helpers import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare data
feature_columns = [
    "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"
]
target_column = "Close"

# Read the data
chunks = pd.read_csv(
    config.DATA_PATH,
    usecols=feature_columns,
    chunksize=200_000,
    engine="python"
)
data = pd.concat(chunks)
data = data.tail(5_000_000)

# Split data
n = len(data)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_data = data.iloc[:train_end]
val_data = data.iloc[train_end:val_end]
test_data = data.iloc[val_end:]

# Scale features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_data[feature_columns].values)
val_features = scaler.transform(val_data[feature_columns].values)
test_features = scaler.transform(test_data[feature_columns].values)

train_targets = train_data[target_column].values
val_targets = val_data[target_column].values
test_targets = test_data[target_column].values

# Create datasets
train_ds = StockDataset(
    features=train_features,
    targets=train_targets,
    window_size=config.WINDOW_SIZE
)
val_ds = StockDataset(
    features=val_features,
    targets=val_targets,
    window_size=config.WINDOW_SIZE
)
test_ds = StockDataset(
    features=test_features,
    targets=test_targets,
    window_size=config.WINDOW_SIZE
)

# Create data loaders
train_loader = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False
)
val_loader = DataLoader(
    val_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False
)
test_loader = DataLoader(
    test_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False
)

model = StockClassifier(num_features=len(feature_columns)).to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device
)

trainer.fit(epochs=config.EPOCHS)

evaluator = Evaluator(model, test_loader, device)
acc = evaluator.accuracy()

print(f"Test Accuracy: {acc:.2%}")

