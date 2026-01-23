import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.stock_predictor import StockPredictor
from src.train.trainer import Trainer
from src.train.eval import Evaluator
from src.data.stock_dataset import StockDataset
from src.helpers.split import ticker_split
from src.helpers import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare data
feature_columns = [
    "Open", "High", "Low",  "Volume", "Dividends", "Stock Splits"
]
target_column = "Close"

# Read the data
chunks = pd.read_csv(
    config.DATA_PATH,
    usecols=["Date", "Ticker", target_column] + feature_columns,

    chunksize=200_000,
    engine="python",
    parse_dates=["Date"]
)
data = pd.concat(chunks)
data = data.sort_values(by=["Ticker", "Date"])
data = data.drop(columns=["Date"])

# Split data
train_data, val_data, test_data = ticker_split(
    data
)

# Scale features
target_scaler = StandardScaler()

train_data[target_column] = target_scaler.fit_transform(
    train_data[[target_column]]
)
val_data[target_column] = target_scaler.transform(
    val_data[[target_column]]
)
test_data[target_column] = target_scaler.transform(
    test_data[[target_column]]
)


# Create datasets
train_ds = StockDataset(
    data=train_data,
    features=feature_columns,
    target=target_column,
    window_size=config.WINDOW_SIZE
)
val_ds = StockDataset(
    data=val_data,
    features=feature_columns,
    target=target_column,
    window_size=config.WINDOW_SIZE
)
test_ds = StockDataset(
    data=test_data,
    features=feature_columns,
    target=target_column,
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

model = StockPredictor(
    num_features=len(feature_columns)
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device
)

trainer.fit(epochs=config.EPOCHS)

evaluator = Evaluator(model, test_loader, device)
mse = evaluator.mse()

print(f"Test MSE: {mse:.4f}")