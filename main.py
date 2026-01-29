import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from src.models.stock_predictor import StockPredictor
from src.data.stock_dataset import StockDataset
from src.helpers.split import ticker_split
from src.helpers import config
from src.train.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features = ["Open", "High", "Low", "Volume"]
target = "Close"

data = pd.read_csv(config.DATA_PATH, parse_dates=["Date"]).sort_values(["Ticker", "Date"])
data = data.replace([np.inf, -np.inf], np.nan)
data[["Dividends", "Stock Splits"]] = data[["Dividends", "Stock Splits"]].fillna(0)
data = data.ffill().bfill()

train_df, val_df, test_df = ticker_split(data)

scaler = StandardScaler()
train_df[features] = scaler.fit_transform(train_df[features])
val_df[features] = scaler.transform(val_df[features])
test_df[features] = scaler.transform(test_df[features])

joblib.dump(scaler, "scaler.pkl")

train_ds = StockDataset(
    train_df,
    features,
    target,
    window_size=config.WINDOW_SIZE,
    horizon=config.HORIZON,
    max_sequences=1_000_000
)

val_ds = StockDataset(
    val_df,
    features,
    target,
    window_size=config.WINDOW_SIZE,
    horizon=config.HORIZON,
    max_sequences=500_000
)

train_loader = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
)

val_loader = DataLoader(
    val_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
)

model = StockPredictor(
    num_features=len(features),
    dropout=0.3
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=1e-6
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    scheduler=scheduler,
    patience=10
)

trainer.fit(config.EPOCHS)
