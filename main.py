import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.models.stock_predictor import AdvancedStockPredictor
from src.train.trainer import Trainer
from src.train.eval import Evaluator
from src.data.stock_dataset import StockDataset
from src.helpers.split import ticker_split
from src.helpers import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*70)
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("="*70)

# Features المتاحة فقط (حسب المشروع)
features = [
    "Open", "High", "Low", "Close",
    "Volume", "Dividends", "Stock Splits"
]
target = "Close"

# تحميل البيانات
data = pd.read_csv(
    config.DATA_PATH,
    parse_dates=["Date"]
).sort_values(["Ticker", "Date"])

# معالجة القيم الشاذة
data = data.replace([np.inf, -np.inf], np.nan)
data[["Dividends", "Stock Splits"]] = data[["Dividends", "Stock Splits"]].fillna(0)
data = data.ffill().bfill()

print(f"إجمالي البيانات: {len(data):,} صف")
print(f"عدد الأسهم: {data['Ticker'].nunique()}")

# تقسيم البيانات
train_df, val_df, test_df = ticker_split(data)

# Normalization
scaler = StandardScaler()
train_df[features] = scaler.fit_transform(train_df[features])
val_df[features] = scaler.transform(val_df[features])
test_df[features] = scaler.transform(test_df[features])

# إنشاء Datasets
train_ds = StockDataset(
    train_df,
    features,
    target,
    window_size=config.WINDOW_SIZE,
    horizon=config.HORIZON,
    max_sequences=2_000_000  # زود البيانات
)

val_ds = StockDataset(
    val_df,
    features,
    target,
    window_size=config.WINDOW_SIZE,
    horizon=config.HORIZON,
    max_sequences=1_000_000
)

# DataLoaders
train_loader = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

# النموذج المتقدم
model = AdvancedStockPredictor(
    num_features=len(features),
    dropout=0.3
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nعدد Parameters: {total_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# التدريب
trainer = Trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    scheduler=scheduler,
    use_label_smoothing=True
)

trainer.fit(config.EPOCHS)

# التقييم
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

evaluator = Evaluator(model, val_loader, device)
final_acc = evaluator.accuracy()
print(f"\n{'='*70}")
print(f"Final Validation Accuracy: {final_acc*100:.2f}%")
print(f"{'='*70}\n")