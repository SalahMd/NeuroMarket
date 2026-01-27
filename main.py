import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
import joblib

from src.models.stock_predictor import StockPredictor
from src.data.stock_dataset import StockDataset
from src.helpers.split import ticker_split
from src.helpers.split import ticker_split
from src.helpers import config

# ===== Setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

features = ["Open", "High", "Low", "Volume",
            #  "Dividends",
            #    "Stock Splits"
               ]
target = "Close"

# ===== Load & Prepare Data =====
data = pd.read_csv(config.DATA_PATH, parse_dates=["Date"]).sort_values(["Ticker", "Date"])
data = data.replace([np.inf, -np.inf], np.nan)
data[["Dividends", "Stock Splits"]] = data[["Dividends", "Stock Splits"]].fillna(0)
data = data.ffill().bfill()

print(f"Total data: {len(data):,} rows, {data['Ticker'].nunique()} tickers")

train_df, val_df, test_df = ticker_split(data)

scaler = StandardScaler()
train_df[features] = scaler.fit_transform(train_df[features])
val_df[features] = scaler.transform(val_df[features])
test_df[features] = scaler.transform(test_df[features])

joblib.dump(scaler, 'scaler.pkl')

train_ds = StockDataset(
    train_df, features, target,
    window_size=config.WINDOW_SIZE,
    horizon=config.HORIZON,
    max_sequences=4_000_000
)

val_ds = StockDataset(
    val_df, features, target,
    window_size=config.WINDOW_SIZE,
    horizon=config.HORIZON,
    max_sequences=1_000_000
)

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

model = StockPredictor(num_features=len(features), dropout=0.3).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config.LEARNING_RATE,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

criterion = torch.nn.CrossEntropyLoss()

best_val_acc = 0.0
patience_counter = 0
patience_limit = 10

print("\n" + "="*70)
print("Training started".center(70))
print("="*70 + "\n")

for epoch in range(config.EPOCHS):
    epoch_start = time.time()
    
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        preds = outputs.argmax(1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}", end='\r')
    
    train_acc = train_correct / train_total if train_total > 0 else 0
    avg_train_loss = train_loss / len(train_loader)
    
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    true_pos = 0
    false_pos = 0
    false_neg = 0
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            val_loss += loss.item()
            preds = outputs.argmax(1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)
            
            true_pos += ((preds == 1) & (y == 1)).sum().item()
            false_pos += ((preds == 1) & (y == 0)).sum().item()
            false_neg += ((preds == 0) & (y == 1)).sum().item()
    
    val_acc = val_correct / val_total if val_total > 0 else 0
    avg_val_loss = val_loss / len(val_loader)
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    scheduler.step()
    
    epoch_time = time.time() - epoch_start
    eta = epoch_time * (config.EPOCHS - epoch - 1)
    
    print(
        f"Epoch [{epoch+1:3d}/{config.EPOCHS}] "
        f"TrLoss:{avg_train_loss:.4f} TrAcc:{train_acc*100:5.2f}% | "
        f"VaLoss:{avg_val_loss:.4f} VaAcc:{val_acc*100:5.2f}% "
        f"F1:{f1*100:5.2f}% | "
        f"{epoch_time/60:.1f}m ETA:{eta/60:.0f}m"
    )
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': avg_val_loss,
            'f1': f1
        }, 'best_model.pt')
        
        print(f"    ✓ Model saved! Acc={val_acc*100:.2f}% F1={f1*100:.2f}%")
    else:
        patience_counter += 1
        if patience_counter >= patience_limit:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break

print("\n" + "="*70)
print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
print("="*70 + "\n")