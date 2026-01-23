import time
import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss() 
    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)  # ❗ بدون float
            if y_batch.dtype != torch.long:
                raise RuntimeError(f"Target dtype is {y_batch.dtype}, expected torch.long")

            self.optimizer.zero_grad()

            outputs = self.model(X_batch)  # (B, 2)
            loss = self.criterion(outputs, y_batch)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)


    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total if total > 0 else 0
        return total_loss / len(self.val_loader), accuracy

    def fit(self, epochs):
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            epoch_time = time.time() - epoch_start
            remaining = epoch_time * (epochs - epoch - 1)

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_acc*100:.2f}% "
                f"| Epoch Time: {epoch_time/60:.2f} min "
                f"| ETA: {remaining/60:.2f} min"
            )