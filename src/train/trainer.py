import torch
import torch.nn as nn
import time

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        scheduler=None,
        patience=10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.patience = patience
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in self.train_loader:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return total_loss / len(self.train_loader), correct / total

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                outputs = self.model(X)
                loss = self.criterion(outputs, y)

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return total_loss / len(self.val_loader), correct / total

    def fit(self, epochs):
        for epoch in range(epochs):
            start = time.time()

            tr_loss, tr_acc = self.train_epoch()
            va_loss, va_acc = self.validate_epoch()

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - start

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"TrLoss:{tr_loss:.4f} TrAcc:{tr_acc*100:.2f}% | "
                f"VaLoss:{va_loss:.4f} VaAcc:{va_acc*100:.2f}% | "
                f"{elapsed/60:.1f}m"
            )

            if va_acc > self.best_val_acc:
                self.best_val_acc = va_acc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    break
