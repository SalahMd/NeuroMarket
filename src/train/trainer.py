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
        tp = fp = fn = 0

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

                tp += ((preds == 1) & (y == 1)).sum().item()
                fp += ((preds == 1) & (y == 0)).sum().item()
                fn += ((preds == 0) & (y == 1)).sum().item()

        acc = correct / total
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return total_loss / len(self.val_loader), acc, f1

    def fit(self, epochs):
        for epoch in range(epochs):
            start = time.time()

            tr_loss, tr_acc = self.train_epoch()
            va_loss, va_acc, f1 = self.validate_epoch()

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - start

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"TrLoss:{tr_loss:.4f} TrAcc:{tr_acc*100:.2f}% | "
                f"VaLoss:{va_loss:.4f} VaAcc:{va_acc*100:.2f}% "
                f"F1:{f1*100:.2f}% | "
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
