# src/train/trainer.py
import torch
import torch.nn as nn
import time
import numpy as np

class LabelSmoothingLoss(nn.Module):
    """تقنية لتحسين التعميم"""
    def __init__(self, classes=2, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        scheduler=None,
        use_label_smoothing=True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        
        # استخدام Label Smoothing
        if use_label_smoothing:
            self.criterion = LabelSmoothingLoss(classes=2, smoothing=0.1)
        else:
            self.criterion = nn.M()
        
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience = 7
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X, y in self.train_loader:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total if total > 0 else 0
        return total_loss / len(self.train_loader), train_acc

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0

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
                
                true_positives += ((preds == 1) & (y == 1)).sum().item()
                false_positives += ((preds == 1) & (y == 0)).sum().item()
                false_negatives += ((preds == 0) & (y == 1)).sum().item()

        acc = correct / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return total_loss / len(self.val_loader), acc, precision, recall, f1

    def fit(self, epochs):
        print(f"\n{'='*70}")
        print(f"{'التدريب بدأ':^70}")
        print(f"{'='*70}\n")
        
        for epoch in range(epochs):
            start = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, precision, recall, f1 = self.validate_epoch()

            epoch_time = time.time() - start
            eta = epoch_time * (epochs - epoch - 1)

            print(
                f"Epoch [{epoch+1:3d}/{epochs}] "
                f"TrLoss: {train_loss:.4f} TrAcc: {train_acc*100:5.2f}% | "
                f"VaLoss: {val_loss:.4f} VaAcc: {val_acc*100:5.2f}% "
                f"F1: {f1*100:5.2f}% | "
                f"{epoch_time/60:.1f}m ETA:{eta/60:.0f}m"
            )
            
            # Scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
                
            # حفظ أفضل نموذج
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, 'best_model.pt')
                print(f"    ✓ نموذج محفوظ! Acc={val_acc*100:.2f}%")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\n⚠ Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\n{'='*70}")
        print(f"أفضل Validation Accuracy: {self.best_val_acc*100:.2f}%")
        print(f"{'='*70}\n")