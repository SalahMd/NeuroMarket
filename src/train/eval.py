import torch
import torch.nn.functional as F


class Evaluator:
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def accuracy(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                preds = (torch.sigmoid(outputs) > 0.5).int()

                correct += (preds.squeeze() == y_batch).sum().item()
                total += y_batch.size(0)

        return correct / total

    def mse(self):
        self.model.eval()
        total_mse = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float()

                outputs = self.model(X_batch).squeeze()
                
                total_mse += F.mse_loss(
                    outputs, y_batch, reduction="sum"
                ).item()
                total += y_batch.size(0)

        return total_mse / total