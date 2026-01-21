import torch


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
 