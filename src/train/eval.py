import torch


class Evaluator:
    def __init__(self, model, loader, device):
        self.model = model
        self.loader = loader
        self.device = device

    def accuracy(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in self.loader:
                X = X.to(self.device)
                y = y.to(self.device)

                outputs = self.model(X)
                preds = outputs.argmax(dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / total if total > 0 else 0
