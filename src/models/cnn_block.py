import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_features, out_channels=32, kernel_size=3):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_features,   # عدد الميزات (features)
            out_channels=out_channels,
            kernel_size=kernel_size
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        # x: (batch, time, features)

        x = x.permute(0, 2, 1)
        # الآن: (batch, features, time)

        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

        return x
