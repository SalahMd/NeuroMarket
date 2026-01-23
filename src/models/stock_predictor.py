import torch
import torch.nn as nn

from .cnn_block import CNNBlock
from .lstm_block import LSTMBlock


import torch
import torch.nn as nn
import time

class StockPredictor(nn.Module):
    def __init__(self, num_features, cnn_channels=32, lstm_hidden_size=64):
        super().__init__()

        # CNN: أنماط قصيرة المدى
        self.cnn = CNNBlock(
            in_features=num_features,
            out_channels=cnn_channels,
            kernel_size=3
        )

        # LSTM: اعتماد زمني طويل
        self.lstm = LSTMBlock(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_size
        )

        # مخرج 2 neurons: 0=هبوط، 1=صعود
        self.fc = nn.Linear(lstm_hidden_size, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = self.lstm(x)
        x = self.fc(x)
        return x  # logits