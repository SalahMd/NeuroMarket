import torch
import torch.nn as nn

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        # x: (batch, channels, time)

        x = x.permute(0, 2, 1)
        # الآن: (batch, time, channels)

        _, (h_n, _) = self.lstm(x)

        # نأخذ آخر hidden state
        last_hidden = h_n[-1]

        return last_hidden
