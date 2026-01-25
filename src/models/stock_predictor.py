import torch
import torch.nn as nn
from .cnn_block import CNNBlock
from .lstm_block import LSTMBlock
# أضف هذا في أول الملف
import torch
import torch.nn as nn
import torch.nn.functional as F  # ← هذا السطر مفقود

# class StockPredictor(nn.Module):
#     def __init__(
#         self,
#         num_features,
#         cnn_channels=32,
#         lstm_hidden_size=64
#     ):
#         super().__init__()

#         self.cnn = CNNBlock(
#             in_features=num_features,
#             out_channels=cnn_channels,
#             kernel_size=3
#         )

#         self.lstm = LSTMBlock(
#             input_size=cnn_channels,
#             hidden_size=lstm_hidden_size
#         )

#         self.fc = nn.Linear(lstm_hidden_size, 2)

#     def forward(self, x):
#         x = self.cnn(x)
#         x = self.lstm(x)
#         x = self.fc(x)
#         return x
class StockPredictor(nn.Module):
    def __init__(self, num_features, dropout=0.3):
        super().__init__()
        
        # CNN أعمق مع Batch Normalization
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # LSTM بطبقتين
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # مهم جداً!
        )
        
        # Attention Layer
        self.attention = nn.Linear(256, 1)
        
        # Classification Head
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, x):
        # CNN layers
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # LSTM
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        x = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
# src/models/advanced_stock_predictor.py
# src/models/advanced_stock_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedStockPredictor(nn.Module):
    """
    بنية هجينة تجمع:
    - CNN: للأنماط قصيرة المدى (Local Patterns)
    - LSTM: للسلوك طويل المدى (Long-Term)
    - Attention: للتركيز على الفترات المهمة (Seasonality)
    """
    def __init__(self, num_features, dropout=0.3):
        super().__init__()
        
        # ==========================================
        # 1. Multi-Scale CNN (للأنماط قصيرة المدى)
        # ==========================================
        # تصحيح الـ padding لضمان نفس الحجم
        
        # Short-term patterns (3 days)
        self.conv_short = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),  # padding=1 correct
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Medium-term patterns (7 days = week)
        self.conv_medium = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=7, padding=3),  # padding=3 correct
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Long-term patterns (15 days) - ← غيّرنا من 14 إلى 15
        self.conv_long = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=15, padding=7),  # padding=7 correct
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combine multi-scale features
        self.conv_combine = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=1),  # 64*3=192
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ==========================================
        # 2. Bi-directional LSTM (للسلوك طويل المدى)
        # ==========================================
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=True
        )
        
        # ==========================================
        # 3. Temporal Attention (للموسمية والفترات المهمة)
        # ==========================================
        self.attention_fc = nn.Linear(256, 128)
        self.attention_weights = nn.Linear(128, 1)
        
        # ==========================================
        # 4. Classification Head
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # ==========================================
        # Multi-Scale CNN
        # ==========================================
        x_cnn = x.permute(0, 2, 1)  # [B, Features, Time]
        
        # استخراج أنماط بمقاييس مختلفة
        short_features = self.conv_short(x_cnn)      # [B, 64, 60]
        medium_features = self.conv_medium(x_cnn)    # [B, 64, 60]
        long_features = self.conv_long(x_cnn)        # [B, 64, 60]
        
        # دمج الأنماط
        multi_scale = torch.cat([short_features, medium_features, long_features], dim=1)  # [B, 192, 60]
        combined = self.conv_combine(multi_scale)  # [B, 128, 60]
        
        # ==========================================
        # LSTM للسلوك طويل المدى
        # ==========================================
        lstm_input = combined.permute(0, 2, 1)  # [B, 60, 128]
        lstm_out, _ = self.lstm(lstm_input)  # [B, 60, 256]
        
        # ==========================================
        # Temporal Attention (للتركيز على الفترات المهمة)
        # ==========================================
        attention_input = torch.tanh(self.attention_fc(lstm_out))  # [B, 60, 128]
        attention_scores = self.attention_weights(attention_input)  # [B, 60, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # تطبيق الأوزان
        attended_features = torch.sum(attention_weights * lstm_out, dim=1)  # [B, 256]
        
        # ==========================================
        # Classification
        # ==========================================
        output = self.classifier(attended_features)
        
        return output