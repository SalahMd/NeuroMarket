import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """إضافة معلومات الموقع الزمني"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerStockPredictor(nn.Module):
    """
    Transformer-based model للتنبؤ بالأسهم
    
    المزايا:
    - أسرع من LSTM (parallel processing)
    - attention على كل الفترة الزمنية
    - يتعلم العلاقات البعيدة بسهولة
    """
    def __init__(self, num_features, d_model=128, nhead=8, num_layers=3, dropout=0.3):
        super().__init__()
        
        # ===== Input Embedding =====
        self.input_projection = nn.Linear(num_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # ===== Transformer Encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # ===== Output Head =====
        self.dropout = nn.Dropout(dropout)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
    
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        
        # Project to d_model dimensions
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        
        # Global pooling over time dimension
        x = x.permute(0, 2, 1)  # [batch, d_model, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch, d_model]
        
        # Classification
        x = self.dropout(x)
        x = self.classifier(x)  # [batch, 2]
        
        return x


class HybridCNNTransformer(nn.Module):
    """
    الأقوى: CNN للأنماط المحلية + Transformer للعلاقات البعيدة
    """
    def __init__(self, num_features, d_model=128, nhead=8, num_layers=2, dropout=0.3):
        super().__init__()
        
        # ===== CNN للأنماط المحلية =====
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, d_model, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(d_model)
        
        # ===== Transformer للعلاقات البعيدة =====
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ===== Output =====
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        # CNN للأنماط المحلية
        x = x.permute(0, 2, 1)  # [B, F, T]
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # [B, T, D]
        
        # Positional encoding
        x = self.positional_encoding(x)
        
        # Transformer للعلاقات البعيدة
        x = self.transformer(x)  # [B, T, D]
        
        # Pooling + Classification
        x = x.permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x