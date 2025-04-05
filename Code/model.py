
# model.py
"""
Model definitions for SHAPAttention.
"""
import torch
import torch.nn as nn


class SHAPAttention(nn.Module):
    def __init__(self, hidden_dim=256, input_dim=1024, shap_dim=1024):
        super().__init__()

        # Input feature normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # Dimension alignment projection
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # SHAP weight processing
        self.shap_proj = nn.Sequential(
            nn.Linear(shap_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=0.1,
            batch_first=True
        )

        # Dynamic parameters
        self.shap_scale = nn.Parameter(torch.ones(1))
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialization
        nn.init.xavier_normal_(self.feature_proj[0].weight)
        nn.init.kaiming_normal_(self.shap_proj[0].weight)

    def forward(self, x: torch.Tensor, shap_weights: torch.Tensor) -> tuple:
        # Input normalization
        x = self.input_norm(x)

        # Feature projection
        x_proj = self.feature_proj(x)
        shap_proj = self.shap_proj(shap_weights)

        # Integrate SHAP information
        x_combined = x_proj * (1 + self.shap_scale * torch.sigmoid(shap_proj))

        # Attention calculation (using standard multi-head attention)
        attn_output, attn_weights = self.attention(
            query=x_combined.unsqueeze(1),
            key=x_combined.unsqueeze(1),
            value=x_combined.unsqueeze(1)
        )

        # Residual connection
        output = self.layer_norm(attn_output.squeeze(1) + x_proj)
        return output, attn_weights


class Conv1DCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(Conv1DCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(2)

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool1d(2)

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(64 * (input_size // 64), 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output_size)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output_size)
        )

        self.attention = SHAPAttention(hidden_size, 64 * (input_size // 64), input_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.Linear = nn.Linear(64 * (input_size // 64), hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, shap_weights=None):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        out = x.view(x.size(0), -1)  # Flatten

        if shap_weights is not None:
            context_vector, attention_weights = self.attention(out, shap_weights)
            out = context_vector
            x = self.fc1(out)
        else:
            x = self.fc(out)
        return x
