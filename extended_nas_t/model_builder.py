import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics
import math

def build_model(model_type, **kwargs):
    print(f"Building model type: {model_type}")
    input_size = kwargs.get("input_size", 1)
    output_size = kwargs.get("output_size", 2)

    #if model_type == "FCNN":
    #    return FCNN(input_size=input_size, hidden_units=kwargs.get("hidden_units", 64), output_size=2)

    #elif model_type == "CNN":
    #    return CNN(input_channels=kwargs.get("input_channels", 1),
                  # num_filters=kwargs.get("num_filters", 32),
                  # kernel_size=kwargs.get("kernel_size", 3),
                  # output_size=2)

    #elif model_type == "LSTM":
    #    return LSTM(input_size=input_size, hidden_units=kwargs["hidden_units"], output_size=2)

    #elif model_type == "GRU":
    #    return GRU(input_size=input_size, hidden_units=kwargs["hidden_units"], output_size=2)

    if model_type == "Transformer":
        # Ensure hidden_dim is divisible by num_heads
        num_heads = kwargs.get("num_heads", 4)
        hidden_dim = kwargs.get("hidden_dim", 64)
        if hidden_dim % num_heads != 0:
            hidden_dim = num_heads * ((hidden_dim // num_heads) + 1)
        
        return TransformerModel(
            input_dim=input_size,
            num_heads=num_heads,
            num_layers=kwargs.get("num_layers", 2),
            hidden_dim=hidden_dim,
            output_size=output_size
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")


# Fully Connected Neural Network (FCNN)
"""class FCNN(pl.LightningModule):
    def __init__(self, input_size, hidden_units=256, output_size=2, num_layers=5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_units))
        self.layers.append(nn.BatchNorm1d(hidden_units))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=0.2))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
            self.layers.append(nn.BatchNorm1d(hidden_units))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=0.2))
        self.layers.append(nn.Linear(hidden_units, output_size))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() > 1:
            y = torch.argmax(y, dim=1)

        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)"""

# Convolutional Neural Network (CNN)
"""class CNN(pl.LightningModule):
    def __init__(self, input_channels, num_filters=64, kernel_size=5, output_size=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size)
        self.bn3 = nn.BatchNorm1d(num_filters * 4)
        self.conv4 = nn.Conv1d(num_filters * 4, num_filters * 8, kernel_size)
        self.bn4 = nn.BatchNorm1d(num_filters * 8)
        self.conv5 = nn.Conv1d(num_filters * 8, num_filters * 16, kernel_size)
        self.bn5 = nn.BatchNorm1d(num_filters * 16)
        self.conv6 = nn.Conv1d(num_filters * 16, num_filters * 32, kernel_size)
        self.bn6 = nn.BatchNorm1d(num_filters * 32)
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(num_filters * 16, output_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # shape: (batch_size, 1, time_steps)

        # Apply Conv & Pooling
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))

        # Global Average Pooling
        x = self.global_pool(x).squeeze(-1)  # shape: (batch_size, num_filters * 16)

        # Fully connected layer
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Ensure labels are in correct format
        if y.dim() > 1:  # Convert one-hot encoded y to class indices
            y = torch.argmax(y, dim=1)

        # Ensure `x` has correct shape before passing to forward()
        if x.dim() == 2:  # If missing channel dimension, add one
            x = x.unsqueeze(1)  # Shape: (batch_size, 1, time_steps)

        logits = self.forward(x)  # Forward pass
        loss = self.loss_fn(logits, y)  # Compute loss
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)"""

# LSTM-based Model
"""class LSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_units=128, output_size=2, num_layers=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_units, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.3 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_units * 2, output_size)  # Multiply by 2 for bidirectional
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.dim() == 2:  # Ensure 3D input for LSTM
            x = x.unsqueeze(1)  # Shape: (batch_size, time_steps=1, feature_dim)

        lstm_out, _ = self.lstm(x)  # Get the LSTM output
        lstm_out = lstm_out[:, -1, :]  # Use the last time step's output
        output = self.fc(lstm_out)  # Fully connected layer
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Ensure labels are in correct format
        if y.dim() > 1:  # Convert one-hot encoded y to class indices
            y = torch.argmax(y, dim=1)

        # Ensure `x` is correctly shaped for LSTM
        if x.dim() == 2:  # If missing time-step dimension, add one
            x = x.unsqueeze(1)  # Shape: (batch_size, 1, feature_dim)

        logits = self.forward(x)  # Forward pass
        loss = self.loss_fn(logits, y)  # Compute loss
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)"""


# GRU-based Model
"""class GRU(pl.LightningModule):
    def __init__(self, input_size, hidden_units=128, output_size=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_units, batch_first=True, num_layers=5, bidirectional=True, dropout=0.4)
        self.fc = nn.Linear(hidden_units * 2, output_size)  # Multiply by 2 for bidirectional
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.dim() == 2:  # Ensure 3D input for GRU
            x = x.unsqueeze(1)  # Shape: (batch_size, time_steps=1, feature_dim)

        gru_out, _ = self.gru(x)  # Get the GRU output
        gru_out = gru_out[:, -1, :]  # Use the last time step's output
        output = self.fc(gru_out)  # Fully connected layer
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() > 1:  # Convert one-hot encoded y to class indices
            y = torch.argmax(y, dim=1)

        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)"""

# Transformer-based Model
class TransformerModel(pl.LightningModule):
    def __init__(self, input_dim, num_heads=4, num_layers=2, hidden_dim=64, output_size=2):
        super().__init__()
        self.save_hyperparameters()
        
        # Ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            hidden_dim = num_heads * (hidden_dim // num_heads + 1)
        
        # Input embedding layer
        #self.embedding = nn.Linear(input_dim, hidden_dim)
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        #self.fc = nn.Linear(hidden_dim, output_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_size)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, hidden_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch_size, seq_len, hidden_dim)
        x = x.mean(dim=1)  # Pool over sequence length
        return self.fc(x)  # (batch_size, output_size)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.long())  # Convert y to long for CrossEntropyLoss
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.long())
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    """def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer"""
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x