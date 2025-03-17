import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics

def build_model(model_type, **kwargs):
    print(f"Building model type: {model_type}")  # Debugging statement
    input_size = kwargs.get("input_size", 1)

    if model_type == "FCNN":
        return FCNN(input_size=input_size, hidden_units=kwargs.get("hidden_units", 64), output_size=2)

    elif model_type == "CNN":
        return CNN(input_channels=kwargs.get("input_channels", 1),
                   num_filters=kwargs.get("num_filters", 32),
                   kernel_size=kwargs.get("kernel_size", 3),
                   output_size=2)

    elif model_type == "LSTM":
        return LSTM(input_size=input_size, hidden_units=kwargs["hidden_units"], output_size=2)

    elif model_type == "GRU":
        return GRU(input_size=input_size, hidden_units=kwargs["hidden_units"], output_size=2)

    elif model_type == "Transformer":
        return TransformerModel(input_dim=input_size, num_heads=kwargs.get("num_heads", 8), num_layers=kwargs.get("num_layers", 2), output_size=2)

    else:
        raise ValueError(f"Invalid model type: {model_type}")


# Fully Connected Neural Network (FCNN)
class FCNN(pl.LightningModule):
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
        return optim.Adam(self.parameters(), lr=1e-3)

# Convolutional Neural Network (CNN)
class CNN(pl.LightningModule):
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
        return optim.Adam(self.parameters(), lr=1e-4)

# LSTM-based Model
class LSTM(pl.LightningModule):
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
        return optim.Adam(self.parameters(), lr=1e-3)


# GRU-based Model
class GRU(pl.LightningModule):
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
        return optim.Adam(self.parameters(), lr=1e-3)

# Transformer-based Model
class TransformerModel(pl.LightningModule):
    def __init__(self, input_dim, num_heads=8, num_layers=7, hidden_dim=128, output_size=2):
        super().__init__()

        # Ensure input_dim is divisible by num_heads by padding if necessary
        self.padding = (num_heads - (input_dim % num_heads)) % num_heads
        self.input_dim = input_dim + self.padding

        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,  # Use padded input dimension
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Fully connected layer
        self.fc = nn.Linear(self.input_dim, output_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.dim() == 2:  # Ensure input has the correct shape
            x = x.unsqueeze(1)  # Shape: (batch_size, 1, feature_dim)

        # Pad the input if necessary
        if self.padding > 0:
            x = torch.nn.functional.pad(x, (0, self.padding))

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)

        # Pooling over sequence length and pass through the fully connected layer
        return self.fc(x.mean(dim=1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() > 1:  # Convert one-hot encoded y to class indices
            y = torch.argmax(y, dim=1)

        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
    