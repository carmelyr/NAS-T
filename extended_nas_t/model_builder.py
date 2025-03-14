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

    # Transformer is disabled
    elif model_type == "Transformer":
        return TransformerModel(input_dim=input_size, num_heads=kwargs.get("num_heads", 8), num_layers=kwargs.get("num_layers", 2), output_size=2)

    else:
        raise ValueError(f"Invalid model type: {model_type}")


# Fully Connected Neural Network (FCNN)
class FCNN(pl.LightningModule):
    def __init__(self, input_size, hidden_units=64, output_size=2, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_units))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_units, output_size))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() > 1:  # Convert one-hot encoded y to class indices
            y = torch.argmax(y, dim=1)

        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

# Convolutional Neural Network (CNN)
class CNN(pl.LightningModule):
    def __init__(self, input_channels, num_filters=32, kernel_size=3, output_size=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size)
        self.pool = nn.MaxPool1d(2)
        self.fc = None  # Placeholder for the fully connected layer
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.dim() == 2:  # Ensure CNN expects 3D input
            x = x.unsqueeze(1)  # Shape: (batch_size, 1, time_steps)

        # Apply Conv & Pooling
        x = self.pool(torch.relu(self.conv1(x)))  # Shape: (batch_size, num_filters, time_steps // 2)
        x = self.pool(torch.relu(self.conv2(x)))  # Shape: (batch_size, num_filters * 2, time_steps // 4)

        # Flatten the output
        x = x.view(x.size(0), -1)  # Shape: (batch_size, num_filters * 2 * (time_steps // 4))

        # Dynamically initialize FC layer on first forward pass
        if self.fc is None:
            self.flattened_size = x.shape[1]  # Dynamically get correct input size
            self.fc = nn.Linear(self.flattened_size, 2).to(x.device)  # Reinitialize on the correct device
        
        return self.fc(x)  # Fully connected layer

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
    def __init__(self, input_size, hidden_units=64, output_size=2, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_units, num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_units, output_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.dim() == 2:  # Ensure 3D input for LSTM
            x = x.unsqueeze(1)  # Shape: (batch_size, time_steps=1, feature_dim)

        _, (h_n, _) = self.lstm(x)  # Get the last hidden state
        h_n = h_n[-1]  # Extract the last layer's hidden state

        output = self.fc(h_n)  # Fully connected layer

        if output.dim() == 1:  # Ensure correct shape for classification
            output = output.unsqueeze(0)  # Make sure it's at least (batch_size, num_classes)
        
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
        return optim.Adam(self.parameters(), lr=1e-4)


# GRU-based Model
class GRU(pl.LightningModule):
    def __init__(self, input_size, hidden_units, output_size=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # Reshape input to ensure it is (batch_size, sequence_length, feature_dim)
        if x.dim() == 4:  # If it's 4D, squeeze the second dimension
            x = x.squeeze(1)  # Reduce from (batch, 1, seq_len, feat_dim) -> (batch, seq_len, feat_dim)
        elif x.dim() == 2:  # If it's 2D, unsqueeze to create a sequence length of 1
            x = x.unsqueeze(1)

        _, h_n = self.gru(x)  # Get hidden state
        return self.fc(h_n[-1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() > 1:  # Convert one-hot encoded y to class indices
            y = torch.argmax(y, dim=1)

        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

# Transformer-based Model
class TransformerModel(pl.LightningModule):
    def __init__(self, input_dim, num_heads=8, num_layers=4, hidden_dim=128, output_size=2):
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
    