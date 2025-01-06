import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
import json
import os

class StandardArchitecture(pl.LightningModule):
    def __init__(self, input_size):
        super(StandardArchitecture, self).__init__()

        # Calculate output sizes
        conv1_output = ((input_size - 7) // 2) + 1  # Conv1 with smaller kernel
        pool1_output = conv1_output // 2  # Pool1

        conv2_output = ((pool1_output - 7) // 2) + 1  # Conv2 with smaller kernel
        pool2_output = conv2_output // 2  # Pool2

        flattened_size = pool2_output * 64  # Reduce number of filters

        # Define the architecture
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2),  # Fewer filters, smaller kernel
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=7, stride=2),  # Fewer filters, smaller kernel
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(flattened_size, 128),  # Smaller dense layer
            nn.ReLU(),
            nn.Dropout(0.8),  # Slightly lower dropout

            nn.Linear(128, 64),  # Smaller additional dense layer
            nn.ReLU(),
            nn.Dropout(0.8),

            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # Define the loss function and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='binary', num_classes=2)
        self.accuracies = []
        self.model_sizes = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), 1, x.size(1))
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(preds.argmax(dim=1), y)
        self.log('train_loss', loss, prog_bar=False)
        self.log('train_acc', acc, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), 1, x.size(1))
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(preds.argmax(dim=1), y)
        self.accuracies.append(acc.item())
        self.log('val_loss', loss, prog_bar=False)
        self.log('val_acc', acc, prog_bar=False)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def on_train_epoch_end(self):
        model_size = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.model_sizes.append(model_size)

    def save_results(self, filename):
        # Save only the last recorded accuracy and model size
        results = {
            "final_accuracy": self.accuracies[-1] if self.accuracies else None,
            "final_model_size": self.model_sizes[-1] if self.model_sizes else None
        }
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
