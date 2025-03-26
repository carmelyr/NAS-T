import torch
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from model_builder import build_model
from data_handler import get_data_splits
from utils import save_results_csv
import time
import os
import csv

# List of model types to test
MODEL_TYPES = ["LSTM"]

def get_layer_info(model):
    """
    Dynamically extracts the architecture layers from the model, including detailed information
    such as input/output dimensions, number of filters, kernel size, etc.
    """
    layers = []
    for name, module in model.named_modules():
        # Skip the root module (the model itself) and loss functions
        if not name and not isinstance(module, torch.nn.ModuleList):
            continue
        if isinstance(module, torch.nn.CrossEntropyLoss):
            continue

        # Format the layer information based on its type
        if isinstance(module, torch.nn.Linear):
            layer_info = f"{name}: Linear(in_features={module.in_features}, out_features={module.out_features})"
        elif isinstance(module, torch.nn.Conv1d):
            layer_info = f"{name}: Conv1d(in_channels={module.in_channels}, out_channels={module.out_channels}, kernel_size={module.kernel_size})"
        elif isinstance(module, torch.nn.MaxPool1d):
            layer_info = f"{name}: MaxPool1d(kernel_size={module.kernel_size})"
        elif isinstance(module, torch.nn.LSTM):
            layer_info = f"{name}: LSTM(input_size={module.input_size}, hidden_size={module.hidden_size}, num_layers={module.num_layers})"
        elif isinstance(module, torch.nn.GRU):
            layer_info = f"{name}: GRU(input_size={module.input_size}, hidden_size={module.hidden_size}, num_layers={module.num_layers})"
        elif isinstance(module, torch.nn.TransformerEncoder):
            # Extract nhead from the encoder layer's self_attn attribute
            encoder_layer = module.layers[0]  # Get the first encoder layer
            nhead = encoder_layer.self_attn.num_heads  # Access num_heads from self_attn
            layer_info = f"{name}: TransformerEncoder(num_layers={len(module.layers)}, nhead={nhead})"
        elif isinstance(module, torch.nn.ReLU):
            layer_info = f"{name}: ReLU()"
        else:
            layer_info = f"{name}: {module.__class__.__name__}"

        layers.append(layer_info)
    return ", ".join(layers)

def benchmark_models():
    results = []
    run_id = 1  # Simplified run_id for now
    
    for model_type in MODEL_TYPES:
        print(f"Benchmarking model: {model_type}")
        
        total_accuracy = 0.0
        total_time = 0.0
        fold_count = 0
        fold_accuracies = []
        
        for train_loader, val_loader in get_data_splits():
            model = None
            input_size = next(iter(train_loader))[0].shape[-1]
            
            if model_type == "LSTM":
                model = build_model(
                    model_type="LSTM",
                    input_size=input_size,
                    hidden_units=128,
                    output_size=2  # Assuming binary classification
                )
            
            if model is None:
                continue
            
            # Configure trainer with both train and val loaders
            trainer = pl.Trainer(
                max_epochs=100,  # Increased epochs
                enable_checkpointing=True,
                callbacks=[
                    pl.callbacks.EarlyStopping(
                        monitor="val_acc",  # Monitor accuracy instead of loss
                        patience=10,
                        mode="max",
                        min_delta=0.001
                    ),
                    pl.callbacks.ModelCheckpoint(
                        monitor="val_acc",
                        mode="max",
                        save_top_k=1
                    )
                ],
                enable_progress_bar=True,
                logger=True,
                gradient_clip_val=1.0  # Prevent exploding gradients
            )
            
            start_time = time.time()
            trainer.fit(model, train_loader, val_loader)  # Pass both loaders
            elapsed_time = time.time() - start_time
            
            acc = evaluate_model(model, val_loader)
            total_accuracy += acc
            total_time += elapsed_time
            fold_count += 1
            fold_accuracies.append(acc)
        
        if fold_count > 0:
            avg_accuracy = total_accuracy / fold_count
            avg_time = total_time / fold_count
            model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            save_results_csv(
                "benchmark_results.csv",
                run_id,
                1,
                model_type,
                get_layer_info(model),
                fold_accuracies,
                avg_accuracy,
                model_size,
                avg_time
            )
            
            results.append([model_type, avg_accuracy, avg_time, model_size])
    
    results_df = pd.DataFrame(results, columns=["Model Type", "Avg Accuracy", "Avg Time (s)", "Model Size"])
    print(results_df)
    return results_df

def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in val_loader:
            outputs = model(X)
            predicted = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(y, dim=1)  # Convert one-hot encoded y to class indices
            correct += (predicted == true_labels).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    benchmark_models()