import torch
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import RepeatedKFold
from torch.utils.data import DataLoader, TensorDataset
from model_builder import build_model
from utils import fitness_function, save_results_csv
import random
import time
import traceback
import pandas as pd

class NASDifferentialEvolution:
    def __init__(self, population_size=10, generations=5, verbose=True):
        self.population_size = population_size
        self.generations = generations
        self.population = self.initialize_population()
        self.verbose = verbose

    def initialize_population(self):
        # Initialize with Transformer models but with different hyperparameters
        population = []
        for _ in range(self.population_size):
            population.append({
            "model_type": "LSTM",
            "hidden_units": random.choice([64, 128, 256]),
            "num_layers": random.choice([1, 2, 3])
            })
        return population

    def mutate(self, parent1, parent2, parent3, F):
        mutant = {
            "model_type": "LSTM",
            "hidden_units": random.choice([parent1["hidden_units"], parent2["hidden_units"], parent3["hidden_units"]]),
            "num_layers": random.choice([parent1["num_layers"], parent2["num_layers"], parent3["num_layers"]])
        }
        return mutant

    def crossover(self, parent, mutant, CR):
        offspring = {
            "model_type": "LSTM",
            "hidden_units": parent["hidden_units"] if random.random() < CR else mutant["hidden_units"],
            "num_layers": parent["num_layers"] if random.random() < CR else mutant["num_layers"]
        }
        return offspring
    
    def evaluate_model(self, model, val_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X)
                predicted = torch.argmax(outputs, dim=1)
                true_labels = y
                correct += (predicted == true_labels).sum().item()
                total += y.size(0)
                
                # Debug print
                print("Predictions:", predicted[:5].cpu().numpy())
                print("True labels:", true_labels[:5].cpu().numpy())
                
        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")
        return accuracy

    def cross_validate(self, model_config, X, y, input_size, generation, num_folds=5, num_repeats=1):
        rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=42)
        scores = []
        model_sizes = []
        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(rkf.split(X)):
            # Get the data splits
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Convert labels to class indices if one-hot encoded
            if y_train.ndim > 1 and y_train.shape[1] > 1:
                y_train = np.argmax(y_train, axis=1)
                y_val = np.argmax(y_val, axis=1)
            
            # Reshape data for Transformer
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # Add sequence dimension
            X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            
            # Convert to tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Use long for CrossEntropyLoss
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            
            # Create datasets and loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

            # Build the model
            try:
                model = build_model(
                    model_config["model_type"],
                    input_size=X_train.shape[-1],
                    num_heads=model_config["num_heads"],
                    num_layers=model_config["num_layers"],
                    hidden_dim=model_config["hidden_dim"],
                    output_size=len(np.unique(y_train))  # Number of classes
                )
                
                # Configure trainer
                trainer = pl.Trainer(
                    max_epochs=30,
                    enable_checkpointing=False,
                    callbacks=[
                        pl.callbacks.EarlyStopping(
                            monitor="val_loss",
                            patience=5,
                            mode="min",
                            min_delta=0.01
                        )
                    ],
                    logger=True,
                    enable_progress_bar=True,
                    enable_model_summary=True
                )
                
                trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                val_acc = self.evaluate_model(model, val_loader)
                model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
                fitness = fitness_function(model_config["model_type"], val_acc, model_size)
                print(f"Fold {fold + 1}: Val Acc = {val_acc:.4f}, Fitness = {fitness:.4f}")

            except Exception as e:
                print(f"Error in fold {fold + 1}: {str(e)}")
                val_acc, model_size, fitness = 0, 0, 0
            
            scores.append(fitness)
            model_sizes.append(model_size)
            fold_accuracies.append(val_acc)
        
        if np.mean(scores) > 0:
            save_results_csv(
                "evolution_results.csv",
                1,  # run_id
                generation + 1,
                model_config["model_type"],
                str(model_config),
                fold_accuracies,
                np.mean(fold_accuracies),
                np.mean(model_sizes),
                time.time()
            )
        
        return np.mean(scores), np.mean(model_sizes)

    def evolve_and_check(self, X, y, input_size):
        generation = 0
        failed_configs = set()
        while generation < self.generations:
            new_population = []
            for i in range(self.population_size):
                parent1, parent2, parent3 = random.sample(self.population, 3)
                mutant = self.mutate(parent1, parent2, parent3, 0.6)
                offspring = self.crossover(self.population[i], mutant, 0.7)
                
                try:
                    fitness, model_size = self.cross_validate(offspring, X, y, input_size, generation)
                    new_population.append((offspring, fitness, model_size))
                    print(f"Generation {generation + 1}: Model {offspring} succeeded with fitness {fitness}")
                except Exception as e:
                    print(f"Generation {generation + 1}: Model {offspring} failed. Evolving again...")
                    traceback.print_exc()
                    failed_configs.add(str(offspring))
                    continue
            
            if new_population:
                self.population = [x[0] for x in sorted(new_population, key=lambda x: x[1], reverse=True)]
                
                if self.verbose:
                    print(f"Generation {generation + 1} Best Model: {self.population[0]}")
            else:
                print(f"Generation {generation + 1}: All models failed. Skipping to the next generation.")
            
            generation += 1

if __name__ == "__main__":
    from data_handler import get_data_splits
    import numpy as np

    X_analysis = pd.read_csv('classification_ozone/X_train.csv')
    y_analysis = pd.read_csv('classification_ozone/y_train.csv')

    nas = NASDifferentialEvolution(population_size=10, generations=5, verbose=True)
    
    nas.evolve_and_check(X_analysis, y_analysis, input_size=10)