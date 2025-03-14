import torch
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import RepeatedKFold
from torch.utils.data import DataLoader, TensorDataset
from model_builder import build_model
from utils import fitness_function, save_results_csv
import random
import time

class NASDifferentialEvolution:
    def __init__(self, population_size=10, generations=5, verbose=True):
        self.population_size = population_size
        self.generations = generations
        self.population = self.initialize_population()
        self.verbose = verbose

    def initialize_population(self):
        return [random.choice(["FCNN", "CNN", "LSTM", "GRU", "Transformer"]) for _ in range(self.population_size)]

    def mutate(self, parent1, parent2, parent3, F):
        return random.choice([parent1, parent2, parent3])

    def crossover(self, parent, mutant, CR):
        return mutant if random.random() < CR else parent

    def cross_validate(self, model_type, X, y, input_size, num_folds=5, num_repeats=3, **kwargs):
        rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=42)
        scores = []
        model_sizes = []
        sequence_length = 1 if len(X.shape) == 2 else X.shape[1]

        for train_idx, val_idx in rkf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Ensure correct shape for sequence models
            X_train = X_train.reshape(X_train.shape[0], sequence_length, input_size)
            X_val = X_val.reshape(X_val.shape[0], sequence_length, input_size)

            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                          torch.tensor(y_train, dtype=torch.long))
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                        torch.tensor(y_val, dtype=torch.long))

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)

            model = build_model(model_type, input_size=input_size, hidden_units=64, output_size=len(set(y)), **kwargs)
            trainer = pl.Trainer(max_epochs=10, enable_checkpointing=False, strategy="auto", devices=1)
            try:
                trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                val_acc = trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()
                model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
                fitness = fitness_function(model_type, val_acc, model_size)
            except Exception as e:
                print(f"Model {model_type} training failed: {e}")
                val_acc, model_size, fitness = 0, 0, 0  # Penalize failed training runs
            
            scores.append(fitness)
            model_sizes.append(model_size)
        
        return np.mean(scores), np.mean(model_sizes)

    def evolve(self, X, y, input_size):
        for generation in range(self.generations):
            new_population = []
            for i in range(self.population_size):
                parent1, parent2, parent3 = random.sample(self.population, 3)
                mutant = self.mutate(parent1, parent2, parent3, 0.6)
                offspring = self.crossover(self.population[i], mutant, 0.7)
                fitness, model_size = self.cross_validate(offspring, X, y, input_size)
                new_population.append((offspring, fitness, model_size))
                save_results_csv("results.csv", 1, generation + 1, offspring, len(offspring), fitness, model_size, time.time())
            
            self.population = [x[0] for x in sorted(new_population, key=lambda x: x[1], reverse=True)]
            
            if self.verbose:
                print(f"Generation {generation + 1} Best Model: {self.population[0]}")