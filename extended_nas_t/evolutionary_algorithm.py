import torch
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from model_builder import build_model
from utils import fitness_function, save_results_csv
from config import initial_F, final_F, initial_CR, final_CR, decay_rate
import random
import time
import traceback
import pandas as pd
from tqdm import tqdm
import sys
import csv
import os

class NASDifferentialEvolution:
    def __init__(self, population_size=8, generations=3, verbose=True):
        self.population_size = population_size
        self.generations = generations
        self.population = self.initialize_population()
        self.best_fitness = -float('inf')
        self.best_model = None
        self.best_accuracy = 0.0
        self.verbose = verbose
        self.history = []
        self.initial_F = initial_F
        self.final_F = final_F
        self.initial_CR = initial_CR
        self.final_CR = final_CR
        self.decay_rate = decay_rate
        self.total_generations = generations
        self.run_id = self.get_next_run_id("evolution_results.csv")

    def get_current_rates(self, generation):
        """Calculate dynamic rates using hybrid strategy"""
        # Hybrid mutation factor calculation
        if generation < self.total_generations // 2:
            # Exponential decay for first half
            current_F = max(
                self.final_F,
                self.initial_F * (self.decay_rate ** generation)
            )
        else:
            # Linear decay for second half
            linear_progress = (generation - self.total_generations//2) / (self.total_generations//2)
            current_F = self.initial_F - (self.initial_F - self.final_F) * linear_progress

        # Linear crossover rate decay
        current_CR = self.initial_CR - (self.initial_CR - self.final_CR) * (generation / self.total_generations)
        
        return current_F, current_CR

    def get_next_run_id(self, results_file):
        """
        Reads the last run_id from the results file and increments it.
        If the file does not exist or is empty, starts with run_id = 1.
        """
        if not os.path.exists(results_file):
            return 1
        try:
            with open(results_file, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) <= 1:  # Only header or empty file
                    return 1
                last_run_id = int(rows[-1][0])  # First column is run_id
                return last_run_id + 1
        except Exception as e:
            print(f"Error reading run_id from {results_file}: {e}")
            return 1

    def build_model(self, model_config, X_train, y_train):
        return build_model(
            model_type=model_config["model_type"],
            input_size=X_train.shape[-1],
            hidden_units=int(model_config["hidden_units"]),
            output_size=y_train.shape[1],
            num_layers=int(model_config["num_layers"]),
            dropout_rate=model_config["dropout_rate"],
            bidirectional=model_config["bidirectional"],
            attention=model_config["attention"],
            learning_rate=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"]
        )

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            population.append({
                "model_type": "LSTM",
                "hidden_units": random.choice([64, 128, 256, 512]),
                "num_layers": random.choice([1, 2, 3, 4]),       
                "dropout_rate": random.uniform(0.1, 0.5),      
                "bidirectional": random.choice([True, False]),
                "attention": True,
                "learning_rate": random.choice([1e-5, 1e-4, 1e-3, 3e-4, 5e-5]), 
                "batch_size": random.choice([32, 64, 128]),
                "weight_decay": random.choice([0, 1e-5, 5e-5]) 
            })
        return population

    def mutate(self, parent1, parent2, parent3, F=0.7):
        mutant = {
            "model_type": "LSTM",
            "hidden_units": max(64, min(512, int(parent1["hidden_units"] + F * (parent2["hidden_units"] - parent3["hidden_units"])))),
            "num_layers": max(1, min(4, int(round(parent1["num_layers"] + F * (parent2["num_layers"] - parent3["num_layers"]))))),
            "dropout_rate": max(0.1, min(0.5, parent1["dropout_rate"] + F * (parent2["dropout_rate"] - parent3["dropout_rate"]))),
            "bidirectional": random.choice([parent1["bidirectional"], parent2["bidirectional"], parent3["bidirectional"]]),
            "attention": True,
            "learning_rate": 10**(np.log10(parent1["learning_rate"]) + F * (np.log10(parent2["learning_rate"]) - np.log10(parent3["learning_rate"]))),
            "batch_size": random.choice([parent1["batch_size"], parent2["batch_size"], parent3["batch_size"]]),
            "weight_decay": random.choice([parent1["weight_decay"], parent2["weight_decay"], parent3["weight_decay"]])
        }
        return mutant

    def crossover(self, parent, mutant, CR=0.85):
        offspring = parent.copy()
        for key in mutant:
            if random.random() < CR:
                offspring[key] = mutant[key]
        return offspring
    
    def evaluate_model(self, model, val_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X)
                predicted = torch.argmax(outputs, dim=1)
                true_labels = torch.argmax(y, dim=1)
                correct += (predicted == true_labels).sum().item()
                total += y.size(0)
        accuracy = correct / total
        return accuracy

    def cross_validate(self, model_config, X, y, input_size, generation):
        from pytorch_lightning.callbacks import EarlyStopping
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        fold_accuracies = []
        model_sizes = []
        training_times = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n--- Fold {fold + 1} ---")
            try:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

                train_loader = DataLoader(train_dataset, batch_size=model_config["batch_size"], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=model_config["batch_size"], shuffle=False)

                model = self.build_model(model_config, X_train, y_train)

                trainer = pl.Trainer(
                    max_epochs=150,
                    enable_checkpointing=False,
                    callbacks=[
                        EarlyStopping(
                            monitor="val_acc",
                            patience=30,
                            mode="max",
                            min_delta=0.001,
                            stopping_threshold=0.9
                        )
                    ],
                    enable_progress_bar=False,
                    logger=False
                )

                start_time = time.time()
                trainer.fit(model, train_loader, val_loader)
                training_time = time.time() - start_time

                val_acc = self.evaluate_model(model, val_loader)
                model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
                fitness = fitness_function("LSTM", val_acc, model_size, training_time)

                print(f"Fold {fold+1} Accuracy: {val_acc:.4f}, Fitness: {fitness:.4f}, Size: {model_size}, Time: {training_time:.2f}s")

                scores.append(fitness)
                fold_accuracies.append(val_acc)
                model_sizes.append(model_size)
                training_times.append(training_time)

            except Exception as e:
                print(f"Error in fold {fold + 1}: {str(e)}")
                traceback.print_exc()
                return -float('inf'), 0.0, 0, 0.0

        avg_fitness = np.mean(scores)
        avg_accuracy = np.mean(fold_accuracies)
        avg_model_size = np.mean(model_sizes)
        avg_training_time = np.mean(training_times)

        print(f"\n>>> Generation {generation+1}: Avg Accuracy = {avg_accuracy:.4f}, Avg Fitness = {avg_fitness:.4f}\n")

        if avg_accuracy > self.best_accuracy:
            self.best_accuracy = avg_accuracy
            self.best_model = model_config
            self.best_fitness = avg_fitness

        clean_config = model_config.copy()
        if 'fitness' in clean_config:
            del clean_config['fitness']
        if 'accuracy' in clean_config:
            del clean_config['accuracy']

        save_results_csv(
            "evolution_results.csv",
            self.run_id,
            generation + 1,
            "LSTM",
            str(clean_config),
            fold_accuracies,
            avg_accuracy,
            avg_model_size,
            avg_training_time
        )

        return avg_fitness, avg_accuracy, avg_model_size, avg_training_time

    def evolve_and_check(self, X, y, input_size):
        for generation in tqdm(range(self.generations), desc="Evolution Progress", file=sys.stdout, dynamic_ncols=True):
            new_population = []
            for i in range(self.population_size):
                # Select 3 distinct parents
                candidates = [idx for idx in range(self.population_size) if idx != i]
                parent1_idx, parent2_idx, parent3_idx = random.sample(candidates, 3)
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                parent3 = self.population[parent3_idx]
                
                mutant = self.mutate(parent1, parent2, parent3)
                offspring = self.crossover(self.population[i], mutant)
                
                fitness, accuracy, model_size, training_time = self.cross_validate(
                    offspring, X, y, input_size, generation)
                
                if fitness > self.population[i].get('fitness', -float('inf')):
                    offspring['fitness'] = fitness
                    offspring['accuracy'] = accuracy
                    new_population.append(offspring)
                else:
                    new_population.append(self.population[i])

            self.population = sorted(new_population, key=lambda x: x['fitness'], reverse=True)
            self.history.append({
                'generation': generation + 1,
                'best_fitness': self.population[0]['fitness'],
                'best_accuracy': self.population[0]['accuracy'],
                'best_model': self.population[0]
            })

            if self.verbose:
                print(f"\nGeneration {generation + 1} Best:")
                print(f"Fitness: {self.population[0]['fitness']:.4f}")
                print(f"Accuracy: {self.population[0]['accuracy']:.4f}")
                print(f"Model: {self.population[0]}\n")

            # Early stopping if we reach target accuracy
            if self.population[0]['accuracy'] >= 0.9:
                print(f"\nTarget accuracy of 0.9 reached at generation {generation + 1}!")
                break

        return self.population[0]