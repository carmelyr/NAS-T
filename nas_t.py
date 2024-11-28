import random
import copy
import numpy as np
import torch
import torch.nn as nn       # neural network module
from sklearn.model_selection import RepeatedKFold, train_test_split
import json
import os
import time
import pandas as pd     # data manipulation and analysis library
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset      # for creating (TensorDataset) and loading (DataLoader) data
from pytorch_lightning import Trainer       # for training models


# ---- Parameters ---- #
population_size = 20
generations = 20
F = 0.7             # mutation factor for the evolutionary algorithm
CR = 0.9            # crossover rate for the evolutionary algorithm
alpha = 0.001       # penalty for model size
BETA = 0.0001       # penalty for training time


# ---- Parameters for repeated k-fold cross-validation ---- #
# n_folds = 5       # number of folds
# n_repeats = 3     # number of repeats
# rkf = RepeatedKFold(n_folds=n_folds, n_repeats=n_repeats, random_state=42)


# ---- Load data from the classification_ozone dataset ---- #
"""
- X_analysis contains the feature (input) data for training;
    each row is training example and each column is a feature
- y_analysis contains the target (output) data for training;
    each row is the target for a training example
"""
X_analysis = pd.read_csv('classification_ozone/X_train.csv')
y_analysis = pd.read_csv('classification_ozone/y_train.csv')
X_test = pd.read_csv('classification_ozone/X_test.csv')
y_test = pd.read_csv('classification_ozone/y_test.csv')


# ---- Split data into training and validation sets ---- #
# 80% training, 20% validation
X_train, X_validation, y_train, y_validation = train_test_split(X_analysis, y_analysis, test_size=0.2, random_state=42)


# ---- Convert data to PyTorch tensors for training ---- #
"""
- X_train_tensor stores input data as a 32-bit float tensor for precise calculations (training data)
- y_train_tensor stores target data as a long tensor for categorical classification (training data)
"""
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)      
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_validation_tensor = torch.tensor(X_validation.values, dtype=torch.float32)
y_validation_tensor = torch.tensor(y_validation.values, dtype=torch.long)


# ---- Create TensorDataset for training and validation sets ---- #
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)


# ---- Create DataLoader for training and validation sets ---- #
"""
- data will be loaded in batches of 32 samples
- shuffle=True shuffles the data after each epoch (improves generalization by preventing overfitting) 
"""
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32)

# ---- Save run results to a JSON file ---- #
"""
- filename: name of the JSON file that saves the run results
- run_results: dictionary that contains where to save the run results
"""
def save_run_results_json(filename, run_results):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if data:
        latest_run_id = max(run['run_id'] for run in data)
        run_results['run_id'] = latest_run_id + 1
    else:
        run_results['run_id'] = 1

    data.append(run_results)

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


# ---- Define the random architecture for the neural network based on the values of the scientific paper---- #
"""
- Publication of the paper: https://ieeexplore.ieee.org/document/9206721 (Neaural Architecture Search for Time Series Classification)

- Conv layer detects patterns in the input data by applying filters to the input
- ZeroOp layer skips or ignores the next layer
- MaxPooling layer reduces the size of the data by taking only the maximum value in a window
- Dense layer connects all input neurons to all output neurons
- Dropout layer randomly sets a fraction of input units to zero to prevent overfitting
- Activation layer applies an activation function to the output of the previous layer
"""
def random_architecture():
    return [
        {'layer': 'Conv', 'filters': random.choice([8, 16, 32, 64, 128, 256]), 'kernel_size': random.choice([3, 5, 8]),
         'activation': random.choice(['relu', 'elu', 'selu', 'sigmoid', 'linear'])},
        {'layer': 'ZeroOp'},
        {'layer': 'MaxPooling', 'pool_size': 3},
        {'layer': 'Dense', 'units': random.choice([4, 16, 32, 64, 128, 256]),
         'activation': random.choice(['relu', 'elu', 'selu', 'sigmoid', 'linear'])},
        {'layer': 'Dropout', 'rate': random.uniform(0, 0.5)},
        {'layer': 'Activation', 'activation': random.choice(['softmax', 'elu', 'selu', 'relu', 'sigmoid', 'linear'])}
    ]


# ---- Genotype class ---- #
# Stores and manipulates the architecture of the neural network
class Genotype:
    """
    - constructor for the Genotype class
    - called when a new instance of the class is created
    - self.config: stores the same architecture of the genotype as defined in self.architecture
    - self.fitness: stores the fitness value of the genotype
    """
    def __init__(self, architecture=None):
        self.architecture = architecture if architecture else random_architecture()
        self.config = self.architecture
        self.fitness = None

    """
    - method that calculates the fitness score of the architecture based on its performance
    """
    def evaluate(self, validation_accuracy=None):
        if validation_accuracy is None:
            # currently assigning a random validation accuracy during the initial evaluation
            validation_accuracy = random.uniform(0.4, 0.9)
        self.fitness = fitness_function(self.architecture, validation_accuracy)
        return self.fitness

    """
    - method that converts the genotype to a phenotype
    """
    def to_phenotype(self):
        return Phenotype(self.architecture)
    
class Phenotype(pl.LightningModule):
    def __init__(self, genotype=None):
        super(Phenotype, self).__init__()
        if genotype:
            self.genotype = genotype
            self.model = self.build_model_from_genotype(genotype)
        self.loss_fn = nn.CrossEntropyLoss()

    def build_model_from_genotype(self, genotype):
        layers = []
        input_channels = 1  # Assuming 1 input channel (e.g., grayscale image)
        output_size = 1201  # Assuming 1201 features in the input data

        for layer in genotype:
            if layer['layer'] == 'Conv':
                layers.append(nn.Conv1d(input_channels, layer['filters'], layer['kernel_size']))
                input_channels = layer['filters']
                # Update the output size after Conv1d (reduce due to kernel size)
                output_size = output_size - layer['kernel_size'] + 1
                
                # Add activation function
                if layer['activation'] == 'relu':
                    layers.append(nn.ReLU())
                elif layer['activation'] == 'elu':
                    layers.append(nn.ELU())
                elif layer['activation'] == 'selu':
                    layers.append(nn.SELU())
                elif layer['activation'] == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif layer['activation'] == 'linear':
                    layers.append(nn.Identity())

            elif layer['layer'] == 'MaxPooling':
                layers.append(nn.MaxPool1d(layer['pool_size']))
                output_size = output_size // layer['pool_size']

            elif layer['layer'] == 'Dense':
                layers.append(nn.Flatten())
                layers.append(nn.Linear(input_channels * output_size, layer['units']))
                input_channels = layer['units']
                
                # Add activation function
                if layer['activation'] == 'relu':
                    layers.append(nn.ReLU())
                elif layer['activation'] == 'elu':
                    layers.append(nn.ELU())
                elif layer['activation'] == 'selu':
                    layers.append(nn.SELU())
                elif layer['activation'] == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif layer['activation'] == 'linear':
                    layers.append(nn.Identity())

            elif layer['layer'] == 'Dropout':
                layers.append(nn.Dropout(layer['rate']))

        return nn.Sequential(*layers)
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, genotype, **kwargs):
        instance = cls(genotype=genotype, **kwargs)
        checkpoint = torch.load(checkpoint_path)
        instance.load_state_dict(checkpoint['state_dict'], strict=False)
        return instance


    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        y = y.view(-1)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        y = y.view(-1)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


def fitness_function(architecture, validation_accuracy):
    model_size = sum(layer.get('filters', 0) + layer.get('units', 0) for layer in architecture)
    training_time = 0.1 * model_size
    accuracy_component = validation_accuracy
    size_penalty = alpha * model_size
    time_penalty = BETA * training_time
    fitness = accuracy_component - size_penalty - time_penalty
    return max(0.0, fitness)

class NASDifferentialEvolution:
    def __init__(self, population_size=population_size, generations=generations, verbose=True):
        self.population_size = population_size
        self.generations = generations
        self.population = self.initialize_population()
        self.verbose = verbose

    def initialize_population(self):
        return [Genotype() for _ in range(self.population_size)]

    def mutate(self, parent1, parent2, parent3):
        mutant = copy.deepcopy(parent1.architecture)
        filter_options = [8, 16, 32, 64, 128, 256]

        for i in range(len(mutant)):
            if random.random() < F:
                if 'filters' in mutant[i]:
                    mutated_filter = int(parent1.architecture[i]['filters'] + F * (
                                parent2.architecture[i]['filters'] - parent3.architecture[i]['filters']))
                    mutant[i]['filters'] = min(filter_options, key=lambda x: abs(x - mutated_filter))
                if 'units' in mutant[i]:
                    mutant[i]['units'] = max(16, min(64, int(
                        parent1.architecture[i]['units'] + F * (
                                    parent2.architecture[i]['units'] - parent3.architecture[i]['units']))))
                if 'rate' in mutant[i]:
                    mutant[i]['rate'] = max(0.05, min(0.5, parent1.architecture[i]['rate'] + F * (
                                parent2.architecture[i]['rate'] - parent3.architecture[i]['rate'])))
        return Genotype(mutant)

    def crossover(self, parent, mutant):
        offspring_architecture = copy.deepcopy(parent.architecture)
        for i in range(len(offspring_architecture)):
            if random.random() < CR:
                offspring_architecture[i] = mutant.architecture[i]
        return Genotype(offspring_architecture)
    
    def train_and_save_phenotype(self, phenotype, train_loader, validation_loader, save_path):
        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(phenotype, train_dataloaders=train_loader, val_dataloaders=validation_loader)
        trainer.save_checkpoint(save_path)

    def load_and_evaluate_phenotype(self, phenotype_path, genotype, validation_loader):
        phenotype = Phenotype.load_from_checkpoint(phenotype_path, genotype=genotype)
        phenotype.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for batch in validation_loader:
                x, y = batch
                logits = phenotype(x)
                predictions = logits.argmax(dim=1)
                correct += (predictions == y.view(-1)).sum().item()
                total += y.size(0)

        validation_accuracy = correct / total
        return validation_accuracy


    def evolve(self):
        run_results = {"run_id": 1, "generations": []}
        best_overall_fitness = float('-inf')
        best_overall_individual = None
        best_generation = -1

        for generation in range(self.generations):
            start_time = time.perf_counter()
            if self.verbose:
                print(f"Generation {generation + 1}")

            new_population = []
            generation_fitnesses = []

            for i in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)
                parent1, parent2, parent3 = [self.population[idx] for idx in random.sample(candidates, 3)]
                mutant = self.mutate(parent1, parent2, parent3)
                offspring = self.crossover(self.population[i], mutant)

                parent_fitness = self.population[i].evaluate()
                offspring_fitness = offspring.evaluate()

                if offspring_fitness > parent_fitness:
                    new_population.append(offspring)
                    generation_fitnesses.append(offspring_fitness)
                else:
                    new_population.append(self.population[i])
                    generation_fitnesses.append(parent_fitness)

            self.population = new_population
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            best_fitness = best_individual.fitness

            if self.verbose:
                print(f"Best fitness in generation {generation + 1}: {best_fitness}")

            if best_fitness > best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_individual = best_individual
                best_generation = generation + 1

            generation_result = {
                "generation": generation + 1,
                "best_fitness": best_fitness,
                "best_architecture": best_individual.architecture,
                "all_fitnesses": generation_fitnesses
            }
            run_results["generations"].append(generation_result)

            end_time = time.perf_counter()
            if self.verbose:
                print(f"Runtime for generation {generation + 1}: {end_time - start_time:.6f} seconds\n")

        if self.verbose:
            print("Best overall architecture:")
            print(f"Found in generation {best_generation}")
            print(best_overall_individual.architecture)
            print("Fitness:", best_overall_fitness, "\n")

        # Train the best overall individual and save the model
        if best_overall_individual:
            phenotype = best_overall_individual.to_phenotype()
            checkpoint_path = f"best_model_gen_{best_generation}.ckpt"
            self.train_and_save_phenotype(phenotype, train_loader, validation_loader, save_path=checkpoint_path)
            validation_accuracy = self.load_and_evaluate_phenotype(checkpoint_path, best_overall_individual.architecture, validation_loader)
            best_overall_individual.evaluate(validation_accuracy)

        save_run_results_json("evolutionary_runs.json", run_results)

if __name__ == "__main__":
    nas_de = NASDifferentialEvolution(verbose=True)
    nas_de.evolve()
