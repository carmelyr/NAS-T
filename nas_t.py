import random
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedKFold, train_test_split
import json
import os
import time
import pandas as pd
import pytorch_lightning as pl
import torch.optim as optim

# Model structure and evolutionary parameters
population_size = 20
generations = 20
F = 0.7  # Mutation factor
CR = 0.9  # Crossover rate
alpha = 0.001  # Penalty for model size
BETA = 0.0001  # Penalty for training time

n_splits = 5
n_repeats = 3
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# Load data
X_analysis = pd.read_csv('classification_ozone/X_train.csv')
y_analysis = pd.read_csv('classification_ozone/y_train.csv')
X_test = pd.read_csv('classification_ozone/X_test.csv')
y_test = pd.read_csv('classification_ozone/y_test.csv')

# Split X_analysis and y_analysis into X_train, X_validation, y_train, y_validation
# 80% training, 20% validation
X_train, X_validation, y_train, y_validation = train_test_split(
    X_analysis, y_analysis, test_size=0.2, random_state=42)

# Function to save results in a JSON file
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

# Function to generate a random architecture
def random_architecture():
    return [
        {'layer': 'Conv', 'filters': random.choice([8, 16, 32, 64, 128, 256]), 'kernel_size': random.choice([3, 5, 8]),
         'activation': random.choice(['relu', 'elu', 'selu', 'sigmoid', 'linear'])},
        {'layer': 'MaxPooling', 'pool_size': 3},
        {'layer': 'Dense', 'units': random.choice([4, 16, 32, 64, 128, 256]),
         'activation': random.choice(['relu', 'elu', 'selu', 'sigmoid', 'linear'])},
        {'layer': 'Dropout', 'rate': random.uniform(0, 0.5)}
    ]

class Genotype:
    def __init__(self, architecture=None):
        self.architecture = architecture if architecture else random_architecture()
        self.config = self.architecture
        self.fitness = None

    def evaluate(self):
        self.fitness = fitness_function(self.architecture)
        return self.fitness

    def to_phenotype(self):
        return Phenotype(self.architecture)
    
class Phenotype(pl.LightningModule):
    def __init__(self, genotype):
        super(Phenotype, self).__init__()
        self.genotype = genotype
        self.model = self.build_model_from_genotype(genotype)
        self.loss_fn = nn.CrossEntropyLoss()

    def build_model_from_genotype(self, genotype):
        layers = []
        input_channels = 1  # Assuming 1 input channel (e.g., grayscale image)

        for layer in genotype:
            if layer['layer'] == 'Conv':
                layers.append(nn.Conv2d(input_channels, layer['filters'], layer['kernel_size']))
                input_channels = layer['filters']
                
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
                layers.append(nn.MaxPool2d(layer['pool_size']))

            elif layer['layer'] == 'Dense':
                layers.append(nn.Flatten())  # Flatten before Dense layer
                layers.append(nn.Linear(input_channels, layer['units']))
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


def fitness_function(architecture):
    accuracy = random.uniform(0.4, 0.9)
    model_size = sum(layer.get('filters', 0) + layer.get('units', 0) for layer in architecture)
    training_time = 0.1 * model_size
    return accuracy - alpha * model_size - BETA * training_time

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

    def evolve(self):
        run_results = {"run_id": 1, "generations": []}
        best_overall_fitness = float('-inf')
        best_overall_individual = None
        best_generation = -1

        for generation in range(self.generations):
            start_time = time.perf_counter()  # Use higher precision timer
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

            end_time = time.perf_counter()  # Higher precision timer
            if self.verbose:
                print(f"Runtime for generation {generation + 1}: {end_time - start_time:.6f} seconds\n")

        if self.verbose:
            print("Best overall architecture:")
            print(f"Found in generation {best_generation}")
            print(best_overall_individual.architecture)
            print("Fitness:", best_overall_fitness)

        save_run_results_json("evolutionary_runs.json", run_results)

if __name__ == "__main__":
    nas_de = NASDifferentialEvolution(verbose=True)
    nas_de.evolve()
