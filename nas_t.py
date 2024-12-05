import random
import copy
import numpy as np
import torch
import torch.nn as nn                                                           # neural network module
from sklearn.model_selection import RepeatedKFold, train_test_split
import json
import os
import time
import pandas as pd                                                             # data manipulation and analysis library
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset                          # for creating (TensorDataset) and loading (DataLoader) data
from pytorch_lightning.callbacks.early_stopping import EarlyStopping            # stops training when a monitored quantity has stopped improving
import torchmetrics
#from torchmetrics import Metric
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler    # for scaling the data


# ---- Parameters ---- #
population_size = 30
generations = 15
F = 0.85             # mutation factor for the evolutionary algorithm
CR = 0.9             # crossover rate for the evolutionary algorithm
alpha = 0.0001       # penalty for model size
BETA = 0.00001       # penalty for training time


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

# ---- Fill NaNs with column means ---- #
X_analysis.fillna(X_analysis.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# ---- Split data into training and validation sets ---- #
# 80% training, 20% validation
X_train, X_validation, y_train, y_validation = train_test_split(X_analysis, y_analysis, test_size=0.2, random_state=42)

# ---- Scale the data ---- #
scaler = StandardScaler()       # standardizes features by removing the mean and scaling to unit variance
#scaler = MinMaxScaler()        # normalizes to [0, 1]
#scaler = RobustScaler()        # less sensitive to outliers


"""
- X_train: scales the training data
- X_validation: scales the validation data
- X_test: scales the test data
"""
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)

# ---- Convert data to PyTorch tensors for training ---- #
"""
- X_train_tensor stores input data as a 32-bit float tensor for precise calculations (training data)
- y_train_tensor stores target data as a long tensor for categorical classification (training data)
- X_validation_tensor stores input data as a 32-bit float tensor for precise calculations (validation data)
- y_validation_tensor stores target data as a long tensor for categorical classification (validation data)
"""
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy().flatten(), dtype=torch.long)
X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
y_validation_tensor = torch.tensor(y_validation.to_numpy().flatten(), dtype=torch.long)

# ---- Normalize input data ---- #
mean = X_train_tensor.mean(dim=0)
std = X_train_tensor.std(dim=0)
X_train_tensor = (X_train_tensor - mean) / std
X_validation_tensor = (X_validation_tensor - mean) / std

# ---- Check for NaNs in input data ---- #
assert not torch.isnan(X_train_tensor).any(), "NaN found in training data"
assert not torch.isnan(X_validation_tensor).any(), "NaN found in validation data"

# ---- Check for NaNs in target data ---- #
assert not torch.isnan(y_train_tensor).any(), "NaN found in training labels"
assert not torch.isnan(y_validation_tensor).any(), "NaN found in validation labels"

# ---- Create TensorDataset for training and validation sets ---- #
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)

# ---- Check for class balance ---- #
print("Class distribution in y_train:")
print(y_train_tensor.unique(return_counts=True))

print("Class distribution in y_validation:")
print(y_validation_tensor.unique(return_counts=True))

# ---- Create DataLoader for training and validation sets ---- #
"""
- data will be loaded in batches of 32 samples
- shuffle=True shuffles the data after each epoch (improves generalization by preventing overfitting)
- num_workers=0 uses a single worker to load the data (faster training)
"""
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=0)

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


# ---- Define the random architecture for the neural network based on the values mentioned in the scientific paper ---- #
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
        {'layer': 'MaxPooling', 'pool_size': random.choice([2, 3])},
        {'layer': 'Dense', 'units': random.choice([16, 32, 64, 128, 256]),
         'activation': random.choice(['relu', 'elu', 'selu', 'sigmoid', 'linear'])},
        {'layer': 'Dropout', 'rate': random.uniform(0.1, 0.5)},
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
    def evaluate(self, generation, max_generations):
        phenotype = self.to_phenotype()
        early_stop_callback = EarlyStopping(monitor='val_acc', patience=15, mode='max')

        trainer = pl.Trainer(min_epochs=10,                         # trains the model for at least 10 epochs
                             max_epochs=30,                         # trains the model for 30 epochs
                             logger=False,
                             enable_checkpointing=False,            # disables checkpointing to save the model
                             enable_progress_bar=False,
                             callbacks=[early_stop_callback],       # stops training when the validation accuracy does not improve for 15 epochs
                             gradient_clip_val=0.5,                 # clips the gradient to prevent exploding gradients
                             gradient_clip_algorithm='norm')        # normalizes the gradient to prevent exploding gradients
        
        # trainer.fit: trains the model
        trainer.fit(phenotype, train_dataloaders=train_loader, val_dataloaders=validation_loader)

        # trainer.validate: evaluates the model on the validation set
        trainer.validate(phenotype, dataloaders=validation_loader)

        # trainer.callback_metrics.get('val_acc', torch.tensor(0.0)): gets the validation accuracy from the trainer
        validation_accuracy = trainer.callback_metrics.get('val_acc', torch.tensor(0.0)).item()

        self.fitness = fitness_function(self.architecture, validation_accuracy, generation, max_generations)
        return self.fitness

    """
    - method that converts the genotype to a phenotype
    """
    def to_phenotype(self):
        return Phenotype(self.architecture)

"""
- method that initializes the weights of the neural network model
"""
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)


# ---- Phenotype class ---- #
"""
- defines the neural network model based on the genotype
- inherits from the PyTorch Lightning Module class
- pl.LightningModule: provides a simple interface for organizing PyTorch code; superclass
- Phenotype: subclass of pl.LightningModule that inherits its properties and methods
"""
class Phenotype(pl.LightningModule):

    """
    - constructor for the Phenotype class
    - initializes the class with the genotype and the loss function
    - initializes the accuracy metric using torchmetrics.Accuracy
    """
    def __init__(self, genotype=None):
        super(Phenotype, self).__init__()
        if genotype:
            self.genotype = genotype
            self.model = self.build_model_from_genotype(genotype)   # builds the neural network model based on the genotype
            self.model.apply(init_weights)                          # initializes the weights of the neural network model
        self.loss_fn = nn.CrossEntropyLoss()                        # calculates the loss between the predictions and the target data
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=2)

    """
    - method that builds the neural network model based on the genotype
    - variable layers: stores the layers of the neural network
    - variable input_channels: stores the number of input channels
    - variable output_size: stores the size of the output
    - for each layer in the genotype, the corresponding layer is added to the neural network
    - returns the neural network model
    """
    def build_model_from_genotype(self, genotype):
        layers = []
        input_channels = 1
        output_size = 1201

        for layer in genotype:
            if layer['layer'] == 'Conv':
                layers.append(nn.Conv1d(input_channels, layer['filters'], layer['kernel_size']))
                input_channels = layer['filters']
                output_size = output_size - layer['kernel_size'] + 1
                
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

        layers.append(nn.Linear(input_channels, 2))

        return nn.Sequential(*layers)
    
    """
    - method that loads the model from a checkpoint to resume training or evaluation with a previously trained model
    - checkpoint_path: path to the checkpoint file
    - genotype: genotype of the model
    - kwargs: additional keyword arguments that are passed to the constructor
    """
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, genotype, **kwargs):
        instance = cls(genotype=genotype, **kwargs)
        checkpoint = torch.load(checkpoint_path)
        """
        - instance.load_state_dict: loads the model state from the checkpoint
        - checkpoint['state_dict']: stores the model state in the checkpoint
        - strict=False: ignores the mismatch between the keys in the model and the checkpoint
        """
        instance.load_state_dict(checkpoint['state_dict'], strict=False)
        return instance

    """
    - method that defines how the data flows through the neural network
    - x: input data
    - x.view: reshapes the input data to fit the neural network
    - returns the output of the neural network (transformed data)
    """
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        return self.model(x)

    """
    - method that defines the training step for the neural network
    - calculates the loss and updates the model's weights to improve its predictions
    - batch: tuple containing input data (x) and target data (y) for training
    """
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        logits = self.forward(x)                        # passes the input data through the neural network (predictions) 
        y = y.view(-1)                                  # reshapes the target data to match the predictions
        loss = self.loss_fn(logits, y)                  # calculates the loss between the predictions and the target data
        preds = logits.argmax(dim=1)                    # returns the index of the maximum value in the predictions
        acc = self.accuracy(preds, y)                   # uses torchmetrics.Accuracy to calculate accuracy
        self.log('train_loss', loss, prog_bar=True)     # logs the training loss
        self.log('train_acc', acc, prog_bar=True)       # logs the training accuracy
        return loss
    
    """
    - method that defines the training epoch end for the neural network
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss, prog_bar = False)"""

    """
    - method that defines the validation step for the neural network
    - measures how well the model is performing on unseen data without updating its weights
    - calculates the loss and accuracy of the model on the validation set
    """
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        logits = self.forward(x)
        y = y.view(-1)
        loss = self.loss_fn(logits, y)                      # calculates the loss between the predictions and the target data
        preds = logits.argmax(dim=1)    

        # debugging logits and predictions
        #print(f"Logits: {logits[:5]}")
        #print(f"Predictions: {preds[:5]}")
        #print(f"Labels: {y[:5]}")

        acc = self.accuracy(preds, y)                               # uses torchmetrics.Accuracy to calculate accuracy
        self.log('val_loss', loss, prog_bar=True)                   # logs the validation loss
        self.log('val_acc', acc, prog_bar=True)                     # logs the validation accuracy
        return {'val_loss': loss, 'val_acc': acc}
    
    """
    - method that resets the accuracy metric at the start of each training epoch
    """
    def on_train_epoch_start(self):
        self.accuracy.reset()

    """
    - method that resets the accuracy metric at the start of each validation epoch
    """
    def on_validation_epoch_start(self):
        self.accuracy.reset()
    
    """
    - method that defines the validation epoch end for the neural network
    - calculates the average validation accuracy of the model
    """
    def on_validation_epoch_end(self):
        self.log('epoch_val_acc', self.accuracy.compute(), prog_bar=True)
        self.accuracy.reset()

    """
    - method that specifies how the model's parameters should be updated during training
    - defines the optimization algorithm -> Adam optimizer
    - adjusts the model's weights during training to minimize the loss
    - improves predictions by using past steps to guide the updates smoothly
    """
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

"""
- method that calculates the fitness score of the architecture based on its performance
- architecture: architecture of the neural network
- validation_accuracy: accuracy of the model on the validation set
- generation: current generation of the evolutionary algorithm
- max_generations: maximum number of generations for the evolutionary algorithm
- model_size: size of the model (number of parameters)
- training_time: time taken to train the model
- size_penalty: reduces fitness for larger models (penalty scaled by alpha)
- time_penalty: reduces fitness for longer training times (penalty scaled by BETA)
- fitness: overall fitness score of the architecture
- higher fitness -> better architecture
"""
def fitness_function(architecture, validation_accuracy, generation, max_generations):
    model_size = sum(layer.get('filters', 0) + layer.get('units', 0) for layer in architecture)
    training_time = 0.1 * model_size

    # dynamic weights for exploration and exploitation
    dynamic_alpha = alpha * (1 + (generation / max_generations))
    dynamic_BETA = BETA * (1 - (generation / max_generations))

    # accuracy_component = validation_accuracy

    size_penalty = dynamic_alpha * model_size
    time_penalty = dynamic_BETA * training_time

    #noise = random.uniform(0, 0.01)    # slight noise (randomness) for the fitness score
    fitness = validation_accuracy - size_penalty - time_penalty #+ noise
    return max(0.0, fitness)


# ---- Differential Evolution algorithm for NAS-T ---- #
class NASDifferentialEvolution:
    """
    - constructor for the NASDifferentialEvolution class
    """
    def __init__(self, population_size=population_size, generations=generations, verbose=True):
        self.population_size = population_size
        self.generations = generations
        self.population = self.initialize_population()      # initializes the population with random genotypes
        self.verbose = verbose                              # prints the progress of the evolutionary algorithm

    """
    - method that initializes the population with random genotypes
    - returns a list of genotypes
    """
    def initialize_population(self):
        return [Genotype() for _ in range(self.population_size)]

    """
    - method that mutates the genotype of an individual in the population to create a new individual (mutant)
    - parent1: first parent genotype
    - parent2: second parent genotype
    - parent3: third parent genotype
    - F: mutation factor for the evolutionary algorithm
    - returns the mutant genotype
    """
    def mutate(self, parent1, parent2, parent3, F):
        mutant = copy.deepcopy(parent1.architecture)    # creates a copy of the parent genotype
        filter_options = [8, 16, 32, 64, 128, 256]

        """
        - mutant[i]: i-th layer of the mutant genotype
        - F: mutation factor for the evolutionary algorithm that controls how much difference is introduced
        - parent1.architecture[i]: i-th layer of the first parent genotype (analogous for parent2 and parent3)
        """
        for i in range(len(mutant)):
            if random.random() < F:
                if 'filters' in mutant[i]:
                    # random.uniform(1.0, 2.0): generates a random number between 1.0 and 2.0 in order to increase or decrease the filter size
                    mutated_filter = int(parent1.architecture[i]['filters'] + F * (parent2.architecture[i]['filters'] - parent3.architecture[i]['filters'])) #* random.uniform(1.0, 2.0))
                    # key=lambda x: abs(x - mutated_filter): finds the closest value to the mutated filter
                    mutant[i]['filters'] = min(filter_options, key=lambda x: abs(x - mutated_filter))
                if 'units' in mutant[i]:
                    mutant[i]['units'] = max(16, min(64, int(parent1.architecture[i]['units'] + F * (parent2.architecture[i]['units'] - parent3.architecture[i]['units']))))
                if 'rate' in mutant[i]:
                    mutant[i]['rate'] = max(0.05, min(0.5, parent1.architecture[i]['rate'] + F * (parent2.architecture[i]['rate'] - parent3.architecture[i]['rate'])))
        return Genotype(mutant)


    """
    - method that performs crossover between a parent and a mutant to create an offspring
    - with a probability of crossover rate, replaces the parent's layer with the corresponding layer from the mutant
    """
    def crossover(self, parent, mutant, CR):
        offspring_architecture = copy.deepcopy(parent.architecture)
        for i in range(len(offspring_architecture)):
            if random.random() < CR:
                offspring_architecture[i] = mutant.architecture[i]
        return Genotype(offspring_architecture)
    
    """
    - method that trains and saves the phenotype of an individual in the population
    - train_loader: DataLoader for the training set
    - validation_loader: DataLoader for the validation set
    - save_path: path to save the phenotype
    """
    def train_and_save_phenotype(self, phenotype, train_loader, validation_loader, save_path):
        trainer = pl.Trainer(max_epochs=20)                                                         # trains the model for 10 epochs
        trainer.fit(phenotype, train_dataloaders=train_loader, val_dataloaders=validation_loader)   # fits the model to the training data
        trainer.save_checkpoint(save_path)                                                          # saves the model to a checkpoint file

    """
    - method that loads and evaluates its accuracy on the validation set
    """
    def load_and_evaluate_phenotype(self, phenotype_path, genotype, validation_loader):
        phenotype = Phenotype.load_from_checkpoint(phenotype_path, genotype=genotype)
        phenotype.eval()                    # sets the model to evaluation mode

        correct = 0                         # number of correct predictions
        total = 0                           # total number of predictions
        with torch.no_grad():               # disables gradient calculation
            for batch in validation_loader:
                x, y = batch
                logits = phenotype(x)
                predictions = logits.argmax(dim=1)                  # returns the index of the maximum value in the predictions
                correct += (predictions == y.view(-1)).sum().item() # compares the predicted class with the target class
                total += y.size(0)                                  # updates the total number of predictions; y.size(0) returns the batch size of the first dimesnion of y

        phenotype.train()                                           # sets the model back to training mode
        validation_accuracy = correct / total                       # calculates the accuracy of the model on the validation set
        return validation_accuracy


    """
    - method that evolves the population over a specified number of generations
    """
    def evolve(self):
        run_results = {"run_id": 1, "generations": []}
        best_overall_fitness = float('-inf')
        best_overall_individual = None
        best_generation = -1
        best_architectures = []

        # makes sure that fitness is evaluated for all individuals before sorting
        for individual in self.population:
            if individual.fitness is None:
                individual.evaluate(generation=0, max_generations=self.generations)

        # keeps the best individual unchanged in the next generation
        elite_count = 1                                               # number of elite individuals to keep

        # selects the elite individuals based on their fitness scores
        elite_individuals = sorted(self.population,
                                   key=lambda ind: ind.fitness,
                                   reverse=True)[:elite_count]

        """
        - method that calculates the population diversity based on the unique architectures in the population
        """
        def population_diversity(population):
            architectures = [ind.architecture for ind in population]
            unique_architectures = set(str(arch) for arch in architectures)
            return len(unique_architectures) / len(population)

        """
        - each iteration represents a generation where the population evolves
        - start_time: tracks how long the current generation takes to complete
        """
        for generation in range(self.generations):
            start_time = time.perf_counter()
            if self.verbose:
                print(f"Generation {generation + 1}")

            initial_F, final_F = 0.9, 0.5       # initial and final mutation factors
            initial_CR, final_CR = 0.9, 0.7     # initial and final crossover rates

            # dynamically adjusts mutation and crossover rates
            F = initial_F - (generation / self.generations) * (initial_F - final_F)
            CR = initial_CR - (generation / self.generations) * (initial_CR - final_CR)

            new_population = []                 # stores the new population of genotypes
            generation_fitnesses = []           # stores the fitness scores of the genotypes in the current generation

            """
            - for each individual in the population, creates a mutant and an offspring
            - evaluates the fitness of the offspring and compares it with the parent
            - if the offspring has a higher fitness, replaces the parent with the offspring
            - updates the population with the new individuals
            """
            for i in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)                                    # removes the current individual from the list of candidates
                parent1, parent2, parent3 = [self.population[idx] for idx in random.sample(candidates, 3)]
                mutant = self.mutate(parent1, parent2, parent3, F)
                offspring = self.crossover(self.population[i], mutant, CR)  # creates an offspring by performing crossover between the parent and the mutant

                parent_fitness = self.population[i].fitness or self.population[i].evaluate(generation, self.generations)          # evaluates the fitness of the parent
                offspring_fitness = offspring.evaluate(generation, self.generations)               # evaluates the fitness of the offspring

                if offspring_fitness > parent_fitness:
                    new_population.append(offspring)
                    generation_fitnesses.append(offspring_fitness)
                else:
                    new_population.append(self.population[i])
                    generation_fitnesses.append(parent_fitness)

            self.population = elite_individuals + new_population[:self.population_size - elite_count]
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            best_fitness = best_individual.fitness
            best_accuracy = best_individual.evaluate(generation, self.generations)

            if self.verbose:
                print(f"Best fitness in generation {generation + 1}: {best_fitness}")
                print(f"Best accuracy in generation {generation + 1}: {best_accuracy}")

            if best_fitness > best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_individual = best_individual
                best_generation = generation + 1

            """
            - end_time: tracks how long the current generation took to complete
            """
            end_time = time.perf_counter()
            runtime = end_time - start_time
            if self.verbose:
                print(f"Runtime for generation {generation + 1}: {runtime:.6f} seconds\n")

            generation_result = {
                "generation": generation + 1,
                "best_fitness": best_fitness,
                "best_accuracy": best_accuracy,
                "best_architecture": best_individual.architecture,
                "all_fitnesses": generation_fitnesses
            }
            run_results["generations"].append(generation_result)

            best_generations = {
                "generation": generation + 1,
                "best_fitness": best_fitness,
                "best_accuracy": best_accuracy,
                "best_architecture": best_individual.architecture,
                "runtime": runtime
            }
            best_architectures.append(best_generations)

        if self.verbose:
            print("Best overall architecture:")
            print(f"Found in generation {best_generation}")
            print(best_overall_individual.architecture)
            print("Fitness:", best_overall_fitness)
            print("Accuracy:", best_overall_fitness - alpha * sum(layer.get('filters', 0) + layer.get('units', 0)
                                                                  for layer in best_overall_individual.architecture), "\n")

            print("Best architecture in each generation:")
            for generation in best_architectures:
                print(f"Generation", generation['generation'])
                print(f"Best architecture:", generation['best_architecture'])
                print("Fitness:", generation['best_fitness'])
                print(f"Accuracy: {generation['best_accuracy']}")
                print("Runtime:", generation['runtime'], "seconds\n")

        save_run_results_json("evolutionary_runs.json", run_results)

if __name__ == "__main__":
    nas_de = NASDifferentialEvolution(verbose=True)
    nas_de.evolve()