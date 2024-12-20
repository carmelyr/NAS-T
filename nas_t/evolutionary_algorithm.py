import random
import copy
import torch
import time
import pytorch_lightning as pl
from config import population_size, generations
from utils import save_run_results_json
import random
import copy
from model_builder import Phenotype, Genotype, random_architecture


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
        """
        Mutates a parent's architecture using Differential Evolution.
        Ensures keys are checked before mutation to avoid KeyErrors.
        """
        mutant = copy.deepcopy(parent1.architecture)  # Creates a copy of the parent genotype
        filter_options = [8, 16, 32, 64, 128, 256]
        activation_options = ['relu', 'elu', 'selu', 'sigmoid', 'linear']
        pooling_options = [2, 3]
        kernel_size_options = [3, 5]

        for i in range(len(mutant)):
            layer_type = parent1.architecture[i]['layer']

            # Debugging messages
            print(f"parent1.architecture[{i}]: {parent1.architecture[i]}")
            print(f"parent2.architecture[{i}]: {parent2.architecture[i]}")
            print(f"parent3.architecture[{i}]: {parent3.architecture[i]}")

            if random.random() < F:
                if layer_type == 'Conv':  # Mutate 'filters' and 'kernel_size'
                    if 'filters' in parent1.architecture[i]:
                        mutated_filter = int(
                            parent1.architecture[i].get('filters', 0) +
                            F * (parent2.architecture[i].get('filters', 0) - parent3.architecture[i].get('filters', 0))
                        )
                        mutant[i]['filters'] = min(filter_options, key=lambda x: abs(x - mutated_filter))
                    if 'kernel_size' in parent1.architecture[i]:
                        mutant[i]['kernel_size'] = random.choice(kernel_size_options)

                if layer_type == 'Dense':  # Mutate 'units'
                    if 'units' in parent1.architecture[i]:
                        mutated_units = int(
                            parent1.architecture[i].get('units', 0) +
                            F * (parent2.architecture[i].get('units', 0) - parent3.architecture[i].get('units', 0))
                        )
                        mutant[i]['units'] = max(16, min(128, mutated_units))  # Ensure it's within a valid range

                if layer_type == 'Dropout':  # Mutate 'rate'
                    if 'rate' in parent1.architecture[i]:
                        mutant[i]['rate'] = max(
                            0.05,
                            min(
                                0.5,
                                parent1.architecture[i].get('rate', 0) +
                                F * (parent2.architecture[i].get('rate', 0) - parent3.architecture[i].get('rate', 0))
                            )
                        )

                if layer_type == 'MaxPooling':  # Mutate 'pool_size'
                    if 'pool_size' in parent1.architecture[i]:
                        mutant[i]['pool_size'] = random.choice(pooling_options)

                if layer_type == 'Activation':  # Mutate 'activation'
                    if 'activation' in parent1.architecture[i]:
                        mutant[i]['activation'] = random.choice(activation_options)

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
            elif random.random() < 0.1:
                offspring_architecture[i] = random_architecture()[i]
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
        best_fitness_so_far = float('-inf')  # initializes the tracker for the best fitness across generations
        best_architectures = []

        for generation in range(self.generations):
            start_time = time.perf_counter()
            if self.verbose:
                print(f"Generation {generation + 1}")

            # dynamic weights for exploration and exploitation
            initial_F, final_F = 0.9, 0.5
            initial_CR, final_CR = 0.9, 0.7
            F = initial_F - (generation / self.generations) * (initial_F - final_F)
            CR = initial_CR - (generation / self.generations) * (initial_CR - final_CR)

            new_population = []
            generation_fitnesses = []

            elitism_count = max(1, int(0.1 * self.population_size))     # elitism: selects the top 10% of the population based on fitness
            sorted_population = sorted(self.population, key=lambda ind: ind.fitness or float('-inf'), reverse=True)   # sorts the population based on fitness
            new_population.extend(sorted_population[:elitism_count])    # adds the top individuals to the new population

            for i in range(elitism_count, self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)
                parent1, parent2, parent3 = [self.population[idx] for idx in random.sample(candidates, 3)]
                mutant = self.mutate(parent1, parent2, parent3, F)
                offspring = self.crossover(self.population[i], mutant, CR)

                parent_fitness = self.population[i].fitness or self.population[i].evaluate(generation, self.generations)
                offspring_fitness = offspring.evaluate(generation, self.generations)

                if offspring_fitness > parent_fitness:
                    new_population.append(offspring)
                    generation_fitnesses.append(offspring_fitness)
                else:
                    new_population.append(self.population[i])
                    generation_fitnesses.append(parent_fitness)

            self.population = new_population

            # ensures that all individuals in the population have been evaluated
            for individual in self.population:
                if individual.fitness is None:
                    individual.evaluate(generation, self.generations)

            # determines the best individual in the population
            try:
                best_individual = max(self.population, key=lambda ind: ind.fitness)
            except ValueError as e:
                print(f"Error finding best individual in generation {generation}: {e}")
                continue

            # determines the best fitness of the generation
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            best_fitness = best_individual.fitness

            # ensures that the best fitness is not decreasing across generations
            # the best fitness can stay the same or grow only
            if best_fitness < best_fitness_so_far:
                best_fitness = best_fitness_so_far
            else:
                best_fitness_so_far = best_fitness

            best_architecture = best_individual.architecture

            end_time = time.perf_counter()
            runtime = "{:.2f}".format(end_time - start_time)

            generation_result = {
                "generation": generation + 1,
                "best_fitness": best_fitness,
                "best_architecture": best_architecture,
                "all_fitnesses": generation_fitnesses,
                "runtime": runtime,
            }
            run_results["generations"].append(generation_result)

            best_generations = {
                "generation": generation + 1,
                "best_fitness": best_fitness,
                "best_architecture": best_architecture,
                "runtime": runtime,
            }
            best_architectures.append(best_generations)

            if self.verbose:
                print(f"Best fitness in generation {generation + 1}: {best_fitness}")
                print(f"Best architecture: {best_architecture}\n")

            total_runtime = sum(float(gen['runtime']) for gen in best_architectures)    # calculates the total runtime of the evolutionary algorithm
            total_runtime = "{:.2f}".format(total_runtime)                              # formats the total runtime to two decimal places

        # final logging
        if self.verbose:
            print(f"Best overall fitness: {best_fitness_so_far}")
            print(f"Overall running time: {total_runtime} seconds")
            print(f"Best overall architecture: {best_architecture}\n")
            print("Best architectures across generations:")
            for gen in best_architectures:
                print(f"Generation {gen['generation']}:\n Fitness {gen['best_fitness']}\nRuntime {gen['runtime']} seconds\n Architecture: {gen['best_architecture']}\n")

            save_run_results_json("nas_t/evolutionary_runs.json", run_results)