import copy
import random
import time
import json
import os
from config import population_size, generations, device, F, CR
from utils import save_run_results_json, save_accuracies_json, save_model_sizes_json
from model_builder import Phenotype, Genotype, random_architecture

class NASDifferentialEvolution:
    def __init__(self, population_size=population_size, generations=generations, verbose=True):
        self.population_size = population_size
        self.generations = generations
        self.population = self.initialize_population()
        self.verbose = verbose

    def initialize_population(self):
        return [Genotype(device=device) for _ in range(self.population_size)]

    def mutate(self, parent1, parent2, parent3, F):
        mutant = copy.deepcopy(parent1.architecture)
        filter_options = [8, 16, 32, 64, 128, 256]
        activation_options = ['relu', 'elu', 'selu', 'sigmoid', 'linear']
        pooling_options = [2, 3]
        kernel_size_options = [3, 5]

        for i in range(len(mutant)):
            layer_type = parent1.architecture[i]['layer']
            try:
                if random.random() < F:
                    if layer_type == 'Conv':
                        if 'filters' in parent1.architecture[i]:
                            mutant[i]['filters'] = random.choice(filter_options)
                        if 'kernel_size' in parent1.architecture[i]:
                            mutant[i]['kernel_size'] = random.choice(kernel_size_options)
                    elif layer_type == 'Dense':
                        if 'units' in parent1.architecture[i]:
                            mutant[i]['units'] = random.choice(filter_options)
                    elif layer_type == 'Dropout':
                        if 'rate' in parent1.architecture[i]:
                            mutant[i]['rate'] = random.uniform(0.1, 0.5)
                    elif layer_type == 'MaxPooling':
                        if 'pool_size' in parent1.architecture[i]:
                            mutant[i]['pool_size'] = random.choice(pooling_options)
                    elif layer_type == 'Activation':
                        if 'activation' in parent1.architecture[i]:
                            mutant[i]['activation'] = random.choice(activation_options)
            except IndexError:
                continue

        return Genotype(mutant)

    def crossover(self, parent, mutant, CR):
        offspring_architecture = copy.deepcopy(parent.architecture)
        for i in range(len(offspring_architecture)):
            if random.random() < CR:
                try:
                    offspring_architecture[i] = mutant.architecture[i]
                except IndexError:
                    continue
        return Genotype(offspring_architecture)

    def evolve(self):
        run_results = {"run_id": 1, "generations": []}
        best_fitness_so_far = float('-inf')
        best_architectures = []
        all_accuracies = []
        all_model_sizes = []

        for generation in range(self.generations):
            start_time = time.perf_counter()
            if self.verbose:
                print(f"Generation {generation + 1}--------------------------------------------")

            generation_accuracies = []
            generation_model_sizes = []
            generation_fitnesses = []

            #initial_F, final_F = 0.9, 0.5
            #initial_CR, final_CR = 0.9, 0.7
            #F = initial_F - (generation / self.generations) * (initial_F - final_F)
            #CR = initial_CR - (generation / self.generations) * (initial_CR - final_CR)

            new_population = []

            elitism_count = max(1, int(0.1 * self.population_size))
            sorted_population = sorted(self.population, key=lambda ind: ind.fitness or float('-inf'), reverse=True)
            new_population.extend(sorted_population[:elitism_count])

            for i in range(elitism_count, self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)
                parent1, parent2, parent3 = [self.population[idx] for idx in random.sample(candidates, 3)]
                mutant = self.mutate(parent1, parent2, parent3, F)
                offspring = self.crossover(self.population[i], mutant, CR)

                parent_fitness, parent_accuracy, parent_size = self.population[i].evaluate(generation, self.generations)
                offspring_fitness, offspring_accuracy, offspring_size = offspring.evaluate(generation, self.generations)

                generation_accuracies.append(parent_accuracy)
                generation_model_sizes.append(parent_size)
                generation_fitnesses.append(parent_fitness)
                generation_fitnesses.append(offspring_fitness)
                generation_accuracies.append(offspring_accuracy)
                generation_model_sizes.append(offspring_size)

                if offspring_fitness > parent_fitness:
                    new_population.append(offspring)
                else:
                    new_population.append(self.population[i])

            self.population = new_population

            all_accuracies.append(generation_accuracies)
            all_model_sizes.append(generation_model_sizes)

            generation_result = {
                "generation": generation + 1,
                "all_accuracies": generation_accuracies,
                "all_model_sizes": generation_model_sizes,
                "all_fitnesses": generation_fitnesses,
            }
            run_results["generations"].append(generation_result)

            if self.verbose:
                print(f"Generation {generation + 1} Summary:")
                print(f"Accuracies: {generation_accuracies}")
                print(f"Model sizes: {generation_model_sizes}\n")

            for individual in self.population:
                if individual.fitness is None:
                    individual.evaluate(generation, self.generations)

            try:
                best_individual = max(self.population, key=lambda ind: ind.fitness)
            except ValueError as e:
                print(f"Error finding best individual in generation {generation}: {e}")
                continue

            best_fitness = best_individual.fitness
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

            total_runtime = sum(float(gen['runtime']) for gen in best_architectures)
            total_runtime = "{:.2f}".format(total_runtime)

        # Load existing accuracies data
        if os.path.exists('nas_t/accuracies.json'):
            with open('nas_t/accuracies.json', 'r') as f:
                try:
                    accuracies_data = json.load(f)
                except json.JSONDecodeError:
                    accuracies_data = {"run_id": 1, "generations": []}
        else:
            accuracies_data = {"run_id": 1, "generations": []}

        # Append new accuracies data
        new_accuracies_data = [{"generation": i + 1, "accuracies": acc} for i, acc in enumerate(all_accuracies)]
        if isinstance(accuracies_data, list):
            # Append data for the latest run
            accuracies_data.append({
                "run_id": len(accuracies_data) + 1,
                "generations": new_accuracies_data
            })
        else:
            # Fallback in case data is not a list
            accuracies_data = [{
                "run_id": 1,
                "generations": new_accuracies_data
            }]

        # Load existing model sizes data
        if os.path.exists('nas_t/model_sizes.json'):
            with open('nas_t/model_sizes.json', 'r') as f:
                try:
                    model_sizes_data = json.load(f)
                except json.JSONDecodeError:
                    model_sizes_data = {"run_id": 1, "generations": []}
        else:
            model_sizes_data = {"run_id": 1, "generations": []}

        # Append new model sizes data
        new_model_sizes_data = [{"generation": i + 1, "model_sizes": size} for i, size in enumerate(all_model_sizes)]
        if isinstance(model_sizes_data, list):
            model_sizes_data.append({
                "run_id": len(model_sizes_data) + 1,
                "generations": new_model_sizes_data
            })
        else:
            model_sizes_data = [{
                "run_id": 1,
                "generations": new_model_sizes_data
            }]

        # Save updated accuracies data
        with open('nas_t/accuracies.json', 'w') as f:
            json.dump(accuracies_data, f, indent=4)

        # Save updated model sizes data
        with open('nas_t/model_sizes.json', 'w') as f:
            json.dump(model_sizes_data, f, indent=4)

        if self.verbose:
            print(f"Best overall fitness: {best_fitness_so_far}")
            print(f"Overall running time: {total_runtime} seconds")
            print(f"Best overall architecture: {best_architecture}\n")
            print("Best architectures across generations:")
            for gen in best_architectures:
                print(f"Generation {gen['generation']}:\n Fitness {gen['best_fitness']}\nRuntime {gen['runtime']} seconds\n Architecture: {gen['best_architecture']}\n")

        save_run_results_json("nas_t/evolutionary_runs.json", run_results)
        save_accuracies_json('nas_t/accuracies.json', all_accuracies)
        save_model_sizes_json('nas_t/model_sizes.json', all_model_sizes)
