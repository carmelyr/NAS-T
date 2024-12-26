import json
import os
from config import alpha, BETA

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


# ---- Save accuracies to a JSON file ---- #
def save_accuracies_json(filename, all_accuracies):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):  # Fix: Ensure data is a list
                    data = []
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if data:
        latest_run_id = max(run['run_id'] for run in data)
        run_id = latest_run_id + 1
    else:
        run_id = 1

    run_data = {
        "run_id": run_id,
        "generations": []
    }

    for i, acc in enumerate(all_accuracies):
        run_data["generations"].append({
            "generation": i + 1,
            "accuracies": acc
        })

    data.append(run_data)

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


# ---- Save model sizes to a JSON file ---- #
def save_model_sizes_json(filename, all_model_sizes):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):  # Fix: Ensure data is a list
                    data = []
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if data:
        latest_run_id = max(run['run_id'] for run in data)
        run_id = latest_run_id + 1
    else:
        run_id = 1

    run_data = {
        "run_id": run_id,
        "generations": []
    }

    for i, size in enumerate(all_model_sizes):
        run_data["generations"].append({
            "generation": i + 1,
            "model_sizes": size
        })

    data.append(run_data)

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


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
def fitness_function(architecture, validation_accuracy):
    #model_size = sum(layer.get('filters', 0) + layer.get('units', 0) for layer in architecture)

    # dynamic weights for exploration and exploitation
    # todo: fixed rates
    #dynamic_alpha = alpha * (1 + (generation / max_generations * 0.5))
    #dynamic_BETA = BETA * (1 - (generation / max_generations * 0.5))
    
    #size_penalty = dynamic_alpha * model_size
    #time_penalty = dynamic_BETA

    #noise = random.uniform(0, 0.01)    # slight noise (randomness) for the fitness score
    fitness = validation_accuracy - alpha - BETA #size_penalty - time_penalty #+ noise
    return max(0.0, fitness)