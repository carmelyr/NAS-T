import json
import os
from config import alpha, BETA
import csv

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
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # assigns run ID dynamically
    if len(data) == 0:
        run_id = 1
    else:
        run_id = max(run['run_id'] for run in data) + 1

    # prepares data for the current run
    run_data = {
        "run_id": run_id,
        "generations": [{"generation": i + 1, "accuracies": acc} for i, acc in enumerate(all_accuracies)]
    }

    # appends new data and save
    data.append(run_data)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


# ---- Save model sizes to a JSON file ---- #
def save_model_sizes_json(filename, all_model_sizes):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if len(data) == 0:
        run_id = 1
    else:
        run_id = max(run['run_id'] for run in data) + 1

    run_data = {
        "run_id": run_id,
        "generations": [{"generation": i + 1, "model_sizes": size} for i, size in enumerate(all_model_sizes)]
    }

    data.append(run_data)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def save_results_csv(filename, run_id, generation, architecture, layers, fold_accuracies, val_accuracy, model_size, runtime):
    file_exists = os.path.exists(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Add headers if file is newly created or empty
        if not file_exists or os.stat(filename).st_size == 0:
            writer.writerow(["Run ID", "Generation", "Architecture", "Layers", "Fold Accuracies", "Validation Accuracy", "Model Size", "Training Time (s)"])
        
        # Write the new row with the provided run_id
        writer.writerow([run_id, generation, architecture, layers, fold_accuracies, val_accuracy, model_size, runtime])

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
def fitness_function(architecture, validation_accuracy, model_size):
    size_penalty = alpha * model_size
    fitness = validation_accuracy - size_penalty - BETA
    print(f"Debug - val_acc: {validation_accuracy}, model_size: {model_size}, size_penalty: {size_penalty}, fitness: {fitness}")
    return max(0.0, fitness)