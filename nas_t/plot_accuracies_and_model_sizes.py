import json
import numpy as np
import matplotlib.pyplot as plt

def plot_accuracies_and_model_sizes(accuracies_file, model_sizes_file):
    with open(accuracies_file, 'r') as file:
        accuracies_data = json.load(file)

    with open(model_sizes_file, 'r') as file:
        model_sizes_data = json.load(file)

    # Ensure only valid runs are processed
    num_runs = min(len(accuracies_data), len(model_sizes_data))

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Define color map
    colormap = plt.get_cmap('tab10')

    # Process accuracies
    all_accuracies_combined = []
    accuracies_x_combined = []

    for run_idx in range(num_runs):
        run = accuracies_data[run_idx]
        accuracies_x = []
        accuracies_y = []
        for gen_idx, generation in enumerate(run["generations"]):
            acc = generation["accuracies"]
            step = 1 / len(acc)
            accuracies_x.extend(np.linspace(gen_idx, gen_idx + 1 - step, len(acc)))
            accuracies_y.extend(acc)
        all_accuracies_combined.extend(accuracies_y)
        accuracies_x_combined.extend(accuracies_x)
        axs[0].plot(accuracies_x, accuracies_y, color=colormap(run_idx % colormap.N), label=f'Accuracy Run {run_idx + 1}')

    # Calculate overall average accuracy across all runs
    num_generations = len(accuracies_data[0]["generations"])
    avg_accuracies = [np.mean([run["generations"][gen]["accuracies"] for run in accuracies_data[:num_runs]]) for gen in range(num_generations)]
    axs[0].plot(np.linspace(0, num_generations, len(accuracies_x_combined)), 
                np.interp(np.linspace(0, num_generations, len(accuracies_x_combined)), 
                          range(0, num_generations), avg_accuracies), 
                'k--', label='Avg Accuracy')

    # Process model sizes
    all_model_sizes_combined = []
    model_sizes_x_combined = []

    for run_idx in range(num_runs):
        run = model_sizes_data[run_idx]
        model_sizes_x = []
        model_sizes_y = []
        for gen_idx, generation in enumerate(run["generations"]):
            size = generation["model_sizes"]
            step = 1 / len(size)
            model_sizes_x.extend(np.linspace(gen_idx, gen_idx + 1 - step, len(size)))
            model_sizes_y.extend(size)
        all_model_sizes_combined.extend(model_sizes_y)
        model_sizes_x_combined.extend(model_sizes_x)
        axs[1].plot(model_sizes_x, model_sizes_y, color=colormap(run_idx % colormap.N), label=f'Model Size Run {run_idx + 1}')

    # Calculate overall average model size across all runs
    avg_model_sizes = [np.mean([run["generations"][gen]["model_sizes"] for run in model_sizes_data[:num_runs]]) for gen in range(num_generations)]
    axs[1].plot(np.linspace(0, num_generations, len(model_sizes_x_combined)), 
                np.interp(np.linspace(0, num_generations, len(model_sizes_x_combined)), 
                          range(0, num_generations), avg_model_sizes), 
                'k--', label='Avg Model Size')

    # Configure Accuracy plot
    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Accuracies')
    axs[0].set_ylim(0, 1.0)
    axs[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0].legend(loc='upper left')

    # Configure Model Sizes plot
    axs[1].set_xlabel('Generation')
    axs[1].set_ylabel('Model Size')
    axs[1].set_title('Model Sizes')
    axs[1].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

plot_accuracies_and_model_sizes('nas_t/accuracies.json', 'nas_t/model_sizes.json')
