import json
import numpy as np
import matplotlib.pyplot as plt

def plot_accuracies_and_model_sizes(accuracies_file, model_sizes_file, standard_results_file):
    with open(accuracies_file, 'r') as file:
        accuracies_data = json.load(file)

    with open(model_sizes_file, 'r') as file:
        model_sizes_data = json.load(file)

    with open(standard_results_file, 'r') as file:
        standard_results = json.load(file)

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
        
        # Set y-ticks for accuracy plot
        axs[0].set_yticks(np.arange(0, 1.1, 0.1))

    # Plot standard model accuracy
    axs[0].axhline(standard_results['final_accuracy'], color='red', linestyle='--', label='Standard Accuracy')

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

    # Plot standard model size
    axs[1].axhline(standard_results['final_model_size'], color='red', linestyle='--', label='Standard Model Size')

    # Add vertical lines for generations
    max_generation = max(len(run["generations"]) for run in accuracies_data)
    for gen in range(1, max_generation + 1):
        axs[0].axvline(gen, color='gray', linestyle='--', linewidth=0.5)
        axs[1].axvline(gen, color='gray', linestyle='--', linewidth=0.5)

    # Configure Accuracy plot
    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Accuracies')
    axs[0].set_ylim(0, 1.0)
    axs[0].legend(loc='upper right')

    # Configure Model Sizes plot
    axs[1].set_xlabel('Generation')
    axs[1].set_ylabel('Model Size')
    axs[1].set_title('Model Sizes')
    axs[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

plot_accuracies_and_model_sizes('nas_t/accuracies.json', 'nas_t/model_sizes.json', 'nas_t/standard_results.json')
