import json
import numpy as np
import matplotlib.pyplot as plt

def plot_accuracies(accuracies_file, standard_results_file):
    with open(accuracies_file, 'r') as file:
        accuracies_data = json.load(file)

    with open(standard_results_file, 'r') as file:
        standard_results = json.load(file)

    # Ensure only valid runs are processed
    num_runs = len(accuracies_data)

    # Create plot
    plt.figure(figsize=(15, 6))  # Increased width to make x-axis ranges wider

    # Define color map
    colormap = plt.get_cmap('tab10')

    # Process accuracies
    for run_idx in range(num_runs):
        run = accuracies_data[run_idx]
        accuracies_x = []
        accuracies_y = []
        for gen_idx, generation in enumerate(run["generations"]):
            acc = generation["accuracies"]
            step = 1 / len(acc)
            # Shift x-axis to start at generation 1
            accuracies_x.extend(np.linspace(gen_idx + 1, gen_idx + 1 + 1 - step, len(acc)))
            accuracies_y.extend(acc)
        plt.plot(accuracies_x, accuracies_y, color=colormap(run_idx % colormap.N), label=f'Accuracy Run {run_idx + 1}')

    # Plot standard model accuracy
    plt.axhline(standard_results['final_accuracy'], color='red', linestyle='--', label='Standard Accuracy')

    # Add vertical dashed lines for generation ranges
    max_generation = max(len(run["generations"]) for run in accuracies_data)
    for gen in range(1, max_generation + 1):
        plt.axvline(gen, color='gray', linestyle='--', linewidth=0.5)

    # Configure plot
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.title('Accuracies')
    plt.ylim(0, 1.0)
    plt.xticks(range(1, max_generation + 1))  # Set x-axis ticks to match generations
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

plot_accuracies('nas_t/accuracies.json', 'nas_t/standard_results.json')
