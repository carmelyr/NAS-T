import json
import numpy as np
import matplotlib.pyplot as plt

def plot_model_sizes(model_sizes_file, standard_results_file):
    with open(model_sizes_file, 'r') as file:
        model_sizes_data = json.load(file)

    with open(standard_results_file, 'r') as file:
        standard_results = json.load(file)

    # Ensure only valid runs are processed
    num_runs = len(model_sizes_data)

    # Create plot
    plt.figure(figsize=(15, 6))  # Increased width to make x-axis ranges wider

    # Define color map
    colormap = plt.get_cmap('tab10')

    # Process model sizes
    for run_idx in range(num_runs):
        run = model_sizes_data[run_idx]
        model_sizes_x = []
        model_sizes_y = []
        for gen_idx, generation in enumerate(run["generations"]):
            size = generation["model_sizes"]
            step = 1 / len(size)
            # Shift x-axis to start at generation 1
            model_sizes_x.extend(np.linspace(gen_idx + 1, gen_idx + 1 + 1 - step, len(size)))
            model_sizes_y.extend(size)
        plt.plot(model_sizes_x, model_sizes_y, color=colormap(run_idx % colormap.N), label=f'Model Size Run {run_idx + 1}')

    # Plot standard model size
    plt.axhline(standard_results['final_model_size'], color='red', linestyle='--', label='Standard Model Size')

    # Add vertical dashed lines for generation ranges
    max_generation = max(len(run["generations"]) for run in model_sizes_data)
    for gen in range(1, max_generation + 1):
        plt.axvline(gen, color='gray', linestyle='--', linewidth=0.5)

    # Configure plot
    plt.xlabel('Generation')
    plt.ylabel('Model Size')
    plt.title('Model Sizes')
    plt.xticks(range(1, max_generation + 1))  # Set x-axis ticks to match generations
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

plot_model_sizes('nas_t/model_sizes.json', 'nas_t/standard_results.json')
