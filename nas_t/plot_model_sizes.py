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
    max_model_size = 0
    for run_idx in range(num_runs):
        run = model_sizes_data[run_idx]
        sizes_x = []
        sizes_y = []
        for gen_idx, generation in enumerate(run["generations"]):
            sizes = generation["model_sizes"]
            step = 1 / len(sizes)

            # Ensure model sizes are integers
            sizes = [int(size) for size in sizes]

            # Shift x-axis to start at generation 1
            sizes_x.extend(np.linspace(gen_idx + 1, gen_idx + 1 + 1 - step, len(sizes)))
            sizes_y.extend(sizes)
            max_model_size = max(max_model_size, max(sizes))  # Track max size for scaling

        plt.plot(sizes_x, sizes_y, color=colormap(run_idx % colormap.N), label=f'Model Size Run {run_idx + 1}')

    # Get the standard model size
    standard_model_size = int(standard_results['final_model_size'])

    # Update max size to include the standard model size
    max_model_size = max(max_model_size, standard_model_size)

    # Plot standard model size as a horizontal dashed line
    plt.axhline(standard_model_size, color='red', linestyle='--', label='Standard Model Size')

    # Add vertical dashed lines for generation ranges
    max_generation = max(len(run["generations"]) for run in model_sizes_data)
    for gen in range(1, max_generation + 1):
        plt.axvline(gen, color='gray', linestyle='--', linewidth=0.5)

    # Configure plot
    plt.xlabel('Generation')
    plt.ylabel('Model Size')
    plt.title('Model Sizes')

    # Set Y-axis dynamically based on data range and fixed intervals of 100,000
    plt.ylim(0, max_model_size * 1.1)  # Slightly higher than max size for padding
    plt.yticks(np.arange(0, max_model_size * 1.1, 100000))  # Dynamic ticks based on max size

    # Set x-axis ticks for generations
    plt.xticks(range(1, max_generation + 1))  # Set x-axis ticks to match generations
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Example call
plot_model_sizes('nas_t/model_sizes.json', 'nas_t/standard_results.json')
