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

    # Process standard results
    for idx, standard in enumerate(standard_results):
        standard_model_size = int(standard['final_model_size'])
        plt.axhline(standard_model_size, color=colormap((idx + num_runs) % colormap.N), 
                    linestyle='--', label=f'Standard {idx + 1}')

    # Configure plot
    plt.xlabel('Generation')
    plt.ylabel('Model Size')
    plt.title('Model Sizes')

    # Adjust Y-axis dynamically
    plt.ylim(0, 1000000)
    plt.yticks(np.arange(0, 1000000, 100000))

    # Set x-axis ticks for generations
    max_generation = max(len(run["generations"]) for run in model_sizes_data)
    plt.xticks(range(1, max_generation + 1))
    plt.legend(loc='upper right', fontsize='x-small')

    plt.tight_layout()
    plt.show()

# Example call
plot_model_sizes('nas_t/model_sizes.json', 'nas_t/standard_results.json')
