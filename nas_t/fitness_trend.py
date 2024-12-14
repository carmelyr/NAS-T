import json
import numpy as np
import matplotlib.pyplot as plt

def plot_fitness_trend(json_file):
    """
    Plots average fitness, accuracy, and runtime trends across generations from a JSON file.

    Parameters:
        json_file (str): Path to the JSON file containing evolutionary run results.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Initialize lists for storing all fitness, accuracy, and runtime trends
    all_fitness_trends = []
    all_runtime_trends = []
    max_generations = 0
    overall_runtime = 0  # Initialize overall runtime

    for run in data:
        fitness_trend = []
        runtime_trend = []

        for generation in run["generations"]:
            fitness_trend.append(float(generation["best_fitness"]))
            runtime_trend.append(float(generation["runtime"]))
            overall_runtime += float(generation["runtime"])  # Add to overall runtime

        all_fitness_trends.append(fitness_trend)
        all_runtime_trends.append(runtime_trend)
        max_generations = max(max_generations, len(fitness_trend))

    for i in range(len(all_fitness_trends)):
        all_fitness_trends[i] += [np.nan] * (max_generations - len(all_fitness_trends[i]))
        all_runtime_trends[i] += [np.nan] * (max_generations - len(all_runtime_trends[i]))

    # Convert to NumPy arrays for easier manipulation
    all_fitness_trends = np.array(all_fitness_trends)
    all_runtime_trends = np.array(all_runtime_trends)

    # Calculate average trends
    avg_fitness_trend = np.nanmean(all_fitness_trends, axis=0)
    avg_runtime_trend = np.nanmean(all_runtime_trends, axis=0)

    # Calculate overall average runtime and fitness
    overall_avg_runtime = np.nanmean(all_runtime_trends)
    overall_avg_fitness = np.nanmean(all_fitness_trends)

    # Print overall average runtime and fitness
    print(f"Overall Average Runtime: {overall_avg_runtime}")
    print(f"Overall Average Fitness: {overall_avg_fitness}")

    # Plot average trends
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot each run's runtime trend with a unique color
    colormap = plt.cm.get_cmap('tab10', len(all_runtime_trends))
    for i, runtime_trend in enumerate(all_runtime_trends):
        axs[0].plot(runtime_trend, label=f'Run {i+1}', color=colormap(i))

    # Plot overall average runtime trend as a dashed line
    axs[0].plot(avg_runtime_trend, label='Average Runtime', color='black', linestyle='--')
    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Runtime in seconds')
    axs[0].set_title('Runtime')
    axs[0].legend()
    axs[0].set_ylim(0, 300)  # Set y-axis limit for runtime
    axs[0].set_yticks([0, 50, 100, 150, 200, 250, 300])  # Set y-axis ticks for runtime
    axs[0].set_xticks(range(max_generations))  # Start from 0
    axs[0].set_xticklabels(range(1, max_generations + 1))  # Label starting from 1

    # Plot each run's fitness trend with a unique color
    for i, fitness_trend in enumerate(all_fitness_trends):
        axs[1].plot(fitness_trend, label=f'Run {i+1}', color=colormap(i))

    # Plot overall average fitness trend as a dashed line
    axs[1].plot(avg_fitness_trend, label='Average Fitness', color='black', linestyle='--')
    axs[1].set_xlabel('Generation')
    axs[1].set_ylabel('Fitness')
    axs[1].set_title('Fitness')
    axs[1].legend()
    axs[1].set_ylim(0, 1.0)  # Set y-axis limit for fitness
    axs[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Set y-axis ticks for fitness
    axs[1].set_xticks(range(max_generations))  # Start from 0
    axs[1].set_xticklabels(range(1, max_generations + 1))  # Label starting from 1

    plt.show()

# Example usage
plot_fitness_trend('nas_t/evolutionary_runs.json')