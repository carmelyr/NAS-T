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

    for run in data:
        fitness_trend = []
        runtime_trend = []

        for generation in run["generations"]:
            fitness_trend.append(float(generation["best_fitness"]))
            runtime_trend.append(float(generation["runtime"]))

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
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot average runtime trend
    axs[0].plot(avg_runtime_trend, label='Runtime per Generation', color='deeppink')
    axs[0].axhline(y=overall_avg_runtime, color='blue', linestyle='--', label='Overall Average Runtime')
    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Runtime')
    axs[0].set_title('Runtime')
    axs[0].legend()
    axs[0].set_xticks(range(1, max_generations + 1))

    # Plot average fitness trend
    axs[1].plot(avg_fitness_trend, label='Fitness per Generation', color='purple')
    axs[1].axhline(y=overall_avg_fitness, color='green', linestyle='--', label='Overall Average Fitness')
    axs[1].set_xlabel('Generation')
    axs[1].set_ylabel('Fitness')
    axs[1].set_title('Fitness')
    axs[1].legend()
    axs[1].set_xticks(range(1, max_generations + 1))

    plt.tight_layout()
    plt.show()

# Example usage
plot_fitness_trend('./evolutionary_runs.json')