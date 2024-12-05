import json
import matplotlib.pyplot as plt
import numpy as np

def plot_avg_fitness_accuracy_trends(json_file):
    """
    Plots average fitness and accuracy trends across generations from a JSON file.

    Parameters:
        json_file (str): Path to the JSON file containing evolutionary run results.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Initialize lists for storing all fitness and accuracy trends
    all_fitness_trends = []
    all_accuracy_trends = []
    max_generations = 0

    for run in data:
        fitness_trend = []
        accuracy_trend = []

        for generation in run["generations"]:
            fitness_trend.append(generation["best_fitness"])
            accuracy_trend.append(generation["best_accuracy"])

        all_fitness_trends.append(fitness_trend)
        all_accuracy_trends.append(accuracy_trend)
        max_generations = max(max_generations, len(fitness_trend))

    for i in range(len(all_fitness_trends)):
        all_fitness_trends[i] += [np.nan] * (max_generations - len(all_fitness_trends[i]))
        all_accuracy_trends[i] += [np.nan] * (max_generations - len(all_accuracy_trends[i]))

    # Convert to NumPy arrays for easier manipulation
    all_fitness_trends = np.array(all_fitness_trends)
    all_accuracy_trends = np.array(all_accuracy_trends)

    # Calculate the average fitness and accuracy trends, ignoring NaNs
    avg_fitness_trend = np.nanmean(all_fitness_trends, axis=0)
    avg_accuracy_trend = np.nanmean(all_accuracy_trends, axis=0)

    # Plot combined average fitness and accuracy trends
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_generations + 1), avg_fitness_trend, label='Average Fitness', color='purple')
    plt.plot(range(1, max_generations + 1), avg_accuracy_trend, label='Average Accuracy', color='deeppink')
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Average Fitness and Accuracy Across All Runs')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, max_generations + 1))  # Set x-axis ticks to be integers
    plt.ylim(0, 1)  # Set y-axis limit to 1.0
    plt.show()

if __name__ == "__main__":
    plot_avg_fitness_accuracy_trends("evolutionary_runs.json")
