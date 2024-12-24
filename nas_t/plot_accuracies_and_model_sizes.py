import json
import numpy as np
import matplotlib.pyplot as plt

def plot_accuracies_and_model_sizes(accuracies_file, model_sizes_file):
    with open(accuracies_file, 'r') as file:
        accuracies_data = json.load(file)

    with open(model_sizes_file, 'r') as file:
        model_sizes_data = json.load(file)

    all_accuracies = []
    all_model_sizes = []
    num_generations = len(accuracies_data["generations"])

    for run in [accuracies_data]:
        accuracies_trend = []
        for generation in run["generations"]:
            accuracies_trend.extend(generation["accuracies"])
        all_accuracies.append(accuracies_trend)

    for run in [model_sizes_data]:
        model_sizes_trend = []
        for generation in run["generations"]:
            model_sizes_trend.extend(generation["model_sizes"])
        all_model_sizes.append(model_sizes_trend)

    # converts the lists to numpy arrays
    all_accuracies = np.array(all_accuracies[0])
    all_model_sizes = np.array(all_model_sizes[0])

    # Calculate averages per generation
    avg_accuracies = np.array([np.mean(all_accuracies[i::num_generations]) for i in range(num_generations)])
    avg_model_sizes = np.array([np.mean(all_model_sizes[i::num_generations]) for i in range(num_generations)])

    # plots the accuracies and model sizes
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Color map for multiple runs
    colormap = plt.cm.get_cmap('tab10', 1)

    # Plot accuracies
    x_range_accuracies = np.linspace(1, num_generations, len(all_accuracies))
    axs[0].plot(x_range_accuracies, all_accuracies, color=colormap(0), label='Accuracy')
    axs[0].plot(np.arange(1, num_generations + 1), avg_accuracies, 'k--', label='Avg Accuracy')

    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Accuracies')
    axs[0].set_ylim(0, 1.0)
    axs[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0].set_xticks(np.arange(1, num_generations + 1))
    axs[0].legend(loc='upper left')

    # Plot model sizes
    x_range_model_sizes = np.linspace(1, num_generations, len(all_model_sizes))
    axs[1].plot(x_range_model_sizes, all_model_sizes, color=colormap(0), label='Model Size')
    axs[1].plot(np.arange(1, num_generations + 1), avg_model_sizes, 'k--', label='Avg Model Size')

    axs[1].set_xlabel('Generation')
    axs[1].set_ylabel('Model Size')
    axs[1].set_title('Model Sizes')
    axs[1].set_xticks(np.arange(1, num_generations + 1))
    axs[1].legend(loc='upper left')

    plt.show()

plot_accuracies_and_model_sizes('nas_t/accuracies.json', 'nas_t/model_sizes.json')
