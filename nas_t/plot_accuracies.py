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
    plt.figure(figsize=(15, 6))

    # Define color map
    colormap = plt.get_cmap('tab10')

    # Process accuracies for evolutionary runs
    max_generations = len(accuracies_data[0]['generations'])
    for run_idx in range(num_runs):
        run = accuracies_data[run_idx]
        accuracies_x = []
        accuracies_y = []
        for gen_idx, generation in enumerate(run["generations"]):
            acc = generation["accuracies"]
            step = 1 / len(acc)
            accuracies_x.extend(np.linspace(gen_idx + 1, gen_idx + 2 - step, len(acc)))
            accuracies_y.extend(acc)
        plt.plot(accuracies_x, accuracies_y, color=colormap(run_idx % colormap.N), label=f'Accuracy Run {run_idx + 1}')

    # Process standard results
    for idx, standard in enumerate(standard_results):
        standard_accuracy = standard['final_accuracy']
        generations = list(range(1, max_generations + 2))
        accuracies = [standard_accuracy] * len(generations)
        plt.plot(generations, accuracies, linestyle='--', 
                 color=colormap((idx + num_runs) % colormap.N), label=f'Standard {idx + 1}')

    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.title('Accuracies')
    plt.ylim(0, 1.0)
    plt.xlim(1, max_generations + 1)  
    plt.xticks(range(1, max_generations + 1))
    plt.legend(loc='lower left', fontsize='x-small')

    plt.tight_layout()
    plt.show()

plot_accuracies('nas_t/accuracies.json', 'nas_t/standard_results.json')
