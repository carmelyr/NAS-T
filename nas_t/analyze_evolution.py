import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Load data
with open("nas_t/evolutionary_runs.json") as file:
    data = json.load(file)

with open("nas_t/standard_results.json") as file:
    standard_results = json.load(file)

# Prepare data
fitness_data = []
for run in data:
    for generation in run['generations']:
        if 'all_fitnesses' in generation:
            for fitness in generation['all_fitnesses']:
                fitness_data.append({
                    "run": int(run['run_id']),
                    "generation": int(generation['generation']),
                    "fitness": fitness
                })

# Convert to DataFrame
df = pd.DataFrame(fitness_data)

# Generate subplots for all runs
unique_runs = df['run'].unique()
num_runs = len(unique_runs)
fig, axes = plt.subplots(1, num_runs, figsize=(6 * num_runs, 4), sharey=True)  # Adjusted height to 4

if num_runs == 1:
    axes = [axes]  # Ensure axes is iterable if only one plot

for i, run_id in enumerate(unique_runs):
    ax = axes[i]
    subset = df[df['run'] == run_id]

    # Process generations
    generations = sorted([int(gen) for gen in subset['generation'].unique()])
    shifted_generations = [gen - min(generations) for gen in generations]

    # Boxplot for fitness distribution
    boxplot_data = [subset[subset['generation'] == gen]['fitness'].values for gen in generations]
    ax.boxplot(boxplot_data, positions=shifted_generations, widths=0.6)

    # Add standard model baseline
    standard_result = [res for res in standard_results if res['run_id'] == run_id][0]
    mean_fitness = standard_result['final_accuracy']
    ax.axhline(mean_fitness, color='red', linestyle='--', label=f"Standard Model (Run {run_id})")

    # Plot dynamic trendline
    generation_means = [np.mean(subset[subset['generation'] == gen]['fitness'].values) for gen in generations]
    ax.plot(shifted_generations, generation_means, color='orange', marker='o', label='Mean Fitness Trend')

    # Formatting
    ax.set_title(f"Run {run_id}")
    ax.set_xlabel("Generation")
    if i == 0:
        ax.set_ylabel("Fitness")

    # Adjust X-axis with padding
    ax.set_xticks(shifted_generations)
    ax.set_xticklabels(generations)
    ax.set_xlim(min(shifted_generations) - 0.5, max(shifted_generations) + 0.5)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer ticks
    ax.grid(True)
    ax.legend(loc='upper right', fontsize='x-small')

plt.tight_layout()
plt.show()
