import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

with open("nas_t/evolutionary_runs.json") as file:
    data = json.load(file)

with open("nas_t/standard_results.json") as file:
    standard_results = json.load(file)

# prepares data
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

# converts to DataFrame
df = pd.DataFrame(fitness_data)

# generates subplots for all runs
unique_runs = df['run'].unique()
num_runs = len(unique_runs)
fig, axes = plt.subplots(1, num_runs, figsize=(6 * num_runs, 4), sharey=True)

if num_runs == 1:
    axes = [axes]

for i, run_id in enumerate(unique_runs):
    ax = axes[i]
    subset = df[df['run'] == run_id]

    # process generations
    generations = sorted([int(gen) for gen in subset['generation'].unique()])
    shifted_generations = [gen - min(generations) for gen in generations]

    # boxplot for fitness distribution
    boxplot_data = [subset[subset['generation'] == gen]['fitness'].values for gen in generations]
    boxplot = ax.boxplot(boxplot_data, positions=shifted_generations, widths=0.6,
                         medianprops=dict(color='deepskyblue', linestyle='--', linewidth=1))

    # adds standard model baseline
    standard_result = [res for res in standard_results if res['run_id'] == run_id][0]
    mean_fitness = standard_result['final_accuracy']
    ax.axhline(mean_fitness, color='red', linestyle='--', linewidth=2, label=f"Standard Model (Run {run_id})")

    # plots mean fitness trendline (aligned with boxplot centers)
    generation_means = [np.mean(data) for data in boxplot_data]
    ax.plot(shifted_generations, generation_means, color='plum', marker='o', markersize=5,
            linestyle='-', label='Mean Fitness Trend', zorder=3)

    # debugging prints
    print(f"Run {run_id} Debug Info:")
    print(f"Generations: {generations}")
    print(f"Boxplot Data: {boxplot_data}")
    print(f"Mean Fitness Values: {generation_means}")

    # formatting
    ax.set_title(f"Run {run_id}")
    ax.set_xlabel("Generation")
    if i == 0:
        ax.set_ylabel("Fitness")

    ax.set_xticks(shifted_generations)
    ax.set_xticklabels(generations)
    ax.set_xlim(min(shifted_generations) - 0.5, max(shifted_generations) + 0.5)

    ax.set_ylim(bottom=0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)

# legend
custom_legend = [
    Line2D([0], [0], color='plum', marker='o', linestyle='-', label='Mean Fitness Trend'),
    Line2D([0], [0], color='deepskyblue', linestyle='--', linewidth=2, label='Median Fitness'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Standard Model Baseline')
]
fig.legend(handles=custom_legend, loc='upper center', ncol=3, fontsize='small', frameon=True)

plt.tight_layout(rect=[0.03, 0, 0.99, 0.9])
plt.show()
