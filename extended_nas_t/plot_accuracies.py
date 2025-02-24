import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

with open("extended_nas_t/accuracies.json") as file:
    accuracies_data = json.load(file)

with open("extended_nas_t/standard_results.json") as file:
    standard_results = json.load(file)

# prepares accuracy data
accuracy_data = []
for run in accuracies_data:
    for generation in run['generations']:
        for acc in generation['accuracies']:
            accuracy_data.append({
                "run": int(run['run_id']),
                "generation": int(generation['generation']),
                "accuracy": acc
            })

# converts to DataFrame
df = pd.DataFrame(accuracy_data)

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

    # boxplot for accuracy distribution
    boxplot_data = [subset[subset['generation'] == gen]['accuracy'].values for gen in generations]
    ax.boxplot(boxplot_data, positions=shifted_generations, widths=0.6,
               medianprops=dict(color='deepskyblue', linestyle='--', linewidth=1))

    # adds standard model baseline
    standard_result = [res for res in standard_results if res['run_id'] == run_id]
    if standard_result:
        standard_accuracy = standard_result[0]['final_accuracy']
        ax.axhline(standard_accuracy, color='red', linestyle='--', linewidth=2, label=f"Standard Model (Run {run_id})")

    # plots mean accuracy trendline
    generation_means = [np.mean(data) for data in boxplot_data]
    ax.plot(shifted_generations, generation_means, color='plum', marker='o', markersize=5,
            linestyle='-', label='Mean Accuracy Trend', zorder=3)

    # formatting
    ax.set_title(f"Run {run_id}")
    ax.set_xlabel("Generation")
    if i == 0:
        ax.set_ylabel("Accuracy")

    ax.set_xticks(shifted_generations)
    ax.set_xticklabels(generations)
    ax.set_xlim(min(shifted_generations) - 0.5, max(shifted_generations) + 0.5)
    ax.set_ylim(0, 1.0)
    ax.grid(True)

# legend
custom_legend = [
    Line2D([0], [0], color='plum', marker='o', linestyle='-', label='Mean Accuracy Trend'),
    Line2D([0], [0], color='deepskyblue', linestyle='--', label='Median Accuracy'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Standard Model Baseline')
]
fig.legend(handles=custom_legend, loc='upper center', ncol=3, fontsize='small', frameon=True)

plt.tight_layout(rect=[0.03, 0, 0.99, 0.9])
plt.show()
