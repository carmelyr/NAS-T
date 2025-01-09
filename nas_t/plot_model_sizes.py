import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

def plot_model_sizes_with_boxplots(model_sizes_file, standard_results_file):
    with open(model_sizes_file, 'r') as file:
        model_sizes_data = json.load(file)

    with open(standard_results_file, 'r') as file:
        standard_results = json.load(file)

    # Prepare data
    size_data = []
    for run_idx, run in enumerate(model_sizes_data):
        for gen_idx, generation in enumerate(run["generations"]):
            sizes = generation["model_sizes"]
            for size in sizes:
                size_data.append({
                    "run": run_idx + 1,
                    "generation": gen_idx + 1,
                    "model_size": int(size)  # Ensure model sizes are integers
                })

    # Convert to DataFrame
    df = pd.DataFrame(size_data)

    # Generate subplots for all runs
    unique_runs = df['run'].unique()
    num_runs = len(unique_runs)
    fig, axes = plt.subplots(1, num_runs, figsize=(6 * num_runs, 4), sharey=True)

    if num_runs == 1:
        axes = [axes]

    for i, run_id in enumerate(unique_runs):
        ax = axes[i]
        subset = df[df['run'] == run_id]

        # Process generations
        generations = sorted([int(gen) for gen in subset['generation'].unique()])
        shifted_generations = [gen - min(generations) for gen in generations]

        # Boxplot for model size distribution
        boxplot_data = [subset[subset['generation'] == gen]['model_size'].values for gen in generations]
        ax.boxplot(boxplot_data, positions=shifted_generations, widths=0.6,
                   medianprops=dict(color='deepskyblue', linestyle='--', linewidth=1))

        # Add standard model baseline
        standard_result = [res for res in standard_results if res['run_id'] == run_id][0]
        standard_model_size = int(standard_result['final_model_size'])
        ax.axhline(standard_model_size, color='red', linestyle='--', linewidth=2, label=f"Standard Model (Run {run_id})")

        # Plot mean model size trendline
        generation_means = [np.mean(data) for data in boxplot_data]
        ax.plot(shifted_generations, generation_means, color='plum', marker='o', markersize=5,
                linestyle='-', label='Mean Model Size Trend', zorder=3)

        # Formatting
        ax.set_title(f"Run {run_id}")
        ax.set_xlabel("Generation")
        if i == 0:
            ax.set_ylabel("Model Size")

        # Adjust X-axis with padding
        ax.set_xticks(shifted_generations)
        ax.set_xticklabels(generations)
        ax.set_xlim(min(shifted_generations) - 0.5, max(shifted_generations) + 0.5)

        # Set y-axis starting at 0 and adjust dynamically
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer ticks
        ax.grid(True)

    # Add a shared legend with custom entries
    custom_legend = [
        Line2D([0], [0], color='plum', marker='o', linestyle='-', label='Mean Model Size Trend'),
        Line2D([0], [0], color='deepskyblue', linestyle='--', linewidth=2, label='Median Model Size'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Standard Model Baseline')
    ]
    fig.legend(handles=custom_legend, loc='upper center', ncol=3, fontsize='small', frameon=True)

    # Adjust layout to add more left padding
    plt.tight_layout(rect=[0.03, 0, 0.99, 0.9])
    plt.show()

# Example call
plot_model_sizes_with_boxplots('nas_t/model_sizes.json', 'nas_t/standard_results.json')
