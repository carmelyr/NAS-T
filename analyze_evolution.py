import json
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import matplotlib.lines as mlines
import matplotlib.cm as cm
import numpy as np
from matplotlib.ticker import MaxNLocator

# ---- Loads the data from the evolutionary_runs.json file ---- #
with open("evolutionary_runs.json") as file:
    data = json.load(file)

# ---- Extracts the fitness data from the JSON file ---- #
fitness_data = []
for run in data:
    for generation in run['generations']:
        fitness_data.append({
            "run": int(run['run_id']),
            "generation": int(generation['generation']),
            "fitness": generation['best_fitness'],
            "type": "Best"
        })
        if 'all_fitnesses' in generation:
            for fitness in generation['all_fitnesses']:
                fitness_data.append({
                    "run": int(run['run_id']),
                    "generation": int(generation['generation']),
                    "fitness": fitness,
                    "type": "Individual"
                })

# ---- Converts the fitness data into a pandas DataFrame ---- #
df = pd.DataFrame(fitness_data)

total_runs = df['run'].nunique()
best_fitness_row = df[df['type'] == 'Best'].loc[df['fitness'].idxmax()]
best_run_id = best_fitness_row['run']
best_generation = best_fitness_row['generation']
best_fitness_value = best_fitness_row['fitness']

# ---- Plots the fitness evolution over generations ---- #
plt.figure(figsize=(13, 7))

# ---- Colors for each run ---- #
colors = cm.get_cmap('tab10', total_runs)
color_map = {run_id: colors(i % colors.N) for i, run_id in enumerate(sorted(df['run'].unique()))}

# ---- Scatter plot for each run ---- #
for run_id, group in df[df['type'] == 'Individual'].groupby("run"):
    plt.scatter(group['generation'], group['fitness'], s=20, alpha=0.4, color=color_map[run_id])

# ---- Scatter plot for the best individual of each run ---- #
for run_id, group in df[df['type'] == 'Best'].groupby("run"):
    plt.scatter(group['generation'], group['fitness'], s=40, alpha=0.8, color=color_map[run_id], label=f'Run {run_id} Best')

plt.xlabel("Generation")
plt.ylabel("Fitness")

# ---- Sets the y-axis limits to [0, 1] ---- #
plt.ylim(0, 1.0)

# ---- Set the x-axis to use integer ticks ---- #
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# ---- Set x-axis ticks to display all generations explicitly ---- #
generations = df['generation'].unique()
plt.xticks(generations)

plt.title(f"Fitness Evolution Over Generations\nTotal Runs: {total_runs}\n Best Fitness Achieved in Run {best_run_id} for Generation {best_generation} (Fitness: {best_fitness_value:.6f})")

# ---- Add markers for the legend ---- #
run_markers = []
for run_id in range(1, total_runs + 1):
    run_markers.append(mlines.Line2D([], [], color=color_map[run_id], marker='o', linestyle='None', markersize=5, alpha=0.4, label=f'Run {run_id} Individual'))
    run_markers.append(mlines.Line2D([], [], color=color_map[run_id], marker='o', linestyle='None', markersize=5, alpha=0.8, label=f'Run {run_id} Best'))

# ---- Add the legend to the plot ---- #
plt.legend(handles=run_markers, loc='lower left', fontsize=8)

# ---- Adds annotations to the plot ---- #
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f"Fitness: {sel.target[1]:.6f}"))

# ---- Adds a pink background to the annotations ---- #
@cursor.connect("add")
def on_add(sel):
    sel.annotation.get_bbox_patch().set(fc="lightpink", alpha=0.8)
    sel.annotation.set_fontsize(10)

plt.show()

