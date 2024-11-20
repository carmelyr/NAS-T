import json
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors  # Import mplcursors for interactive annotations
import matplotlib.lines as mlines  # Import for custom legend lines

# Load the JSON file
with open("evolutionary_runs.json") as file:
    data = json.load(file)

# Extract key metrics across generations
fitness_data = []
for run in data:
    for generation in run['generations']:
        # Add the best fitness value of the generation
        fitness_data.append({
            "run": int(run['run_id']),  # Make sure run_id is an integer
            "generation": int(generation['generation']),
            "fitness": generation['best_fitness'],
            "type": "Best"
        })
        # Check if 'all_fitnesses' key exists in the generation dictionary
        if 'all_fitnesses' in generation:
            # Add all other fitness values of the generation
            for fitness in generation['all_fitnesses']:
                fitness_data.append({
                    "run": int(run['run_id']),
                    "generation": int(generation['generation']),
                    "fitness": fitness,
                    "type": "Individual"
                })

# Convert to DataFrame for easier manipulation and analysis
df = pd.DataFrame(fitness_data)

# Calculate total runs and best fitness value
total_runs = df['run'].nunique()  # Corrected: Use nunique to accurately count the number of distinct runs
best_fitness_row = df[df['type'] == 'Best'].loc[df['fitness'].idxmax()]
best_run_id = best_fitness_row['run']
best_generation = best_fitness_row['generation']
best_fitness_value = best_fitness_row['fitness']

# Plot fitness over generations using scatter plot
plt.figure(figsize=(12, 7))

# Plot all individual fitnesses in light pink for each generation
for run_id, group in df[df['type'] == 'Individual'].groupby("run"):
    plt.scatter(group['generation'], group['fitness'], s=20, alpha=0.4, color='lightpink')

# Plot best fitnesses in darker color to distinguish from individual points
for run_id, group in df[df['type'] == 'Best'].groupby("run"):
    scatter = plt.scatter(group['generation'], group['fitness'], s=40, alpha=0.8, color='deeppink')

plt.xlabel("Generation")
plt.ylabel("Fitness")

# Update plot title with total number of runs and which run and generation achieved the best fitness
plt.title(f"Fitness Evolution Over Generations\nTotal Runs: {total_runs}\n Best Fitness Achieved in Run {best_run_id} for Generation {best_generation} (Fitness: {best_fitness_value:.4f})")

# Set x-ticks to be integers for each generation
plt.xticks(ticks=sorted(df['generation'].unique()))

# Add a fixed legend in the bottom left corner
light_pink_marker = mlines.Line2D([], [], color='lightpink', marker='o', linestyle='None', markersize=8, alpha=0.4, label='Individual Fitness (All Runs)')
dark_pink_marker = mlines.Line2D([], [], color='deeppink', marker='o', linestyle='None', markersize=8, alpha=0.8, label='Best Fitness (All Runs)')
plt.legend(handles=[light_pink_marker, dark_pink_marker], loc='lower left')

# Add interactive cursor for displaying fitness values on hover
cursor = mplcursors.cursor(hover=True)  # Enable cursor to hover
cursor.connect("add", lambda sel: sel.annotation.set_text(f"Fitness: {sel.target[1]:.4f}"))

# Customize the annotation appearance
@cursor.connect("add")
def on_add(sel):
    sel.annotation.get_bbox_patch().set(fc="lightpink", alpha=0.8)
    sel.annotation.set_fontsize(10)

plt.show()
