import json
import numpy as np
import matplotlib.pyplot as plt

def plot_runtime_only(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    print("Data loaded from JSON file:")
    print(data)  # debug print

    all_runtime_trends = []
    max_generations = 0
    overall_runtime = 0

    for run in data:
        runtime_trend = []

        for generation in run["generations"]:
            if "runtime" in generation:
                runtime_trend.append(float(generation["runtime"]))
            overall_runtime += float(generation.get("runtime", 0))

        print(f"Runtime trend for run: {runtime_trend}")  # debug print
        all_runtime_trends.append(runtime_trend)
        max_generations = max(max_generations, len(runtime_trend))

    # does padding with NaN values for runs with fewer generations
    for i in range(len(all_runtime_trends)):
        all_runtime_trends[i] += [np.nan] * (max_generations - len(all_runtime_trends[i]))

    # converts the lists to numpy arrays
    all_runtime_trends = np.array(all_runtime_trends)

    # calculates the average runtime trend
    avg_runtime_trend = np.nanmean(all_runtime_trends, axis=0)

    # calculates the overall average runtime
    overall_avg_runtime = np.nanmean(all_runtime_trends)

    print(f"Overall Average Runtime: {overall_avg_runtime}")

    # plots runtime trends
    fig, ax = plt.subplots(figsize=(8, 6))

    # plots each run's runtime trend with a unique color
    colormap = plt.cm.get_cmap('tab10', len(all_runtime_trends))
    for i, runtime_trend in enumerate(all_runtime_trends):
        ax.plot(runtime_trend, label=f'Run {i+1}', color=colormap(i))

    # plots overall average runtime trend as a dashed line
    ax.plot(avg_runtime_trend, label='Average Runtime', color='black', linestyle='--')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Runtime in seconds')
    ax.set_title('Runtime')
    ax.legend(loc='upper right', fontsize='x-small')
    ax.set_ylim(0, 300)
    ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_xticks(range(max_generations))
    ax.set_xticklabels(range(1, max_generations + 1))

    plt.tight_layout()
    plt.show()

plot_runtime_only('nas_t/evolutionary_runs.json')
