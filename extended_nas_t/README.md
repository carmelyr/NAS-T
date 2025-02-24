This folder contains the NAS-T in a splitted files version.

**main.py** runs the the whole implemented framework.
It measures and logs the overall runtime for the evolutionary algorithm.

**config.py** contains hyperparameters and functions for generating random neural network architectures based on predefined layer options.

**data_handler.py** does the data preprocessing, loading, and splitting.
It defines data loaders for training and validation datasets (train_loader and validation_loader)
and ensures compatibility with PyTorch by converting data to tensors.

**model_builder.py** handles the initialization and evaluation of neural network architectures.
It includes genotype-to-phenotype mapping.
It has methods for dynamically building networks based on architecture and managing training procedures.

**evolutionary_algorithm.py** is the differential evolution algorithm for the framework.
It manages a population of neural network architectures (Genotype instances).
It mutates and does the crossover to produce offspring.
It evaluates architectures using fitness values computed based on their accuracy, size, and training time.
It handles evolutionary strategies like selection, mutation, and elitism.

**utils.py** writes evolutionary run results to a JSON file for later analysis.
It evaluates neural network architectures based on validation accuracy, size, and training time,
incorporating penalties for large models or lengthy training.

**run_standard.py** trains and validates a standard neural network architecture.
It saves the results, including final accuracy and model size, to a JSON file for comparison with the evolutionary algorithm results.

**standard_architecture.py** defines the standard neural network architecture used for baseline comparisons.
It includes methods for training, validation, and saving results.

**analyze_evolution.py** analyzes the evolutionary runs by loading the results from JSON files.
It generates visualizations such as boxplots for fitness distribution across generations and compares them with standard model baselines.

**plot_accuracies.py** plots the accuracies of the neural network architectures across generations.
It generates subplots for each run, showing the distribution of accuracies and comparing them with the standard model baseline.

**plot_model_sizes.py** visualizes the model sizes of the neural network architectures across generations.
It generates subplots for each run, showing the distribution of model sizes and comparing them with the standard model baseline.

**fitness_trend.py** plots the runtime trends of the evolutionary algorithm across generations.
It calculates and visualizes the average runtime trend and compares individual run trends.

**standard_results.json** stores each run results from standard architecture.

**accuracies.json** stores each accuracy in the generated architectures for every run.

**model_sizes.json** stores each model size in the generated architectures for every run.

**evolutionary_runs.json** stores for each run:
- all accuracies, model sizes and fitnesses for each generation
- best fitness score, top-performing architecture, and runtime per generation for each run
