This folder contains the NAS-T in a splitted files version.

**main.py** runs the the whole implemented framework.
It measures and logs the overall runtime for the evolutionary algorithm.

**config.py** contains hyperparameters.

**data_handler.py** does the data preprocessing, loading, and splitting.
It defines data loaders for training and validation datasets (train_loader and validation_loader)
and ensures compatibility with PyTorch by converting data to tensors.

**model_builder.py** handles the generation and evaluation of neural network architectures.
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

