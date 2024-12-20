import random

# Hyperparameters
population_size = 15
generations = 6
F = 0.8                 # mutation factor
CR = 0.9                # crossover rate
alpha = 0.0001          # size penalty
BETA = 0.00001          # time penalty
device = 'mps'          # device to run the model on (mps: multi-processing server, cuda: GPU, cpu: CPU)

# ---- Defines the random architecture for the neural network based on the values mentioned in the scientific paper ---- #
"""
- Publication of the paper: https://ieeexplore.ieee.org/document/9206721 (Neural Architecture Search for Time Series Classification)
- Conv layer detects patterns in the input data by applying filters to the input
- ZeroOp layer skips or ignores the next layer
- MaxPooling layer reduces the size of the data by taking only the maximum value in a window
- Dense layer connects all input neurons to all output neurons
- Dropout layer randomly sets a fraction of input units to zero to prevent overfitting
- Activation layer applies an activation function to the output of the previous layer
"""
def random_architecture():
    return [
        {'layer': 'Conv', 'filters': random.choice([8, 16, 32, 64, 128]), 'kernel_size': random.choice([3, 5]),
         'activation': random.choice(['relu', 'elu', 'selu', 'sigmoid', 'linear'])},
        {'layer': 'ZeroOp'},
        {'layer': 'MaxPooling', 'pool_size': random.choice([2, 3])},
        {'layer': 'Dense', 'units': random.choice([16, 32, 64, 128]),
         'activation': random.choice(['relu', 'elu', 'selu', 'sigmoid', 'linear'])},
        {'layer': 'Dropout', 'rate': random.uniform(0.1, 0.5)},
        {'layer': 'Activation', 'activation': random.choice(['softmax', 'elu', 'selu', 'relu', 'sigmoid', 'linear'])}
    ]