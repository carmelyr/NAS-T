import random

# Hyperparameters
population_size = 15
generations = 6
F = 0.8                 # mutation factor
CR = 0.9                # crossover rate
alpha = 0.0001          # size penalty
BETA = 0.00001          # time penalty

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
# ---- Fixed layer amount and no repeated layers ---- #
def random_architecture(n=5):
    """
    Generate a random architecture with n unique layers, ensuring no layer type repeats.
    """
    layer_options = [
        {'layer': 'Conv', 'filters': [8, 16, 32, 64, 128], 'kernel_size': [3, 5],
         'activation': ['relu', 'elu', 'selu', 'sigmoid', 'linear']},
        {'layer': 'ZeroOp'},
        {'layer': 'MaxPooling', 'pool_size': [2, 3]},
        {'layer': 'Dense', 'units': [16, 32, 64, 128],
         'activation': ['relu', 'elu', 'selu', 'sigmoid', 'linear']},
        {'layer': 'Dropout', 'rate': (0.1, 0.5)},
        {'layer': 'Activation', 'activation': ['softmax', 'elu', 'selu', 'relu', 'sigmoid', 'linear']}
    ]

    # Randomly shuffle and select n unique layers
    selected_layers = random.sample(layer_options, min(n, len(layer_options)))

    architecture = []
    for layer in selected_layers:
        layer_config = {'layer': layer['layer']}
        for key, value in layer.items():
            if key == 'layer':
                continue
            if isinstance(value, list):
                layer_config[key] = random.choice(value)
            elif isinstance(value, tuple):
                layer_config[key] = random.uniform(*value)
        architecture.append(layer_config)

    return architecture