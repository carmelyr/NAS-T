import random
import torch

# Hyperparameters
population_size = 10    # number of individuals in the population
generations = 5
F = 0.6                 # mutation factor
CR = 0.7                # crossover rate
alpha = 0.0001          # size penalty
BETA = 0.00001          # time penalty
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'mps'          # device to run the model on (mps: multi-processing server, cuda: GPU, cpu: CPU)
n = 5                   # number of layers in the neural network

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
    # generate a random architecture with exactly n layers.
    layer_options = [
        {'layer': 'Conv', 'filters': [8, 16, 32, 64, 128], 'kernel_size': [3, 5],
         'activation': ['relu', 'elu', 'selu', 'sigmoid', 'linear']},
        {'layer': 'ZeroOp'},
        {'layer': 'MaxPooling', 'pool_size': [2, 3]},
        {'layer': 'Dense', 'units': [16, 32, 64, 128],
         'activation': ['relu', 'elu', 'selu', 'sigmoid', 'linear']},
        {'layer': 'Dropout', 'rate': (0.1, 0.5)}
        # {'layer': 'Activation', 'activation': ['softmax', 'elu', 'selu', 'relu', 'sigmoid', 'linear']}
    ]

    selected_layers = []
    only_linear = False

    for i in range(n):
        random_number = random.random()
        if random_number < 0.6 and not only_linear:     # select Convolutional block
            selected_layers.append(layer_options[0])    # conv
            selected_layers.append(layer_options[2])    # max pooling
        elif random_number < 0.7:
            selected_layers.append(layer_options[4])    # dropout
        elif random_number < 0.9 and only_linear:
            selected_layers.append(layer_options[3])    # dense
            only_linear = True
        else:
            selected_layers.append(layer_options[1])    # zeroop

    # makes sure that the architecture has exactly n layers; takes the first n layers (slicing)
    selected_layers = selected_layers[:n]

    architecture = []
    for layer in selected_layers:
        layer_config = {}
        if layer['layer'] == 'ZeroOp':  # Skip ZeroOp layers -> no operation; the end result may have less than n layers because of this
            continue
        elif layer['layer'] == 'Conv':
            layer_config['filters'] = random.choice(layer['filters'])
            layer_config['kernel_size'] = random.choice(layer['kernel_size'])
            layer_config['activation'] = random.choice(layer['activation'])
        elif layer['layer'] == 'MaxPooling':
            layer_config['pool_size'] = random.choice(layer['pool_size'])
        elif layer['layer'] == 'Dense':
            layer_config['units'] = random.choice(layer['units'])
            layer_config['activation'] = random.choice(layer['activation'])
        elif layer['layer'] == 'Dropout':
            layer_config['rate'] = random.uniform(layer['rate'][0], layer['rate'][1])
        else:
            print(f"Invalid layer configuration: {layer}")
        layer_config['layer'] = layer['layer']
        architecture.append(layer_config)

    return architecture