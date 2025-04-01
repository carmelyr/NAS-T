import random
import torch

# Hyperparameters
population_size = 5    # number of individuals in the population
generations = 3
initial_F = 0.8       # Starting mutation factor
final_F = 0.3         # Minimum mutation factor
initial_CR = 0.9      # Starting crossover rate
final_CR = 0.4        # Ending crossover rate
decay_rate = 0.85     # Exponential decay rate
alpha = 0.0000001       # size penalty
BETA = 0.00000001        # time penalty
num_folds = 5           # number of folds for cross-validation
num_repeats = 1         # number of repeats for cross-validation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'mps'          # device to run the model on (mps: multi-processing server, cuda: GPU, cpu: CPU)
n = 7                   # number of layers in the neural network

# ---- Defines the random architecture for the neural network based on the values mentioned in the scientific paper ---- #
"""
- Publication of the paper: https://ieeexplore.ieee.org/document/9206721 (Neural Architecture Search for Time Series Classification)
- Conv layer detects patterns in the input data by applying filters to the input
- ZeroOp layer skips or ignores the next layer
- MaxPooling layer reduces the size of the data by taking only the maximum value in a window
- Dense layer connects all input neurons to all output neurons
- Dropout layer randomly sets a fraction of input units to zero to prevent overfitting
- Activation layer applies an activation function to the output of the previous layer
- LSTM layer adds recurrent connections to capture temporal dependencies
- GRU layer adds gated recurrent connections to capture temporal dependencies
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
        {'layer': 'Dropout', 'rate': (0.1, 0.5)},
        {'layer': 'LSTM', 'hidden_units': [16, 32, 64, 128]},
        {'layer': 'GRU', 'hidden_units': [16, 32, 64, 128]},
        {'layer': 'Ensure3D'}  # Add Ensure3D layer option
    ]

    selected_layers = []
    only_linear = False
    is_flattened = False

    first_layer = random.choice([layer_options[0], layer_options[3]])  # Conv or Dense
    selected_layers.append(first_layer)
    if first_layer['layer'] == 'Dense':
        is_flattened = True

    for i in range(n-1):
        random_number = random.random()

        # prevents RNN layers before Conv layers
        #if len(selected_layers) > 0 and selected_layers[-1]['layer'] in ['LSTM', 'GRU']:
        #    random_number += 0.1

        if random_number < 0.3 and not only_linear:
            selected_layers.append(layer_options[0])    # conv
            selected_layers.append(layer_options[2])    # max pooling
        elif random_number < 0.4:
            selected_layers.append(layer_options[4])    # dropout
        elif random_number < 0.55 and not is_flattened:
            selected_layers.append(layer_options[5])    # LSTM
            is_flattened = True
        elif random_number < 0.65 and not is_flattened:
            selected_layers.append(layer_options[6])    # GRU
            is_flattened = True
        elif random_number < 0.85:
            selected_layers.append(layer_options[3])    # dense
            only_linear = True
        else:
            selected_layers.append(layer_options[1])    # zeroop

    # ensures valid transitions from Dense to Conv and MaxPooling
    if any(layer['layer'] == 'Dense' for layer in selected_layers):
        for i, layer in enumerate(selected_layers):
            if layer['layer'] == 'Dense' and i < len(selected_layers) - 1:
                next_layer = selected_layers[i + 1]
                if next_layer['layer'] in ['Conv', 'MaxPooling']:
                    selected_layers.insert(i + 1, {'layer': 'Ensure3D'})

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
        elif layer['layer'] == 'LSTM':
            layer_config['hidden_units'] = random.choice(layer['hidden_units'])
        elif layer['layer'] == 'GRU':
            layer_config['hidden_units'] = random.choice(layer['hidden_units'])
        elif layer['layer'] == 'Ensure3D':
            pass  # No additional configuration needed for Ensure3D
        else:
            print(f"Invalid layer configuration: {layer}")
        layer_config['layer'] = layer['layer']
        architecture.append(layer_config)

    return architecture