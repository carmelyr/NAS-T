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
        {'layer': 'GRU', 'hidden_units': [16, 32, 64, 128]}
    ]

    selected_layers = []
    only_linear = False

    first_layer = random.choice([layer_options[0], layer_options[3]])  # Conv or Dense
    selected_layers.append(first_layer)

    for i in range(n-1):
        random_number = random.random()

        # Prevent RNN layers before Conv layers
        if len(selected_layers) > 0 and selected_layers[-1]['layer'] in ['LSTM', 'GRU']:
            random_number += 0.3  # Reduce chance of LSTM/GRU appearing in a row

        if random_number < 0.5 and not only_linear:     # select Convolutional block
            selected_layers.append(layer_options[0])    # conv
            selected_layers.append(layer_options[2])    # max pooling
        elif random_number < 0.6:
            selected_layers.append(layer_options[4])    # dropout
        elif random_number < 0.7:
            selected_layers.append(layer_options[5])    # LSTM
        elif random_number < 0.8:
            selected_layers.append(layer_options[6])    # GRU
        elif random_number < 0.9 and only_linear:
            selected_layers.append(layer_options[3])    # dense
            only_linear = True
        else:
            selected_layers.append(layer_options[1])    # zeroop

    # Ensure valid transitions from LSTM/GRU to Conv
    #if any(layer['layer'] in ['LSTM', 'GRU'] for layer in selected_layers):
    #    if any(layer['layer'] == 'Conv' for layer in selected_layers):
    #        print("Skipping invalid architecture with LSTM/GRU before Conv")
    #        return random_architecture()  # Regenerate a valid architecture
    
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
        else:
            print(f"Invalid layer configuration: {layer}")
        layer_config['layer'] = layer['layer']
        architecture.append(layer_config)

    return architecture