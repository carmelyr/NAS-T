"""
- This file contains the unit tests for the NAS-T project.
- The tests are written using the unittest module.
- The tests are written to test the Genotype class, fitness_function, and Phenotype class.
- The tests are written to check the initialization of the Genotype class,
  the conversion of Genotype to Phenotype, and the evaluation of the fitness function.
- The tests are written to check if the Genotype class initializes the architecture
  with valid values, if the Genotype class converts to Phenotype correctly,
  and if the fitness function returns a float value.
- The tests are written to check if the Phenotype class is an instance of Phenotype,
    if the Phenotype class has a PyTorch Sequential model, and if the fitness function
    returns a float value.
- The tests are written to check if the fitness function returns a float value.
"""

import torch.nn as nn
import unittest                                             # imports the unittest module     
from nas_t import Genotype, fitness_function, Phenotype     # imports the Genotype, fitness_function, and Phenotype classes

class TestGenotype(unittest.TestCase):
    # tests genotype initialization
    def test_genotype_initialization(self):
        genotype = Genotype()
        architecture = genotype.architecture
        
        for layer in architecture:
            self.assertIn(layer['layer'], ['Conv', 'MaxPooling', 'Dense', 'Dropout', 'ZeroOp', 'Activation'])
            if 'filters' in layer:
                self.assertIn(layer['filters'], [8, 16, 32, 64, 128, 256])
            if 'units' in layer:
                self.assertIn(layer['units'], [4, 16, 32, 64, 128, 256])
            if 'activation' in layer:
                self.assertIn(layer['activation'], ['softmax', 'relu', 'elu', 'selu', 'sigmoid', 'linear'])

    # method to test genotype to phenotype conversion
    def test_genotype_to_phenotype(self):
        genotype = Genotype()
        model = genotype.to_phenotype()
        
        # checks if model is an instance of Phenotype
        self.assertIsInstance(model, Phenotype)
        
        # checks if model's internal model is a PyTorch Sequential model
        self.assertIsInstance(model.model, nn.Sequential)
        
    # method to test fitness evaluation
    def test_fitness_evaluation(self):
        architecture = [
            {'layer': 'Conv', 'filters': 32, 'kernel_size': 3},
            {'layer': 'MaxPooling', 'pool_size': 2},
            {'layer': 'Dense', 'units': 128, 'activation': 'relu'},
            {'layer': 'Dropout', 'rate': 0.5}
        ]

        validation_accuracy = 0.85  # dummy validation accuracy value for testing
        
        # tests if fitness function returns a float value
        fitness = fitness_function(architecture, validation_accuracy)
        self.assertIsInstance(fitness, float)
        self.assertGreaterEqual(fitness, 0.0)  
        
if __name__ == '__main__':
    unittest.main()
