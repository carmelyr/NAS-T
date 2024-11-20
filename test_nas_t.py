import torch.nn as nn
import unittest
from nas_t import Genotype, fitness_function

class TestGenotype(unittest.TestCase):
    
    # tests genotype initialization
    def test_genotype_initialization(self):
        genotype = Genotype()
        architecture = genotype.architecture
        
        for layer in architecture:
            self.assertIn(layer['layer'], ['Conv', 'MaxPooling', 'Dense', 'Dropout'])
            if 'filters' in layer:
                self.assertIn(layer['filters'], [8, 16, 32, 64, 128, 256])
            if 'units' in layer:
                self.assertIn(layer['units'], [4, 16, 32, 64, 128, 256])
            if 'activation' in layer:
                self.assertIn(layer['activation'], ['softmax', 'relu', 'elu', 'selu', 'sigmoid', 'linear'])

    # method to test genotype evaluation
    def test_genotype_to_phenotype(self):
        genotype = Genotype()
        model = genotype.to_phenotype()
        
        # checks if model is a PyTorch Sequential model
        self.assertIsInstance(model, nn.Sequential)
        
    # method to test genotype evaluation
    def test_fitness_evaluation(self):
        architecture = [
            {'layer': 'Conv', 'filters': 32, 'kernel_size': 3},
            {'layer': 'MaxPooling', 'pool_size': 2},
            {'layer': 'Dense', 'units': 128, 'activation': 'relu'},
            {'layer': 'Dropout', 'rate': 0.5}
        ]
        
        # tests if fitness function returns a float value
        fitness = fitness_function(architecture)
        self.assertIsInstance(fitness, float)
        self.assertGreaterEqual(fitness, -1)
        
if __name__ == '__main__':
    unittest.main()
