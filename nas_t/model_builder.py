import random
import torch
import torch.nn as nn                                                           # neural network module
import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping            # stops training when a monitored quantity has stopped improving
import torchmetrics
from utils import fitness_function
from data_handler import train_loader, validation_loader, X_train_tensor

# ---- Define the random architecture for the neural network based on the values mentioned in the scientific paper ---- #
"""
- Publication of the paper: https://ieeexplore.ieee.org/document/9206721 (Neaural Architecture Search for Time Series Classification)

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


# ---- Genotype class ---- #
# Stores and manipulates the architecture of the neural network
class Genotype:
    """
    - constructor for the Genotype class
    - called when a new instance of the class is created
    - self.config: stores the same architecture of the genotype as defined in self.architecture
    - self.fitness: stores the fitness value of the genotype
    """
    def __init__(self, architecture=None):
        self.architecture = architecture if architecture else random_architecture()
        self.config = self.architecture
        self.fitness = None

    """
    - method that calculates the fitness score of the architecture based on its performance
    """
    def evaluate(self, generation, max_generations):
        phenotype = self.to_phenotype()
        early_stop_callback = EarlyStopping(monitor='val_acc', patience=15, mode='max')

        trainer = pl.Trainer(min_epochs=10,                         # trains the model for at least 10 epochs
                             max_epochs=100,                         # trains the model for 30 epochs; increase
                             logger=False,
                             accelerator='cpu',
                             enable_checkpointing=False,            # disables checkpointing to save the model
                             enable_progress_bar=False,
                             precision=16,                          # uses 16-bit precision for faster training
                             callbacks=[early_stop_callback],       # stops training when the validation accuracy does not improve for 15 epochs
                             gradient_clip_val=0.5,                 # clips the gradient to prevent exploding gradients
                             gradient_clip_algorithm='norm')        # normalizes the gradient to prevent exploding gradients
        
        # trainer.fit: trains the model
        trainer.fit(phenotype, train_dataloaders=train_loader, val_dataloaders=validation_loader)

        # trainer.validate: evaluates the model on the validation set
        trainer.validate(phenotype, dataloaders=validation_loader)

        # trainer.callback_metrics.get('val_acc', torch.tensor(0.0)): gets the validation accuracy from the trainer
        validation_accuracy = trainer.callback_metrics.get('val_acc', torch.tensor(0.0)).item()

        self.fitness = fitness_function(self.architecture, validation_accuracy, generation, max_generations)

        # Move the phenotype to CPU to free up GPU memory
        phenotype.to('cpu')

        # Clear GPU memory
        torch.mps.empty_cache()
        return self.fitness

    """
    - method that converts the genotype to a phenotype
    """
    def to_phenotype(self):
        return Phenotype(self.architecture)

"""
- method that initializes the weights of the neural network model
"""
#def init_weights(m):
#    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
#        nn.init.xavier_uniform_(m.weight)


# ---- Phenotype class ---- #
"""
- defines the neural network model based on the genotype
- inherits from the PyTorch Lightning Module class
- pl.LightningModule: provides a simple interface for organizing PyTorch code; superclass
- Phenotype: subclass of pl.LightningModule that inherits its properties and methods
"""
class Phenotype(pl.LightningModule):

    """
    - constructor for the Phenotype class
    - initializes the class with the genotype and the loss function
    - initializes the accuracy metric using torchmetrics.Accuracy
    """
    def __init__(self, genotype=None):
        super(Phenotype, self).__init__()
        if (genotype):
            self.genotype = genotype
            self.model = self.build_model_from_genotype(genotype)   # builds the neural network model based on the genotype
            #self.model.apply(init_weights)                          # initializes the weights of the neural network model
        self.loss_fn = nn.CrossEntropyLoss()                        # calculates the loss between the predictions and the target data
        self.accuracy = torchmetrics.Accuracy(task='binary', num_classes=2)
        print(f"Number of trainable parameters: {self.get_number_parameter()}")

    def get_number_parameter(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    """
    - method that builds the neural network model based on the genotype
    - variable layers: stores the layers of the neural network
    - variable input_channels: stores the number of input channels
    - variable output_size: stores the size of the output
    - for each layer in the genotype, the corresponding layer is added to the neural network
    - returns the neural network model
    """
    def build_model_from_genotype(self, genotype):
        layers = []
        input_channels = 1
        #output_size = 1201  # todo should be number time steps

        # Dynamically set the output size from the number of features in the input tensor
        output_size = X_train_tensor.size(1)  # Get the number of time steps from the training data tensor

        for layer in genotype:
            if layer['layer'] == 'Conv':
                layers.append(nn.Conv1d(input_channels, layer['filters'], layer['kernel_size']))
                input_channels = layer['filters']
                output_size = output_size - layer['kernel_size'] + 1
                
                if layer['activation'] == 'relu':
                    layers.append(nn.ReLU())
                elif layer['activation'] == 'elu':
                    layers.append(nn.ELU())
                elif layer['activation'] == 'selu':
                    layers.append(nn.SELU())
                elif layer['activation'] == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif layer['activation'] == 'linear':
                    layers.append(nn.Identity())

            elif layer['layer'] == 'MaxPooling':
                layers.append(nn.MaxPool1d(layer['pool_size']))
                output_size = output_size // layer['pool_size']

            elif layer['layer'] == 'Dense':
                layers.append(nn.Flatten())
                layers.append(nn.Linear(input_channels * output_size, layer['units']))
                input_channels = layer['units']
                
                if layer['activation'] == 'relu':
                    layers.append(nn.ReLU())
                elif layer['activation'] == 'elu':
                    layers.append(nn.ELU())
                elif layer['activation'] == 'selu':
                    layers.append(nn.SELU())
                elif layer['activation'] == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif layer['activation'] == 'linear':
                    layers.append(nn.Identity())

            elif layer['layer'] == 'Dropout':
                layers.append(nn.Dropout(layer['rate']))

        #layers.append(nn.Softmax(nn.Linear(input_channels, 2), dim=1))
        layers.append(nn.Linear(input_channels, 2))  # Final linear layer
        layers.append(nn.Softmax(dim=1))             # Apply softmax along the last dimension

        return nn.Sequential(*layers)
    
    """
    - method that loads the model from a checkpoint to resume training or evaluation with a previously trained model
    - checkpoint_path: path to the checkpoint file
    - genotype: genotype of the model
    - kwargs: additional keyword arguments that are passed to the constructor
    """
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, genotype, **kwargs):
        instance = cls(genotype=genotype, **kwargs)
        checkpoint = torch.load(checkpoint_path)
        """
        - instance.load_state_dict: loads the model state from the checkpoint
        - checkpoint['state_dict']: stores the model state in the checkpoint
        - strict=False: ignores the mismatch between the keys in the model and the checkpoint
        """
        instance.load_state_dict(checkpoint['state_dict'], strict=False)
        return instance

    """
    - method that defines how the data flows through the neural network
    - x: input data
    - x.view: reshapes the input data to fit the neural network
    - returns the output of the neural network (transformed data)
    """
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        return self.model(x)

    """
    - method that defines the training step for the neural network
    - calculates the loss and updates the model's weights to improve its predictions
    - batch: tuple containing input data (x) and target data (y) for training
    """
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        logits = self.forward(x)                        # passes the input data through the neural network (predictions) 
        y = y.view(-1)                                  # reshapes the target data to match the predictions
        loss = self.loss_fn(logits, y)                  # calculates the loss between the predictions and the target data
        preds = logits.argmax(dim=1)                    # returns the index of the maximum value in the predictions
        acc = self.accuracy(preds, y)                   # uses torchmetrics.Accuracy to calculate accuracy
        self.log('train_loss', loss, prog_bar=False)     # logs the training loss
        self.log('train_acc', acc, prog_bar=False)       # logs the training accuracy
        return loss

    """
    - method that defines the validation step for the neural network
    - measures how well the model is performing on unseen data without updating its weights
    - calculates the loss and accuracy of the model on the validation set
    """
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        logits = self.forward(x)
        y = y.view(-1)
        loss = self.loss_fn(logits, y)                      # calculates the loss between the predictions and the target data
        preds = logits.argmax(dim=1)    

        # debugging logits and predictions
        #print(f"Logits: {logits[:5]}")
        #print(f"Predictions: {preds[:5]}")
        #print(f"Labels: {y[:5]}")

        acc = self.accuracy(preds, y)                               # uses torchmetrics.Accuracy to calculate accuracy
        self.log('val_loss', loss, prog_bar=False)                   # logs the validation loss
        self.log('val_acc', acc, prog_bar=False)                     # logs the validation accuracy
        return {'val_loss': loss, 'val_acc': acc}
    
    """
    - method that resets the accuracy metric at the start of each training epoch
    """
    def on_train_epoch_start(self):
        self.accuracy.reset()

    """
    - method that resets the accuracy metric at the start of each validation epoch
    """
    def on_validation_epoch_start(self):
        self.accuracy.reset()

    def on_train_epoch_end(self, unused_outputs=None):
        self.accuracy.reset()
        torch.mps.empty_cache()  # clears GPU memory after the epoch
    
    """
    - method that defines the validation epoch end for the neural network
    - calculates the average validation accuracy of the model
    """
    def on_validation_epoch_end(self):
        self.log('epoch_val_acc', self.accuracy.compute(), prog_bar=False)
        self.accuracy.reset()
        torch.mps.empty_cache()  # clears GPU memory after the epoch

    """
    - method that specifies how the model's parameters should be updated during training
    - defines the optimization algorithm -> Adam optimizer
    - adjusts the model's weights during training to minimize the loss
    - improves predictions by using past steps to guide the updates smoothly
    """
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)