import torch
import torch.nn as nn                                                           # neural network module
import pytorch_lightning as pl
import torch.optim as optim                                                     # optimization algorithms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchmetrics
from utils import fitness_function
from data_handler import train_loader, validation_loader, X_train_tensor
from config import random_architecture

# ---- Genotype class ---- #
# Stores and manipulates the architecture of the neural network
class Genotype:
    """
    - constructor for the Genotype class
    - called when a new instance of the class is created
    - self.config: stores the same architecture of the genotype as defined in self.architecture
    - self.fitness: stores the fitness value of the genotype
    """
    def __init__(self, architecture=None, device='cpu'):
        self.architecture = random_architecture() if architecture is None else architecture
        self.config = self.architecture
        self.device = device
        self.fitness = None

    """
    - method that calculates the fitness score of the architecture based on its performance
    """
    def evaluate(self, generation, max_generations):
        phenotype = self.to_phenotype()

        early_stop_callback = EarlyStopping(monitor='val_acc', patience=12, mode='max')

        trainer = pl.Trainer(min_epochs=20,                         # trains the model for at least 20 epochs
                             max_epochs=200,                        # trains the model for maximum 200 epochs
                             logger=False,
                             accelerator='gpu',                     # uses CPU for training
                             enable_checkpointing=False,
                             enable_progress_bar=False,
                             precision=16,                          # 16-bit precision for faster training
                             callbacks=[early_stop_callback],       # stops training when the validation accuracy does not improve for 15 epochs
                             gradient_clip_val=0.5,                 # prevent exploding gradients
                             gradient_clip_algorithm='norm')        # normalizes the gradient to prevent exploding gradients
        
        # trainer.fit: trains the model
        trainer.fit(phenotype, train_dataloaders=train_loader, val_dataloaders=validation_loader)

        # trainer.validate: evaluates the model on the validation set
        trainer.validate(phenotype, dataloaders=validation_loader)

        # trainer.callback_metrics.get('val_acc', torch.tensor(0.0)): gets the validation accuracy from the trainer
        validation_accuracy = trainer.callback_metrics.get('val_acc', torch.tensor(0.0)).item()
        model_size = phenotype.get_number_parameter()

        self.fitness = fitness_function(self.architecture, validation_accuracy)

        # move the phenotype to CPU to free up GPU memory
        phenotype.to(self.device)

        # clears GPU memory
        #torch.mps.empty_cache()
        return self.fitness, validation_accuracy, model_size

    """
    - method that converts the genotype to a phenotype
    """
    def to_phenotype(self):
        return Phenotype(self.architecture)

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
            self.model = self.build_model_from_genotype(genotype)               # builds the neural network model based on the genotype
        self.loss_fn = nn.CrossEntropyLoss()                                    # calculates the loss between the predictions and the target data
        self.accuracy = torchmetrics.Accuracy(task='binary', num_classes=2)     # calculates the accuracy of the model
        print(f"Number of trainable parameters: {self.get_number_parameter()}")

    """
    - method that returns the number of trainable parameters in the neural network model
    """
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
        out_dim_1 = 1
        out_dim_2 = X_train_tensor.size(1)  # number of time steps in the input data

        out_dim_tracker = [(out_dim_1, out_dim_2)]

        for i, layer in enumerate(genotype):
            if layer['layer'] == 'Conv':
                layers.append(nn.Conv1d(out_dim_1, layer['filters'], kernel_size=layer['kernel_size']))
                out_dim_1 = layer['filters']
                out_dim_2 = out_dim_2 - layer['kernel_size'] + 1
                layers.append(self.get_activation(layer['activation']))

            elif layer['layer'] == 'MaxPooling':
                layers.append(nn.MaxPool1d(kernel_size=layer['pool_size']))
                out_dim_2 = out_dim_2 // layer['pool_size']

            elif layer['layer'] == 'Dense':
                if out_dim_2 != 1:                  # otherwise already flattened
                    layers.append(nn.Flatten())
                    out_dim_tracker.append((out_dim_1 * out_dim_2, 1))
                layers.append(nn.Linear(out_dim_1 * out_dim_2, layer['units']))
                out_dim_1 = layer['units']
                out_dim_2 = 1
                layers.append(self.get_activation(layer['activation']))

            elif layer['layer'] == 'Dropout':
                layers.append(nn.Dropout(layer['rate']))
            else:
                raise("Layer not implemented")
            out_dim_tracker.append((out_dim_1, out_dim_2))
        if out_dim_2 != 1:
            layers.append(nn.Flatten())
            out_dim_tracker.append((out_dim_1*out_dim_2, 1))
        if out_dim_1*out_dim_2 < 1:
            print(f"error: cannot be dimension {out_dim_1*out_dim_2} ({out_dim_1} x {out_dim_2})")
            print(layers)
            print(f"inputs dim 2: {X_train_tensor.size(1)}")
            print(out_dim_tracker)
            exit(32)
        layers.append(nn.Linear(out_dim_1*out_dim_2, 2))
        layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)

    def get_activation(self, activation):
        """
        Returns the activation function corresponding to the given name.
        """
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'selu':
            return nn.SELU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'linear':
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    """
    - method that loads the model from a checkpoint to resume training or evaluation with a previously trained model
    - checkpoint_path: path to the checkpoint file
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
        return self.model(x)

    """
    - method that defines the training step for the neural network
    - calculates the loss and updates the model's weights to improve its predictions
    - batch: tuple containing input data (x) and target data (y) for training
    """
    def training_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx)

        self.log('train_loss', loss, prog_bar=False)    # logs the training loss
        self.log('train_acc', acc, prog_bar=False)      # logs the training accuracy
        return loss

    """
    - method that defines the validation step for the neural network
    - measures how well the model is performing on unseen data without updating its weights
    - calculates the loss and accuracy of the model on the validation set
    """
    def validation_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx)

        self.log('val_loss', loss, prog_bar=False)          # logs the validation loss
        self.log('val_acc', acc, prog_bar=False)            # logs the validation accuracy
        return {'val_loss': loss, 'val_acc': acc}

    # utility method that implements the common processing step for making a prediction of the model
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), 1, x.size(1))  # dim is batch size x channels x sequence length
        logits = self.forward(x)                # passes the input data through the neural network (predictions)

        loss = self.loss_fn(logits, y)          # calculates the loss between the predictions and the target data
        preds = logits.argmax(dim=1)            # returns the index of the maximum value in the predictions
        acc = self.accuracy(preds, y)           # uses torchmetrics.Accuracy to calculate accuracy

        return loss, preds, acc

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

    """
    - method that defines the training epoch end for the neural network
    """
    def on_train_epoch_end(self, unused_outputs=None):
        self.accuracy.reset()
        #torch.mps.empty_cache()  # clears GPU memory after the epoch
    
    """
    - method that defines the validation epoch end for the neural network
    - calculates the average validation accuracy of the model
    """
    def on_validation_epoch_end(self):
        self.log('epoch_val_acc', self.accuracy.compute(), prog_bar=False)
        self.accuracy.reset()
        #torch.mps.empty_cache()  # clears GPU memory after the epoch

    """
    - method that specifies how the model's parameters should be updated during training
    - defines the optimization algorithm -> Adam optimizer
    - adjusts the model's weights during training to minimize the loss
    - improves predictions by using past steps to guide the updates smoothly
    """
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
