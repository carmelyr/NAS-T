import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from data_handler import X_train_tensor, y_train_tensor, X_validation_tensor, y_validation_tensor
from standard_architecture import StandardArchitecture

# prepares DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)

# gets input size from data
input_size = X_train_tensor.size(1)

# initializes model with input size
model = StandardArchitecture(input_size)

# defines training parameters
trainer = pl.Trainer(max_epochs=200, logger=False, enable_checkpointing=False, enable_progress_bar=True)

# trains and validates the model
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
trainer.validate(model, dataloaders=validation_loader)

# saves results to JSON after training and validation
model.save_results('extended_nas_t/standard_results.json')

print("Results saved to 'extended_nas_t/standard_results.json'")
