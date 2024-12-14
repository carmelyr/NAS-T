import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load data
X_analysis = pd.read_csv('classification_ozone/X_train.csv')
y_analysis = pd.read_csv('classification_ozone/y_train.csv')
X_test = pd.read_csv('classification_ozone/X_test.csv')
y_test = pd.read_csv('classification_ozone/y_test.csv')

# Fill NaNs
X_analysis.fillna(X_analysis.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Split data
X_train, X_validation, y_train, y_validation = train_test_split(X_analysis, y_analysis, test_size=0.2, random_state=42)

# ---- Parameters for repeated k-fold cross-validation ---- #
# n_folds = 5       # number of folds
# n_repeats = 3     # number of repeats
# rkf = RepeatedKFold(n_folds=n_folds, n_repeats=n_repeats, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy().flatten(), dtype=torch.long)
X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
y_validation_tensor = torch.tensor(y_validation.to_numpy().flatten(), dtype=torch.long)

# Create datasets and loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)
