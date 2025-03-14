import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

# loads the data from the classification_ozone folder
X_analysis = pd.read_csv('classification_ozone/X_train.csv')
y_analysis = pd.read_csv('classification_ozone/y_train.csv')
X_test = pd.read_csv('classification_ozone/X_test.csv')
y_test = pd.read_csv('classification_ozone/y_test.csv')

# data preprocessing -> data cleaning: fills missing values with the column-wise mean
X_analysis.fillna(X_analysis.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# scales data using z-score normalization
scaler = StandardScaler()
X_analysis = scaler.fit_transform(X_analysis)
X_test = scaler.transform(X_test)

y_analysis = y_analysis.to_numpy().flatten()
y_test = y_test.to_numpy().flatten()

# Prepare datasets for cross-validation
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
def get_data_splits():
    for train_idx, val_idx in rkf.split(X_analysis):
        X_train, X_validation = X_analysis[train_idx], X_analysis[val_idx]
        y_train, y_validation = y_analysis[train_idx], y_analysis[val_idx]
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
        y_validation_tensor = torch.tensor(y_validation, dtype=torch.long)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        yield train_loader, validation_loader