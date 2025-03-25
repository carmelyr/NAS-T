import numpy as np
import pandas as pd
from data_handler import get_data_splits
from evolutionary_algorithm import NASDifferentialEvolution
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def main():
    print("Starting Evolutionary Model Checking...")
    
    # Load and preprocess data
    X = pd.read_csv('classification_ozone/X_train.csv').fillna(0)
    y = pd.read_csv('classification_ozone/y_train.csv')
    
    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert y to numpy and ensure proper one-hot encoding
    y = y.to_numpy().reshape(-1, 1)  # Ensure 2D shape
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)
    
    # Verify encoding
    print("Class distribution:", np.unique(np.argmax(y_onehot, axis=1), return_counts=True))
    print("Sample one-hot encoding:", y_onehot[:5])
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y_onehot, dtype=np.float32)
    
    # Initialize and run evolution
    nas = NASDifferentialEvolution(population_size=5, generations=3, verbose=True)
    nas.evolve_and_check(X, y, input_size=X.shape[1])
    
    print("Evolutionary Model Checking Completed!")

if __name__ == "__main__":
    main()