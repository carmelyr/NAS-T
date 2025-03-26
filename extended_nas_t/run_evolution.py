import numpy as np
import pandas as pd
from evolutionary_algorithm import NASDifferentialEvolution
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Load data
    X = pd.read_csv('classification_ozone/X_train.csv').fillna(0)
    y = pd.read_csv('classification_ozone/y_train.csv')
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # One-hot encode labels
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.to_numpy().reshape(-1, 1))
    
    return X, y

def main():
    print("Starting Enhanced Evolutionary Optimization...")
    
    X, y = load_and_preprocess_data()
    
    # Initialize and run evolution
    nas = NASDifferentialEvolution(
        population_size=15,
        generations=30,  # Increased max generations
        verbose=True
    )
    
    best_model = nas.evolve_and_check(X, y, input_size=X.shape[1])
    
    print("\nFinal Best Model:")
    print(f"Configuration: {best_model}")
    print(f"Best Accuracy: {nas.best_accuracy:.4f}")
    
    # Save final best model information
    with open("best_model.txt", "w") as f:
        f.write(f"Best Model Configuration:\n{best_model}\n")
        f.write(f"Best Accuracy: {nas.best_accuracy:.4f}\n")
    
    print("Evolutionary Optimization Completed!")

if __name__ == "__main__":
    main()