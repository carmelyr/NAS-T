import pandas as pd
from benchmark_models import benchmark_models

def main():
    print("Starting Model Benchmarking...")
    results_df = benchmark_models()
    print("Benchmarking Completed!")

if __name__ == "__main__":
    main()
