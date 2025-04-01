import pandas as pd

def rank_evolution_results(csv_file='evolution_results.csv', output_file='ranked_evolution_results.csv'):
    """
    Ranks all entries in evolution_results.csv by 'Validation Accuracy' (descending),
    adds a 'Rank' column, and saves the sorted results to a new CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Sort by accuracy (descending) and reset index
    ranked_df = df.sort_values(by='Validation Accuracy', ascending=False)
    ranked_df.reset_index(drop=True, inplace=True)
    
    # Add Rank column (1 = best accuracy)
    ranked_df.insert(0, 'Rank', ranked_df.index + 1)
    
    # Save to new CSV (without index)
    ranked_df.to_csv(output_file, index=False)
    
    # Print confirmation
    print(f"Ranked results saved to '{output_file}'. Top 5 entries:")
    print(ranked_df[['Rank', 'Run ID', 'Generation', 'Validation Accuracy']].head())
    
    return ranked_df

if __name__ == "__main__":
    rank_evolution_results()