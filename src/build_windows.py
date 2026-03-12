import pandas as pd
import numpy as np
import os

def create_window_features(df, W=30, H=10):
    """
    Transforms raw time-series metrics into a standard Machine Learning dataset 
    (Feature Matrix X and Target Vector y) using a sliding window.
    
    CONCEPT:
    At any given minute 't', we look back at the past 'W' minutes to calculate features.
    Then, we look forward at the next 'H' minutes. If an incident happens anywhere in 
    those 'H' minutes, we label our current window as '1' (Danger approaching).
    
    Args:
        df (pd.DataFrame): Raw metrics data.
        W (int): Window size (history to look at). Default is 30 minutes.
        H (int): Horizon (future to predict). Default is 10 minutes.
        
    Returns:
        tuple: (X_DataFrame, y_Series)
    """
    print(f"Applying sliding window approach: Looking back W={W} mins, predicting H={H} mins ahead.")
    
    # These are the raw columns we will extract features from
    metrics = ['cpu_usage', 'memory_usage', 'latency', 'error_rate']
    
    X = [] # List to store dictionaries of features
    y = [] # List to store binary targets
    
    # Loop through the dataset. 
    # Start at index W (we need enough history).
    # End at len(df) - H (we need enough future to verify if an incident occurred).
    for i in range(W, len(df) - H):
        
        # --------------------------------------------------
        # FEATURE EXTRACTION (The "X" variables)
        # --------------------------------------------------
        # Extract the historical window of size W
        window = df.iloc[i-W:i]
        features = {}
        
        for col in metrics:
            vals = window[col].values
            
            # Instead of feeding raw 30 minutes to the model (which causes high dimensionality),
            # we compress the 30 minutes into meaningful statistical summaries:
            features[f'{col}_mean'] = np.mean(vals) # Average load over the window
            features[f'{col}_std'] = np.std(vals)   # Volatility/instability of the metric
            features[f'{col}_min'] = np.min(vals)   # Lowest value recorded
            features[f'{col}_max'] = np.max(vals)   # Peak load recorded
            features[f'{col}_last'] = vals[-1]      # The most recent value right now
            
            # Trend calculation: Is it going up or down? (Last value minus First value)
            features[f'{col}_diff'] = vals[-1] - vals[0] 
            
        X.append(features)
        
        # --------------------------------------------------
        # TARGET FORMULATION (The "y" variable)
        # --------------------------------------------------
        # Extract the future window of size H
        future_window = df.iloc[i:i+H]
        
        # If the sum of 'is_incident' in the next H steps is greater than 0, 
        # it means at least one minute in the near future is an incident.
        has_incident = int(future_window['is_incident'].sum() > 0)
        y.append(has_incident)
        
    # Convert lists to Pandas objects for ML compatibility
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series

if __name__ == "__main__":
    input_path = '../data/synthetic_metrics.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run generate_data.py first.")
        exit(1)
        
    df_raw = pd.read_csv(input_path)
    X, y = create_window_features(df_raw, W=30, H=10)
    
    # Save the processed dataset
    X.to_csv('../data/features.csv', index=False)
    y.to_csv('../data/target.csv', index=False)
    
    print(f"Extracted {len(X)} windows successfully.")
    print("Class imbalance check:")
    print(y.value_counts(normalize=True).round(3)) # Display ratio of 0s vs 1s