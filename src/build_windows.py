import pandas as pd
import numpy as np
import os

def extract_features(vals: np.ndarray) -> dict:
    """
    Computes statistical aggregates for a single metric window.
    
    This helper function abstracts the feature engineering logic, 
    making it easier to unit test and maintain.
    
    Args:
        vals (np.ndarray): 1D array of metric values (e.g., CPU for 30 mins).
        
    Returns:
        dict: Set of calculated statistics (mean, std, min, max, last, diff).
    """
    stats = {
        'mean': np.mean(vals),
        'std': np.std(vals),
        'min': np.min(vals),
        'max': np.max(vals),
        'last': vals[-1],
        'diff': vals[-1] - vals[0]  # Trend: positive if rising, negative if falling
    }
    return stats

def create_window_features(df: pd.DataFrame, W: int = 30, H: int = 10) -> tuple:
    """
    Transforms raw time-series metrics into a supervised Machine Learning dataset.
    
    Architecture:
    - Inputs: Look-back window 'W' (History)
    - Target: Horizon 'H' (Predictive goal). 1 if incident occurs within H steps.
    
    Args:
        df (pd.DataFrame): Raw metrics with 'is_incident' column.
        W (int): History window size in minutes.
        H (int): Prediction horizon in minutes.
        
    Returns:
        tuple: (X_df, y_series) ready for model training.
    """
    print(f"--- Feature Engineering: W={W}, H={H} ---")
    
    metrics = ['cpu_usage', 'memory_usage', 'latency', 'error_rate']
    X = []
    y = []
    
    # Iterate through the timeline ensuring we have enough past (W) and future (H)
    for i in range(W, len(df) - H):
        
        # 1. SLIDING WINDOW FEATURE EXTRACTION
        window = df.iloc[i-W:i]
        features = {}
        
        for col in metrics:
            vals = window[col].values
            # Get statistics for each metric
            col_stats = extract_features(vals)
            
            # Map statistics to column names (e.g., cpu_usage_mean)
            for stat_name, stat_val in col_stats.items():
                features[f'{col}_{stat_name}'] = stat_val
        
        X.append(features)
        
        # 2. TARGET FORMULATION
        # Look at the next H steps. If any step is an incident, the whole window is labeled '1'.
        future_window = df.iloc[i:i+H]
        has_incident = int(future_window['is_incident'].sum() > 0)
        y.append(has_incident)
        
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y, name='target')
    
    print(f"Created {len(X_df)} samples. Input features: {X_df.shape[1]}")
    return X_df, y_series

if __name__ == "__main__":
    # Ensure relative paths work regardless of where the script is called from
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    input_file = os.path.join(DATA_DIR, 'synthetic_metrics.csv')
    
    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: {input_file} not found. Generate data first.")
        exit(1)
        
    df_raw = pd.read_csv(input_file)
    X, y = create_window_features(df_raw)
    
    # Save processed datasets for training
    X.to_csv(os.path.join(DATA_DIR, 'features.csv'), index=False)
    y.to_csv(os.path.join(DATA_DIR, 'target.csv'), index=False)
    
    print("Dataset saved. Checking for class imbalance (Target = 1 is the incident class):")
    print(y.value_counts(normalize=True).round(4))