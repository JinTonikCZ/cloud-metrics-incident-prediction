import pandas as pd
import numpy as np
import os

def generate_metrics_data(n_minutes: int = 15000) -> pd.DataFrame:
    """
    Simulates a synthetic cloud monitoring dataset.
    
    The simulation includes:
    - Normal noisy operations (Gaussian noise)
    - Gradual degradation (Trend injection)
    - Critical incidents (Spikes and sustained high load)
    
    Args:
        n_minutes (int): Total duration of the generated data in minutes.
        
    Returns:
        pd.DataFrame: A dataset with CPU, RAM, Latency, and Error Rate.
    """
    print(f"--- Data Generation: Simulating {n_minutes} minutes of infrastructure metrics ---")
    
    np.random.seed(42)  # For reproducibility
    t = np.arange(n_minutes)
    
    # 1. CPU Usage (%) - Modeled as a baseline with daily seasonality and noise
    cpu = 30 + 5 * np.sin(2 * np.pi * t / 1440) + np.random.normal(0, 3, n_minutes)
    
    # 2. Memory Usage (%) - Often stays stable with occasional leaks
    memory = 40 + np.cumsum(np.random.normal(0, 0.01, n_minutes))
    
    # 3. Latency (ms) - Lognormal distribution is more realistic for response times
    latency = np.random.lognormal(mean=2, sigma=0.3, size=n_minutes)
    
    # 4. Error Rate (%) - Low baseline with sudden spikes
    error_rate = np.random.exponential(scale=0.5, size=n_minutes)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2026-01-01', periods=n_minutes, freq='min'),
        'cpu_usage': np.clip(cpu, 0, 100),
        'memory_usage': np.clip(memory, 0, 100),
        'latency': latency,
        'error_rate': error_rate,
        'is_incident': 0
    })

    # 5. Incident Injection
    # We manually inject 15 incidents to ensure the model has something to learn
    for _ in range(15):
        start = np.random.randint(100, n_minutes - 200)
        duration = np.random.randint(20, 60)
        
        # Degradation phase: Metrics start rising before the actual failure
        df.loc[start-15:start, 'cpu_usage'] += np.linspace(0, 40, 16)
        df.loc[start-15:start, 'latency'] *= 2
        
        # Active incident phase: High metrics and error rate
        df.loc[start:start+duration, 'is_incident'] = 1
        df.loc[start:start+duration, 'cpu_usage'] = np.random.uniform(90, 100, duration+1)
        df.loc[start:start+duration, 'error_rate'] += np.random.uniform(5, 15, duration+1)
        
    return df

if __name__ == "__main__":
    # Path handling compatible with Docker and local environments
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(BASE_DIR, 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = generate_metrics_data()
    dataset.to_csv(os.path.join(output_dir, 'synthetic_metrics.csv'), index=False)
    print(f"Dataset saved to {output_dir}/synthetic_metrics.csv")