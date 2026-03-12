import pandas as pd
import numpy as np
import os

# Fix the random seed to ensure that every time we run this script,
# we get the exact same dataset. This is crucial for reproducibility.
np.random.seed(42)

def generate_synthetic_data(num_steps=10000, num_incidents=50):
    """
    Generates a synthetic time-series dataset simulating cloud infrastructure metrics.
    
    The generation follows a specific lifecycle for anomalies:
    1. Normal operation (baseline + noise)
    2. Gradual degradation (metrics slowly worsen)
    3. Active incident (metrics breach thresholds, errors spike)
    4. Recovery (metrics return to baseline)
    
    Args:
        num_steps (int): Total number of minutes to simulate.
        num_incidents (int): Number of incidents to inject into the timeline.
        
    Returns:
        pd.DataFrame: DataFrame containing timestamps, metrics, and incident labels.
    """
    print(f"Generating {num_steps} minutes of synthetic cloud monitoring data...")
    
    # ==========================================
    # STEP 1: Generate Base Background Noise
    # ==========================================
    # We use different statistical distributions to make metrics look realistic:
    # - Normal distribution for CPU and Memory (fluctuating around a mean)
    # - Lognormal distribution for Latency (mostly low, with occasional long tails/spikes)
    # - Exponential distribution for Error Rate (mostly near zero)
    
    cpu_usage = np.random.normal(loc=30, scale=5, size=num_steps)        # Mean 30%, Std 5%
    memory_usage = np.random.normal(loc=50, scale=2, size=num_steps)     # Mean 50%, Std 2%
    latency = np.random.lognormal(mean=3, sigma=0.5, size=num_steps)     # Log scale for ms
    error_rate = np.random.exponential(scale=0.01, size=num_steps)       # Mostly 0.0 - 0.05
    
    # Array to store our ground truth labels (1 = incident occurring right now, 0 = normal)
    is_incident = np.zeros(num_steps, dtype=int)
    
    # ==========================================
    # STEP 2: Inject Incidents
    # ==========================================
    # Select random timestamps where incidents will start.
    # We leave margins (100 steps) at the beginning and end to avoid index out-of-bounds errors.
    incident_starts = np.random.choice(range(100, num_steps - 100), size=num_incidents, replace=False)
    incident_starts.sort()
    
    for start in incident_starts:
        # Define random durations for each phase of the incident lifecycle
        degradation_len = np.random.randint(15, 40) # Phase 1: Worsening over 15-40 mins
        incident_len = np.random.randint(10, 30)    # Phase 2: Active fire for 10-30 mins
        recovery_len = np.random.randint(10, 20)    # Phase 3: Fixing it takes 10-20 mins
        
        # Calculate array indices for these phases
        deg_idx = range(start - degradation_len, start)
        inc_idx = range(start, start + incident_len)
        rec_idx = range(start + incident_len, start + incident_len + recovery_len)
        
        # --- Phase 1: Degradation ---
        # CPU and Latency slowly increase linearly before the actual incident hits.
        cpu_usage[deg_idx] += np.linspace(0, 40, degradation_len)
        latency[deg_idx] += np.linspace(0, 200, degradation_len)
        
        # --- Phase 2: Active Incident ---
        # Metrics go critical. We label these exact minutes as '1' in our target variable.
        is_incident[inc_idx] = 1
        cpu_usage[inc_idx] = np.random.normal(loc=95, scale=3, size=incident_len)     # CPU pegs near 100%
        memory_usage[inc_idx] += np.random.normal(loc=30, scale=5, size=incident_len) # Sudden memory leak
        latency[inc_idx] = np.random.lognormal(mean=6, sigma=0.8, size=incident_len)  # Latency skyrockets
        error_rate[inc_idx] = np.random.uniform(0.1, 0.5, size=incident_len)          # 10% to 50% of requests fail
        
        # --- Phase 3: Recovery ---
        # Metrics smoothly transition back down to their baseline values.
        cpu_usage[rec_idx] = np.linspace(cpu_usage[rec_idx[0]-1], 30, recovery_len)
        latency[rec_idx] = np.linspace(latency[rec_idx[0]-1], 20, recovery_len)

    # ==========================================
    # STEP 3: Data Cleaning and Formatting
    # ==========================================
    # Physical limits: percentages cannot go below 0 or above 100.
    cpu_usage = np.clip(cpu_usage, 0, 100)
    memory_usage = np.clip(memory_usage, 0, 100)
    error_rate = np.clip(error_rate, 0, 1)

    # Construct the final dataset with timestamps starting from a fixed date
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=num_steps, freq='min'),
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'latency': latency,
        'error_rate': error_rate,
        'is_incident': is_incident
    })
    
    return df

if __name__ == "__main__":
    # Ensure the data directory exists
    os.makedirs('../data', exist_ok=True)
    
    # Generate 15,000 minutes (~10.4 days) of data with 30 simulated outages
    df_metrics = generate_synthetic_data(num_steps=15000, num_incidents=30)
    
    # Save raw data to CSV
    output_path = '../data/synthetic_metrics.csv'
    df_metrics.to_csv(output_path, index=False)
    
    print(f"Data generation complete! Saved to {output_path}")