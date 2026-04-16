import pandas as pd
import numpy as np
import os

def generate_podcast_data(num_samples=2500):
    np.random.seed(42)
    
    # 1. Binge Listeners (High time, high completion, low skip)
    n_binge = int(num_samples * 0.4)
    binge = pd.DataFrame({
        'listening_time': np.random.normal(150, 30, n_binge),
        'episode_length_pref': np.random.choice(['long', 'medium'], n_binge, p=[0.8, 0.2]),
        'genre': np.random.choice(['True Crime', 'History', 'Fiction'], n_binge),
        'skip_rate': np.random.normal(0.1, 0.05, n_binge),
        'completion_rate': np.random.normal(0.9, 0.08, n_binge),
        'frequency': ['daily'] * n_binge
    })
    
    # 2. Casual Users (Moderate time, mid completion, mid skip)
    n_casual = int(num_samples * 0.4)
    casual = pd.DataFrame({
        'listening_time': np.random.normal(60, 20, n_casual),
        'episode_length_pref': np.random.choice(['medium', 'short'], n_casual, p=[0.7, 0.3]),
        'genre': np.random.choice(['Comedy', 'News', 'Tech'], n_casual),
        'skip_rate': np.random.normal(0.3, 0.1, n_casual),
        'completion_rate': np.random.normal(0.6, 0.15, n_casual),
        'frequency': ['weekly'] * n_casual
    })
    
    # 3. Impatient/Low Engagement (Low time, low completion, high skip)
    n_impatient = num_samples - n_binge - n_casual
    impatient = pd.DataFrame({
        'listening_time': np.random.normal(20, 10, n_impatient),
        'episode_length_pref': ['short'] * n_impatient,
        'genre': np.random.choice(['News', 'Self-Help'], n_impatient),
        'skip_rate': np.random.normal(0.7, 0.15, n_impatient),
        'completion_rate': np.random.normal(0.2, 0.1, n_impatient),
        'frequency': np.random.choice(['weekly', 'monthly'], n_impatient)
    })
    
    df = pd.concat([binge, casual, impatient], ignore_index=True)
    df['user_id'] = [f'USR_{i:05d}' for i in range(1, num_samples + 1)]
    
    # Clip limits
    df['listening_time'] = df['listening_time'].clip(lower=1)
    df['skip_rate'] = df['skip_rate'].clip(0, 1)
    df['completion_rate'] = df['completion_rate'].clip(0, 1)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample.csv', index=False)
    print(f"Generated {num_samples} rows at data/sample.csv")

if __name__ == "__main__":
    generate_podcast_data()