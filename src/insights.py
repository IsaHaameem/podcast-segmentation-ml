import pandas as pd

def generate_cluster_profiles(df, labels):
    df_analysis = df.copy()
    df_analysis['cluster'] = labels
    
    profiles = df_analysis.groupby('cluster').agg(
        user_count=('user_id', 'count'),
        avg_listening_time=('listening_time', 'mean'),
        avg_completion_rate=('completion_rate', 'mean'),
        avg_skip_rate=('skip_rate', 'mean'),
        top_genre=('genre', lambda x: x.mode()[0]),
        primary_frequency=('frequency', lambda x: x.mode()[0])
    ).reset_index()
    
    return assign_business_labels(profiles), df_analysis

def assign_business_labels(profiles_df):
    segment_names = []
    
    for _, row in profiles_df.iterrows():
        # Business Logic applied to Centroids
        if row['avg_listening_time'] > 100 and row['avg_completion_rate'] > 0.75:
            segment_names.append("Binge Listeners 🎧")
        elif row['avg_skip_rate'] > 0.45 and row['avg_completion_rate'] < 0.5:
            segment_names.append("Impatient / Low Engagement ⏭️")
        elif 40 <= row['avg_listening_time'] <= 100:
            segment_names.append("Casual Users 🚶")
        else:
            segment_names.append("Mixed Segment 🧩")
            
    profiles_df['segment_name'] = segment_names
    return profiles_df