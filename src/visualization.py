import plotly.express as px
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_scatter(df):
    fig = px.scatter(
        df, 
        x='listening_time', 
        y='completion_rate',
        color=df['cluster'].astype(str),
        hover_data=['genre', 'skip_rate'],
        title="Listening Time vs Completion Rate",
        labels={'cluster': 'Cluster', 'listening_time': 'Listening Time (mins)', 'completion_rate': 'Completion %'},
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    return fig

def plot_cluster_sizes(profiles_df):
    fig = px.bar(
        profiles_df, 
        x='segment_name', 
        y='user_count',
        color='segment_name',
        title="Audience Size per Segment",
        text='user_count',
        template='plotly_white'
    )
    fig.update_traces(textposition='outside')
    return fig

def plot_silhouette_scores(scores_df, best_k):
    fig = px.line(
        scores_df, 
        x='k', 
        y='silhouette_score',
        markers=True,
        title="Silhouette Score vs K (Optimal Selection)",
        template='plotly_white'
    )
    fig.add_vline(x=best_k, line_dash="dash", line_color="red", annotation_text=f"Optimal K={best_k}")
    return fig

def create_dendrogram(linkage_matrix):
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage_matrix, truncate_mode='level', p=5, ax=ax)
    ax.set_title("Ward's Hierarchical Clustering Dendrogram (Truncated)")
    ax.set_xlabel("Data Points / Cluster Size")
    ax.set_ylabel("Euclidean Distance")
    return fig