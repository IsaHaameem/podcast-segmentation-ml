import streamlit as st
import pandas as pd
import warnings

# Suppress sklearn convergence/imputer warnings for clean UI
warnings.filterwarnings('ignore')

from src.preprocessing import validate_schema, preprocess_features
from src.clustering import apply_ward_clustering, compute_linkage_matrix
from src.evaluation import calculate_silhouette, find_optimal_k
from src.insights import generate_cluster_profiles
from src.visualization import plot_scatter, plot_cluster_sizes, plot_silhouette_scores, create_dendrogram

st.set_page_config(page_title="Podcast ML Segmentation", layout="wide", page_icon="🎙️")

# --- CACHED FUNCTIONS --- #
@st.cache_data(show_spinner=False)
def load_and_preprocess(file):
    df = pd.read_csv(file)
    validate_schema(df)
    X = preprocess_features(df)
    return df, X

@st.cache_data(show_spinner=False)
def get_evaluation(X):
    return find_optimal_k(X)

@st.cache_data(show_spinner=False)
def get_linkage(X):
    return compute_linkage_matrix(X)

# --- UI LAYOUT --- #
st.title("🎙️ Podcast Listener Segmentation Engine")
st.markdown("An end-to-end ML pipeline using **Ward's Hierarchical Clustering** to discover behavioral segments.")

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload Podcast CSV", type=['csv'])
    
    st.markdown("---")
    st.header("2. Model Configuration")
    manual_k = st.slider("Select K (Number of Clusters)", min_value=2, max_value=10, value=3)
    
    run_btn = st.button("Run ML Pipeline", type="primary", use_container_width=True)

if uploaded_file is not None:
    try:
        with st.spinner("Processing data..."):
            df, X = load_and_preprocess(uploaded_file)
            
        if run_btn:
            # Main Execution
            with st.spinner("Calculating optimal K and evaluating..."):
                best_k, scores_df = get_evaluation(X)
                
            with st.spinner(f"Applying Ward's Clustering (K={manual_k})..."):
                model, labels = apply_ward_clustering(X, manual_k)
                current_silhouette = calculate_silhouette(X, labels)
                
                # Insights
                profiles_df, df_results = generate_cluster_profiles(df, labels)
            
            # --- TABS UI --- #
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Business Insights", 
                "🔬 Evaluation & Optimal K", 
                "🌳 Dendrogram",
                "🗄️ Raw Data & Profiles"
            ])
            
            # TAB 1: INSIGHTS
            with tab1:
                st.subheader(f"Audience Segments (K={manual_k})")
                
                # Top-level metrics
                cols = st.columns(len(profiles_df))
                for i, row in profiles_df.iterrows():
                    cols[i].metric(label=row['segment_name'], value=f"{row['user_count']} Users")
                
                st.plotly_chart(plot_cluster_sizes(profiles_df), use_container_width=True)
                
                st.markdown("### Behavioral Mapping")
                st.plotly_chart(plot_scatter(df_results), use_container_width=True)
            
            # TAB 2: EVALUATION
            with tab2:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Current Silhouette Score", f"{current_silhouette:.3f}")
                    if manual_k != best_k:
                        st.warning(f"Note: Algorithmic optimal is **K={best_k}**. You selected **K={manual_k}**.")
                    else:
                        st.success(f"You have selected the optimal K ({best_k}) based on Silhouette Score.")
                
                with col2:
                    st.plotly_chart(plot_silhouette_scores(scores_df, best_k), use_container_width=True)

            # TAB 3: DENDROGRAM
            with tab3:
                st.markdown("### Hierarchical Linkage (Truncated for performance)")
                with st.spinner("Computing distance matrix..."):
                    Z = get_linkage(X)
                    st.pyplot(create_dendrogram(Z))
                    
            # TAB 4: DATA PREVIEW
            with tab4:
                st.markdown("### Segment Profiles (Centroids)")
                st.dataframe(
                    profiles_df.style.format({
                        'avg_listening_time': "{:.1f}m", 
                        'avg_completion_rate': "{:.1%}", 
                        'avg_skip_rate': "{:.1%}"
                    }),
                    use_container_width=True
                )
                
                st.markdown("### Processed Dataset")
                st.dataframe(df_results.head(100), use_container_width=True)

    except Exception as e:
        st.error(f"Pipeline Error: {str(e)}")
        st.info("Please ensure your CSV matches the required schema.")
else:
    st.info("👈 Please upload a dataset in the sidebar to begin. (Use the generator script to create `sample.csv`)")