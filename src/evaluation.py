from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pandas as pd

def calculate_silhouette(X, labels):
    if len(set(labels)) > 1:
        return silhouette_score(X, labels)
    return 0.0

def find_optimal_k(X, max_k=10):
    scores = []
    best_k = 2
    best_score = -1
    
    # Cap max_k to sample size if very small data
    max_k = min(max_k, len(X) - 1)
    
    for k in range(2, max_k + 1):
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        
        scores.append({'k': k, 'silhouette_score': score})
        
        if score > best_score:
            best_score = score
            best_k = k
            
    return best_k, pd.DataFrame(scores)