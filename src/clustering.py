from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage

def apply_ward_clustering(X, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(X)
    return model, labels

def compute_linkage_matrix(X):
    # Generates linkage matrix for the dendrogram
    return linkage(X, method='ward')