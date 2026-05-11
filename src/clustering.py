from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_best_k(data, max_k=10):
    """Find best number of clusters using silhouette score"""
    scores = {}

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        scores[k] = score

    best_k = max(scores, key=scores.get)
    return best_k, scores


def run_kmeans(data, n_clusters):
    """Train KMeans model"""
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)

    return model, labels