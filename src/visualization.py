import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_pca(data, labels):
    """PCA 2D visualization"""
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)

    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels)
    plt.title("PCA Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig("visualizations/pca_plot.png")
    plt.show()


def plot_tsne(data, labels):
    """t-SNE visualization"""
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(data)

    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels)
    plt.title("t-SNE Clusters")
    plt.savefig("visualizations/tsne_plot.png")
    plt.show()


def plot_cluster_distribution(labels):
    """Show cluster sizes"""
    plt.figure()
    plt.hist(labels, bins=len(set(labels)))
    plt.title("Cluster Distribution")
    plt.savefig("visualizations/cluster_distribution.png")
    plt.show()