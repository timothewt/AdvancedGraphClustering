import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, silhouette_score, adjusted_rand_score


def get_clusters(z: np.ndarray, n_clusters: int, method: str = "kmeans") -> np.ndarray:
	"""
	Cluster the data using the specified method
	Parameters
	----------
	z: np.ndarray
		The data to cluster
	n_clusters: int
		The number of clusters to create
	method: str
		The clustering method to use (default: KMeans)

	Returns
	-------
	cluster_labels: np.ndarray
		The cluster labels
	"""
	if method == "kmeans":
		clusters: KMeans = KMeans(n_clusters=n_clusters).fit(z)
	else:
		raise ValueError(f"Unknown clustering method: {method}")
	return clusters.labels_


def evaluate_clustering(z: np.ndarray, true_labels: np.ndarray, cluster_labels: np.ndarray) -> tuple[
	float, float, float, float]:
	"""
	Evaluate the clustering results
	Parameters
	----------
	z: np.ndarray
		The data to cluster
	true_labels: np.ndarray
		The true labels
	cluster_labels: np.ndarray
		The cluster labels

	Returns
	-------
	accuracy, nmi, ari, silouhette: tuple[float]
		The evaluation metrics
	"""
	label_mapping = {}
	for cluster in np.unique(cluster_labels):
		true_label = np.argmax(np.bincount(true_labels[cluster_labels == cluster]))
		label_mapping[cluster] = true_label

	accuracy = np.mean(np.array(true_labels) == np.array([label_mapping[cluster] for cluster in cluster_labels]))
	nmi = normalized_mutual_info_score(true_labels, cluster_labels)
	ari = adjusted_rand_score(true_labels, cluster_labels)
	silouhette = silhouette_score(z, cluster_labels)
	return accuracy, nmi, ari, silouhette
