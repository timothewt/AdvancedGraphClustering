import numpy as np
from sklearn.cluster import KMeans


def get_clusters(z: np.ndarray, n_clusters: int, method: str = "kmeans") -> np.ndarray:
	"""
	Cluster the encoded data using the specified method

	:param z: Latent space
	:type z: np.ndarray
	:param n_clusters: Number of clusters
	:type n_clusters: int
	:param method: Clustering method
	:type method: str
	:returns: Cluster labels
	:rtype: np.ndarray
	"""
	if method == "kmeans":
		clusters: KMeans = KMeans(n_clusters=n_clusters).fit(z)
	else:
		raise ValueError(f"Unknown clustering method: {method}")
	return clusters.labels_
