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


def compute_diffusion_matrix(adj: np.ndarray, alpha: float = 0.2) -> np.ndarray:
	"""Computes the diffusion matrix for MVGRL using PageRank

	:param adj: Adjacency matrix
	:type adj: np.ndarray
	:param alpha: Teleport probability
	:type alpha: float
	:return: Diffusion matrix
	:rtype: np.ndarray
	"""
	adj = adj + np.eye(adj.shape[0])
	deg_sqrt_inv = np.diag(1.0 / np.sqrt(np.sum(adj, axis=1)))
	return np.dot(np.linalg.inv(np.eye(adj.shape[0]) - alpha * np.dot(deg_sqrt_inv, adj).T), (1 - alpha) * deg_sqrt_inv)
