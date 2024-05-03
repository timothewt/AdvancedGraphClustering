import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import normalized_mutual_info_score, silhouette_score, adjusted_rand_score


def plot_training_history(labels: list[str], values: np.ndarray[tuple[float]], title: str) -> None:
	"""
	Plot the training history (loss, metrics)
	Parameters
	----------
	labels: list[str]
		The labels for the plot
	values: np.ndarray[tuple[float]]
		The values to plot
	title: str
		The title of the plot
	"""
	fig, axs = plt.subplots(1, len(labels), figsize=(3 * len(labels), 3))
	plt.title(title)
	for idx in range(len(labels)):
		ax = axs[idx]
		ax.set_title(f'{labels[idx]}')
		ax.set_xlabel('Batch')
		ax.set_ylabel(f'{labels[idx]}')
		ax.plot(values[:, idx], label=f'{labels[idx]}', color=f'C{idx}')
		ax.legend(loc="upper left")

	plt.tight_layout()
	plt.show()


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


def plot_3d_clustering_comparison(z: np.ndarray, true_labels: np.ndarray, cluster_labels: np.ndarray, n_clusters: int = None) -> None:
	"""
	Plot the clustering results in 3D
	Parameters
	----------
	z: np.ndarray
		The data to cluster
	true_labels: np.ndarray
		The true labels
	cluster_labels: np.ndarray
		The cluster labels
	"""
	def plot_3d(z_embedded: np.ndarray, labels: np.ndarray, title: str, fig: plt.Figure, subplot_idx: int, subplots_nb: int) -> None:
		ax = fig.add_subplot(100 + 10 * subplots_nb + subplot_idx, projection='3d')
		ax.set_title(title)
		for i in range(n_clusters):
			ax.scatter(z_embedded[labels == i, 0], z_embedded[labels == i, 1], z_embedded[labels == i, 2], label=f"Cluster {i}")
		ax.legend()

	if n_clusters is None:
		n_clusters = len(np.unique(true_labels))
	z_embedded = TSNE(n_components=3).fit_transform(z)

	fig = plt.figure(figsize=(15, 5))

	plot_3d(z_embedded, true_labels, "True labels", fig, 1, 3)
	plot_3d(z_embedded, cluster_labels, "Clusters labels", fig, 2, 3)

	plt.tight_layout()
	plt.show()

