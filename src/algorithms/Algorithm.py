import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from graph import Graph


class Algorithm:
	"""Base class for all algorithms

	:param graph: Graph object
	:type graph: Graph
	"""

	def __init__(self, graph: Graph, *args, **kwargs):
		"""Constructor method
		"""
		self.graph: Graph = graph
		self.clusters: list[int] = [0 for _ in range(self.graph.adj_matrix.shape[0])]

	def run(self) -> None:
		"""Runs the algorithm
		"""
		raise NotImplementedError

	def evaluate(self) -> list[(str, float)]:
		"""Evaluates the clustering through various supervised (if labels provided in the Graph object) and unsupervised metrics:
		Supervised:
		- Accuracy
		- Normalized Mutual Information
		- Adjusted Rand Index
		Unsupervised:
		- Silhouette
		- Normalized Cut
		- ...
		"""
		metrics: list[(str, float)] = []
		if self.graph.labels is not None:
			accuracy: float = self._get_accuracy()
			nmi: float = self._get_nmi()
			ari: float = self._get_ari()
			metrics.append(("ACC", accuracy))
			metrics.append(("NMI", nmi))
			metrics.append(("ARI", ari))

		# TODO: unsupervised (silouhette, NCut, ...)
		return metrics

	def get_clusters(self) -> list[int]:
		"""Returns the clusters

		:return: Clusters
		:rtype: list[int]
		"""
		return self.clusters

	def _get_accuracy(self) -> float:
		"""Returns the accuracy of the clustering

		:return: Accuracy of the clustering
		:rtype: float
		"""
		if self.graph.labels is None:
			raise ValueError("No labels provided for the graph")
		label_mapping = {}
		for cluster in np.unique(self.clusters):
			true_label = np.argmax(np.bincount(self.graph.labels[np.array(self.clusters) == cluster]))
			label_mapping[cluster] = true_label

		return np.mean(np.array(self.graph.labels) == np.array([label_mapping[cluster] for cluster in self.clusters]))

	def _get_nmi(self) -> float:
		"""Returns the Normalized Mutual Information of the clustering

		:return: Normalized Mutual Information of the clustering
		:rtype: float
		"""
		if self.graph.labels is None:
			raise ValueError("No labels provided for the graph")
		return normalized_mutual_info_score(self.graph.labels, self.clusters)

	def _get_ari(self) -> float:
		"""Returns the Adjusted Rand Index of the clustering

		:return: Adjusted Rand Index of the clustering
		:rtype: float
		"""
		if self.graph.labels is None:
			raise ValueError("No labels provided for the graph")
		return adjusted_rand_score(self.graph.labels, self.clusters)

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "Algorithm object"
