import numpy as np
import networkx as nx
from collections import defaultdict
from networkx.algorithms.community import modularity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from graph import Graph


class Algorithm:
	"""Base class for graph clustering algorithms

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

		* Accuracy

		* Normalized Mutual Information

		* Adjusted Rand Index

		Unsupervised:

		* Conductance

		* Silhouette

		* Modularity
		"""
		metrics: list[(str, float)] = []
		if self.graph.labels is not None:
			accuracy: float = self._get_accuracy()
			nmi: float = self._get_nmi()
			ari: float = self._get_ari()
			metrics.append(("ACC", accuracy))
			metrics.append(("NMI", nmi))
			metrics.append(("ARI", ari))

		conductance: float = self._get_conductance()
		silouhette: float = self. _get_modularity()
		internal_density: float = self._get_internal_density()
		metrics.append(("Conductance", conductance))
		metrics.append(("Modularity", silouhette))
		metrics.append(("Internal density", internal_density))

		return metrics

	def get_clusters(self) -> list[int]:
		"""Returns the clusters

		:return: Clusters
		:rtype: list[int]
		"""
		return self.clusters

	def get_communities(self) -> list[list[int]]:
		""" Returns the clusters as communities (list of nodes list)

		:return: Clusters as communities
		:rtype: list[list[int]]
		"""
		communities = [[] for _ in range(max(self.clusters) + 1)]
		for node, cluster in enumerate(self.clusters):
			communities[cluster].append(node)
		return communities

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

	def _get_conductance(self) -> float:
		"""Returns the average conductance of the clustering

		:return: Average conductance of the clustering
		:rtype: float
		"""
		conductances = []
		for cluster in self.get_communities():
			cut_size = nx.cut_size(self.graph.nx_graph, cluster)
			volume = sum(self.graph.nx_graph.degree(node) for node in cluster)
			conductance = cut_size / volume if volume != 0 else 0
			conductances.append(conductance)

		return np.mean(conductances)

	def _get_modularity(self) -> float:
		"""Returns the Modularity of the clustering

		:return: Modularity of the clustering
		:rtype: float
		"""
		return modularity(self.graph.nx_graph, self.get_communities())
	
	def _get_internal_density(self) -> float:
		"""Calculates the average internal density.

		:return: The average internal density of each cluster.
		:rtype: float
		"""
		densities = []
		for cluster in self.get_communities():
			subgraph = self.graph.nx_graph.subgraph(cluster)
			density = nx.density(subgraph)
			densities.append(density)

		return np.mean(densities)

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
