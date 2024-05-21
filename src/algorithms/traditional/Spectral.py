from cdlib.algorithms import spectral

from algorithms.Algorithm import Algorithm
from graph import Graph
from .utils import extract_clusters_from_communities_list


class Spectral(Algorithm):
	"""Spectral clustering algorithm

	:param graph: Graph object
	:type graph: Graph
	:param num_clusters: Number of clusters to form
	:type num_clusters: int
	"""

	def __init__(self, graph: Graph, num_clusters: int = 3) -> None:
		"""Constructor method
		"""
		super().__init__(graph)
		self.num_clusters = num_clusters

	def run(self) -> None:
		"""Runs the algorithm

		:param num_clusters: Number of clusters to form
		:type num_clusters: int
		"""
		clustering = spectral(self.graph.nx_graph, kmax=self.num_clusters)
		self.clusters = extract_clusters_from_communities_list(clustering.communities)

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "Spectral algorithm object"
