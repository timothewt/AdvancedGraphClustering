from cdlib.algorithms import spectral

from algorithms.Algorithm import Algorithm
from .utils import extract_clusters_from_communities_list


class Spectral(Algorithm):
	"""Spectral clustering algorithm
	"""

	def run(self, num_clusters: int) -> None:
		"""Runs the algorithm

		:param num_clusters: Number of clusters to form
		:type num_clusters: int
		"""
		clustering = spectral(self.graph.nx_graph, kmax=num_clusters)
		self.clusters = extract_clusters_from_communities_list(clustering.communities)

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "Spectral algorithm object"
