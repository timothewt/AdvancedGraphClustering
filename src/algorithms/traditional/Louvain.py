from cdlib.algorithms import louvain

from algorithms.Algorithm import Algorithm
from .utils import extract_clusters_from_communities_list


class Louvain(Algorithm):
	"""Louvain clustering algorithm
	"""

	def run(self) -> None:
		"""Runs the algorithm
		"""
		clustering = louvain(self.graph.nx_graph)
		self.clusters = extract_clusters_from_communities_list(clustering.communities)

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "Louvain algorithm object"
