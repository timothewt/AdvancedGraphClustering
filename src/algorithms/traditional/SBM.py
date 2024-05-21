from cdlib.algorithms import sbm_dl

from algorithms.Algorithm import Algorithm
from .utils import extract_clusters_from_communities_list


class SBM(Algorithm):
	"""Stochastic block model clustering algorithm

	// Installer graph_tools pour l'utiliser
	"""

	def run(self) -> None:
		"""Runs the algorithm
		"""
		clustering = sbm_dl(self.graph.nx_graph)
		self.clusters = extract_clusters_from_communities_list(clustering.communities)

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "SBM algorithm object"
