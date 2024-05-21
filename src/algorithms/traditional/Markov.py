import markov_clustering as mc

from algorithms.Algorithm import Algorithm
from .utils import extract_clusters_from_communities_list


class Markov(Algorithm):
	"""Markov clustering algorithm
	"""

	def run(self, expansion: int = 2, inflation: int = 2, iterations: int = 100, pruning_threshold: int = .001) -> None:
		"""Runs the algorithm with the given parameters

		:param expansion: Cluster expansion factor
		:type expansion: int
		:param inflation: Cluster inflation factor
		:type inflation: int
		:param iterations: Maximum number of iterations
		:type iterations: int
		:param pruning_threshold: Threshold below which matrix elements will be set to 0
		:type pruning_threshold: float
		"""
		clustering = mc.run_mcl(
			self.graph.adj_matrix,
			expansion=expansion,
			inflation=inflation,
			iterations=iterations,
			pruning_threshold=pruning_threshold
		)
		self.clusters = extract_clusters_from_communities_list(mc.get_clusters(clustering))

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "Markov algorithm object"