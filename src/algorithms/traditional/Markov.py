import markov_clustering as mc

from algorithms.Algorithm import Algorithm
from graph import Graph
from .utils import extract_clusters_from_communities_list


class Markov(Algorithm):
	"""Markov clustering algorithm

	:param graph: Graph object
	:type graph: Graph
	:param expansion: Cluster expansion factor
	:type expansion: int
	:param inflation: Cluster inflation factor
	:type inflation: int
	:param iterations: Maximum number of iterations
	:type iterations: int
	"""
	def __init__(self, graph: Graph, expansion: int = 2, inflation: int = 2, iterations: int = 100) -> None:
		"""Constructor method
		"""
		super().__init__(graph)
		self.expansion = expansion
		self.inflation = inflation
		self.iterations = iterations

	def run(self) -> None:
		"""Runs the algorithm with the given parameters
		"""
		clustering = mc.run_mcl(
			self.graph.adj_matrix,
			expansion=self.expansion,
			inflation=self.inflation,
			iterations=self.iterations,
		)
		self.clusters = extract_clusters_from_communities_list(mc.get_clusters(clustering))

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "Markov algorithm object"