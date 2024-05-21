from graph import Graph


class Algorithm:
	"""Base class for all algorithms
	"""

	def __init__(self, graph: Graph):
		"""Constructor method
		"""
		self.graph: Graph = graph
		self.clusters: list[int] = [0 for _ in range(self.graph.adj_matrix.shape[0])]

	def run(self, num_clusters: int, *args, **kwargs):
		"""Runs the algorithm
		"""
		raise NotImplementedError

	def get_clusters(self) -> list[int]:
		"""Returns the clusters

		:return: Clusters
		:rtype: list[int]
		"""
		return self.clusters

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "Algorithm object"
