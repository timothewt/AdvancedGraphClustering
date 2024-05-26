import sys
from graph import Graph
from algorithms.Algorithm import Algorithm

sys.path.append('..\\library\\')
import pysbm


class SBM_metropolis(Algorithm):
	"""Stochastic block model clustering algorithm
	"""

	def __init__(self, graph: Graph, num_clusters: int, iterations: int = 10000):
		"""Constructor method
		"""
		super(SBM_metropolis, self).__init__(graph)
		self.iterations: int = iterations
		self.num_clusters: int = num_clusters
		self.graph = graph

	def run(self) -> None:
		"""Runs the algorithm
		"""
		standard_partition = pysbm.NxPartition(graph=self.graph.nx_graph, number_of_blocks=self.num_clusters)
		standard_objective_function = pysbm.TraditionalUnnormalizedLogLikelyhood(is_directed=False)
		standard_inference = pysbm.MetropolisHastingInference(self.graph.nx_graph, standard_objective_function, standard_partition)
		standard_inference.infer_stochastic_block_model(self.iterations)

		self.clusters = [node[1] for node in sorted(standard_inference.partition.partition.items())]

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "SBM algorithm object"
