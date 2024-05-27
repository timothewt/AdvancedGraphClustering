import sys
from graph import Graph
from algorithms.Algorithm import Algorithm

sys.path.append('..\\library\\')
import pysbm


class DCSBM(Algorithm):
	"""Stochastic block model clustering algorithm
	"""

	def __init__(self, graph: Graph, num_clusters: int, iterations: int = 10000):
		"""Constructor method
		"""
		super(DCSBM, self).__init__(graph)
		self.iterations: int = iterations
		self.num_clusters: int = num_clusters
		self.graph = graph

	def run(self) -> None:
		"""Runs the algorithm
		"""
		degree_corrected_partition = pysbm.NxPartition(
            graph=self.graph.nx_graph, 
            number_of_blocks=self.num_clusters)
		degree_corrected_objective_function = pysbm.DegreeCorrectedUnnormalizedLogLikelyhood(is_directed=False)
		degree_corrected_inference = pysbm.PeixotoInference(self.graph.nx_graph, degree_corrected_objective_function, degree_corrected_partition)
		degree_corrected_inference.infer_stochastic_block_model()

		self.clusters = [node[1] for node in sorted(degree_corrected_inference.partition.partition.items())]

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "SBM Metropolis algorithm object"