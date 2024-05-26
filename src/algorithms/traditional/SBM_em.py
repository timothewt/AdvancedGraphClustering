from graph import Graph
from algorithms.Algorithm import Algorithm
from .utils import extract_clusters_from_communities_list

import sys
sys.path.append('../library/')
import pysbm

class SBM_em(Algorithm):
    """Stochastic block model clustering algorithm

    // Installer graph_tools pour l'utiliser
    """
    def __init__(self, graph: Graph, num_clusters: int, iterations: int = 10000):
        """Constructor method
        """
        super(SBM_em, self).__init__(graph, num_clusters, iterations)
        self.iterations: int = iterations
        self.num_clusters: int = num_clusters
        self.graph = graph

    def run(self) -> None:
        """Runs the algorithm
        """
        standard_partition = pysbm.NxPartition(
            graph=self.graph.nx_graph,
            number_of_blocks=self.num_clusters
        )

        standard_objective_function = pysbm.TraditionalUnnormalizedLogLikelyhood(is_directed=False)
        standard_inference = EMInference(self.graph.nx_graph, standard_objective_function, standard_partition)
        standard_inference.infer_stochastic_block_model()  # Note: No iterations argument here

        partition_dict = standard_partition.partition

        clusters = {}
        for node, cluster_id in partition_dict.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(node)

        cluster_list = list(clusters.values())

        self.clusters = extract_clusters_from_communities_list(cluster_list)

    def __str__(self):
        """Returns the string representation of the algorithm object

        :return: String representation of the algorithm object
        :rtype: str
        """
        return "SBM algorithm object"


class EMInference(pysbm.Inference):
    """Expectation-Maximization Algorithm for SBM inference"""

    def __init__(self, graph, objective_function, partition, with_toggle_detection=True, limit_possible_blocks=False):
        super(EMInference, self).__init__(graph, objective_function, partition)
        self.with_toggle_detection = with_toggle_detection
        self._old_value = self._objective_function.calculate(partition)
        self.limit_possible_blocks = limit_possible_blocks

    def infer_stochastic_block_model(self):
        if self.partition.is_graph_directed():
            try:
                for _ in range(2 * len(self.graph)):
                    self.infer_stepwise_directed()
            except Exception as e:
                print("EMInference: could not find an optimal partition in", 2 * len(self.graph), "steps", self.partition.get_representation(), self.graph.edges())

