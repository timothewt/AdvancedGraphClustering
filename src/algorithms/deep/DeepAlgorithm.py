import torch

from algorithms.Algorithm import Algorithm
from algorithms.deep.utils import get_clusters
from graph import Graph


class DeepAlgorithm(Algorithm):
	"""Base class for Deep Graph Clustering algorithms

	:param graph: Graph object
	:type graph: Graph
	:param lr: Learning rate
	:type lr: float
	:param latent_dim: Latent dimension
	:type latent_dim: int
	:param dropout: Dropout rate
	:type dropout: int
	:param epochs: Number of epochs to run
	:type epochs: int
	"""

	def __init__(self, graph: Graph, num_clusters: int, lr: float = .001, latent_dim: int = 16, dropout: int = .0, epochs: int = 100):
		"""Constructor method
		"""
		super(DeepAlgorithm, self).__init__(graph)
		self.num_clusters: int = num_clusters
		self.lr: float = lr
		self.latent_dim: int = latent_dim
		self.dropout: int = dropout
		self.epochs: int = epochs

		self.model: torch.nn.Module = torch.nn.Module()

	def _train(self) -> None:
		"""Trains the model, to be implemented by subclasses
		"""
		raise NotImplementedError

	def run(self) -> None:
		"""Trains the model and runs k-means clustering on the node embeddings.
		"""
		self._train()
		self.model.eval()
		z_np = self.model.encode(torch.tensor(self.graph.features, dtype=torch.float), torch.tensor(self.graph.edge_index, dtype=torch.long)).detach().numpy()
		clusters = [
			get_clusters(z_np, self.num_clusters) for _ in range(10)
		]  # Run clustering 10 times and get the most common clustering
		best_clustering = None
		best_acc = 0
		for clustering in clusters:
			self.clusters = clustering
			acc = self._get_accuracy()
			if acc > best_acc:
				best_acc = acc
				best_clustering = clustering
		self.clusters = best_clustering

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "Deep Graph Clustering algorithm object"
