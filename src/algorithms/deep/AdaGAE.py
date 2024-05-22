from algorithms.Algorithm import Algorithm
from graph import Graph


class AdaGAE(Algorithm):
	"""Adaptive Graph Autoencoder algorithm

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
		super(AdaGAE, self).__init__(graph)
		self.num_clusters: int = num_clusters
		self.lr: float = lr
		self.latent_dim: int = latent_dim
		self.dropout: int = dropout
		self.epochs: int = epochs

	def run(self) -> None:
		"""Runs the algorithm
		"""
		pass

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "AdaGAE algorithm object"
