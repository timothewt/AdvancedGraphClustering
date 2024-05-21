from algorithms.Algorithm import Algorithm
from graph import Graph


class ARGA(Algorithm):
	"""Adversarially Regularized Graph Autoencoder algorithm
	"""

	def __init__(self, graph: Graph, num_clusters: int, lr: float, latent_dim: int, dropout: int, epochs: int):
		"""Constructor method

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
		super(ARGA, self).__init__(graph)
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
		return "ARGA algorithm object"
