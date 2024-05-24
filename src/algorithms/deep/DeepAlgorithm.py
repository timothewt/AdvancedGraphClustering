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
	:param epochs: Number of epochs to run
	:type epochs: int
	:param use_pretrained: Boolean flag to indicate if pretrained model should be used
	:type use_pretrained: bool
	:param save_model: Boolean flag to indicate if the model should be saved after training
	:type save_model: bool
	"""

	def __init__(self, graph: Graph, num_clusters: int, lr: float = .001, latent_dim: int = 16, epochs: int = 100, use_pretrained: bool = True, save_model: bool = False):
		"""Constructor method
		"""
		super(DeepAlgorithm, self).__init__(graph)
		self.num_clusters: int = num_clusters
		self.lr: float = lr
		self.latent_dim: int = latent_dim
		self.epochs: int = epochs
		self.use_pretrained = use_pretrained
		self.save_model = save_model

		self.x_t = torch.tensor(self.graph.features, dtype=torch.float)
		self.edge_index_t = torch.tensor(self.graph.edge_index, dtype=torch.long)

		self.model: torch.nn.Module = torch.nn.Module()

	def _train(self) -> None:
		"""Trains the model, to be implemented by subclasses
		"""
		raise NotImplementedError

	def _encode_nodes(self) -> torch.tensor:
		"""Encodes the node features using the model

		:return: Node embeddings
		:rtype: torch.tensor
		"""
		return self.model.encode(self.x_t, self.edge_index_t).detach().numpy()

	def run(self) -> None:
		"""Trains the model and runs k-means clustering on the node embeddings.
		"""
		if not self.use_pretrained:
			self._train()
			if self.save_model:
				torch.save(self.model.state_dict(), f"algorithms/deep/pretrained/{self.__class__.__name__.lower()}_{self.graph.dataset_name}.pt")
		self.model.eval()
		z_np = self._encode_nodes()
		clusters = [
			get_clusters(z_np, self.num_clusters) for _ in range(10)
		]  # Run clustering 10 times and get the best clustering
		best_clustering = None
		best_acc = 0
		for clustering in clusters:
			self.clusters = clustering
			acc = self._get_accuracy()
			if acc > best_acc:
				best_acc = acc
				best_clustering = clustering
		self.clusters = best_clustering

	def _load_pretrained(self) -> None:
		"""Loads the pretrained model
		"""
		try:
			self.model.load_state_dict(torch.load(f"algorithms/deep/pretrained/{self.__class__.__name__.lower()}_{self.graph.dataset_name}.pt"))
		except FileNotFoundError:
			print("No pretrained model found.")
			self.use_pretrained = False

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "Deep Graph Clustering algorithm object"
