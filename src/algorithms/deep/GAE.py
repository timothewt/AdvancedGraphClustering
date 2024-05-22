import torch
from torch_geometric.nn import GAE as GAEModel
from tqdm import trange

from algorithms.Algorithm import Algorithm
from algorithms.deep.GCNEncoder import GCNEncoder
from algorithms.deep.utils import get_clusters
from graph import Graph


class GAE(Algorithm):
	"""Graph Autoencoder algorithm

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

	def __init__(self, graph: Graph, num_clusters: int, lr: float, latent_dim: int, dropout: int, epochs: int):
		"""Constructor method
		"""
		super(GAE, self).__init__(graph)
		self.num_clusters: int = num_clusters
		self.lr: float = lr
		self.latent_dim: int = latent_dim
		self.dropout: int = dropout
		self.epochs: int = epochs

		self.encoder: GCNEncoder = GCNEncoder(in_channels=graph.features.shape[1], latent_dim=latent_dim, dropout=dropout)
		self.model: GAEModel = GAEModel(encoder=self.encoder)
		self.optimizer: torch.optim.Optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

	def _train(self) -> None:
		"""Trains the algorithm
		"""
		x_t = torch.tensor(self.graph.features, dtype=torch.float)
		edge_index_t = torch.tensor(self.graph.edge_index, dtype=torch.long)
		for _ in (pbar := trange(self.epochs, desc="GAE Training")):
			self.optimizer.zero_grad()
			z = self.model.encode(x_t, edge_index_t)
			loss = self.model.recon_loss(z, edge_index_t)
			loss.backward()
			self.optimizer.step()
			self.clusters = get_clusters(z.detach().numpy(), self.num_clusters)
			evaluation = self.evaluate()
			pbar.set_postfix({"Loss": loss.item(), **{metric: value for metric, value in evaluation}})

	def run(self) -> None:
		"""Runs the algorithm
		# TODO refactor this method in a parent class DeepAlgorithm
		"""
		self._train()
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
		return "GAE algorithm object"
