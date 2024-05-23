import torch
from torch_geometric.nn import ARGA as ARGAModel
from tqdm import trange

from algorithms.deep.DeepAlgorithm import DeepAlgorithm
from algorithms.deep.FCNet import FCNet
from algorithms.deep.GCNEncoder import GCNEncoder
from algorithms.deep.utils import get_clusters
from graph import Graph


class ARGA(DeepAlgorithm):
	"""Adversarially Regularized Graph Autoencoder algorithm

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
	:param k: Number of iterations to train the discriminator
	:type k: int
	:param use_pretrained: Boolean flag to indicate if pretrained model should be used
	:type use_pretrained: bool
	:param save_model: Boolean flag to indicate if the model should be saved after training
	:type save_model: bool
	"""

	def __init__(self, graph: Graph, num_clusters: int, lr: float = .001, latent_dim: int = 16, dropout: int = .0, epochs: int = 100, k: int = 3, use_pretrained: bool = True, save_model: bool = False):
		"""Constructor method
		"""
		super(ARGA, self).__init__(graph, num_clusters=num_clusters, lr=lr, latent_dim=latent_dim, dropout=dropout, epochs=epochs, use_pretrained=use_pretrained, save_model=save_model)
		self.k = k

		self.encoder: GCNEncoder = GCNEncoder(in_channels=graph.features.shape[1], latent_dim=latent_dim, dropout=dropout)
		self.discriminator: FCNet = FCNet(in_channels=latent_dim, out_channels=1)
		self.model: ARGAModel = ARGAModel(encoder=self.encoder, discriminator=self.discriminator)
		self.optimizer: torch.optim.Optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
		self.discriminator_optimizer: torch.optim.Optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
		if self.use_pretrained:
			self._load_pretrained()

	def _train(self) -> None:
		"""Trains the model
		"""
		x_t = torch.tensor(self.graph.features, dtype=torch.float)
		edge_index_t = torch.tensor(self.graph.edge_index, dtype=torch.long)
		for _ in (pbar := trange(self.epochs, desc="ARGA Training")):
			self.model.train()
			self.optimizer.zero_grad()
			z = self.model.encode(x_t, edge_index_t)
			# Training the discriminator
			for i in range(self.k):
				self.discriminator_optimizer.zero_grad()
				discriminator_loss = self.model.discriminator_loss(z)
				discriminator_loss.backward()
				self.discriminator_optimizer.step()
			# Training the encoder
			loss = self.model.recon_loss(z, edge_index_t) + self.model.reg_loss(z)
			loss.backward()
			self.optimizer.step()
			# Evaluation
			self.model.eval()
			self.clusters = get_clusters(z.detach().numpy(), self.num_clusters)
			evaluation = self.evaluate()
			pbar.set_postfix({"Loss": loss.item(), **dict(evaluation)})

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "ARGA algorithm object"
