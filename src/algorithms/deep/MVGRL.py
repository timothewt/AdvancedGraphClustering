import torch
from torch import nn
from torch_geometric.utils import dense_to_sparse
from tqdm import trange

from algorithms.deep.DeepAlgorithm import DeepAlgorithm
from algorithms.deep.models.MVGRLModel import MVGRLModel
from algorithms.deep.utils import get_clusters, compute_diffusion_matrix
from graph import Graph


class MVGRL(DeepAlgorithm):
	"""Multi-View Graph Representation Learning algorithm

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
	:param use_pretrained: Boolean flag to indicate if pretrained model should be used
	:type use_pretrained: bool
	:param save_model: Boolean flag to indicate if the model should be saved after training
	:type save_model: bool
	"""

	def __init__(self, graph: Graph, num_clusters: int, lr: float = .001, latent_dim: int = 16, dropout: int = .0, epochs: int = 100, use_pretrained: bool = True, save_model: bool = False):
		"""Constructor method
		"""
		super(MVGRL, self).__init__(graph, num_clusters=num_clusters, lr=lr, latent_dim=latent_dim, dropout=dropout, epochs=epochs, use_pretrained=use_pretrained, save_model=save_model)

		self.model: MVGRLModel = MVGRLModel(in_channels=graph.features.shape[1], latent_dim=latent_dim, dropout=dropout)
		if self.use_pretrained:
			self._load_pretrained()
		self.optimizer: torch.optim.Optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

		# Compute the diffusion matrix
		diff = torch.from_numpy(compute_diffusion_matrix(graph.adj_matrix)).float()
		diff[diff < 1e-5] = 0  # remove small values for computational time (sparse matrix, from 6M to 300k edges for cora)
		diff = dense_to_sparse(diff)
		self.diff_edge_index = diff[0]
		self.diff_edge_weight = diff[1].float()

	def _train(self) -> None:
		"""Trains the model
		"""
		x_t = torch.tensor(self.graph.features, dtype=torch.float)
		edge_index_t = torch.tensor(self.graph.edge_index, dtype=torch.long)

		true_labels = torch.ones(x_t.size(0) * 2, dtype=torch.float)
		labels = torch.cat([true_labels, true_labels * 0], dim=-1)
		criterion = nn.BCEWithLogitsLoss()
		for _ in (pbar := trange(self.epochs, desc="GAE Training")):
			self.model.train()
			self.optimizer.zero_grad()
			# Training the model
			discriminator_output, _, _, _, _ = self.model(x_t, edge_index_t, self.diff_edge_index, self.diff_edge_weight)
			loss = criterion(discriminator_output, labels)
			loss.backward()
			self.optimizer.step()
			# Evaluation
			self.model.eval()
			self.clusters = get_clusters(self._encode_nodes(), self.num_clusters)
			evaluation = self.evaluate()
			pbar.set_postfix({"Loss": loss.item(), **dict(evaluation)})

	def _encode_nodes(self) -> torch.tensor:
		"""Encodes the node features using the model

		:return: Node embeddings
		:rtype: torch.tensor
		"""
		return self.model.encode(self.x_t, self.edge_index_t, self.diff_edge_index, self.diff_edge_weight).detach().numpy()

	def __str__(self):
		"""Returns the string representation of the algorithm object

		:return: String representation of the algorithm object
		:rtype: str
		"""
		return "MVGRL algorithm object"
