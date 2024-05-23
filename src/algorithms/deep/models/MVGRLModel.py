"""Adapted from https://github.com/kavehhassani/mvgrl/blob/master/node/train.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
	"""Graph Convolutional Network (GCN) module that returns the embeddings of the nodes at each GCN layer. The model is
	redefined to enable the output of each layer.

	:param in_channels: Number of input features
	:type in_channels: int
	:param out_channels: Number of output features
	:type out_channels: int
	"""

	def __init__(self, in_channels: int, out_channels: int, dropout: float = .0):
		super(GCN, self).__init__()
		self.conv1: GCNConv = GCNConv(in_channels, out_channels)
		self.conv2: GCNConv = GCNConv(out_channels, out_channels)
		self.prelu: nn.PReLU = nn.PReLU()
		self.dropout: float = dropout

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor = None) -> tuple[
		torch.tensor, torch.tensor]:
		"""Forward pass

		:param x: Input features
		:type x: torch.tensor
		:param edge_index: Edge index tensor
		:type edge_index: torch.tensor
		:param edge_weight: Edge weight tensor (if any)
		:type edge_weight: torch.tensor
		:return: Embeddings of the nodes at each GCN layer
		:rtype: tuple[torch.tensor, torch.tensor]
		"""
		h1 = self.conv1(x, edge_index, edge_weight)
		h2 = F.dropout(self.prelu(h1), p=self.dropout, training=self.training)
		return h1, self.conv2(h2, edge_index, edge_weight)


class Readout(nn.Module):
	"""Readout function for a two-layer GCN model

	:param in_channels: Number of input features in each hidden layer
	:type in_channels: int
	"""

	def __init__(self, in_channels: int):
		super(Readout, self).__init__()
		self.linear: nn.Linear = nn.Linear(in_channels * 2, in_channels, bias=False)
		self.prelu: nn.PReLU = nn.PReLU()

	def forward(self, h1: torch.tensor, h2: torch.tensor) -> torch.tensor:
		"""Pooling layer

		:param h1: Embeddings of the nodes at the first GCN layer
		:type h1: torch.tensor
		:param h2: Embeddings of the nodes at the second GCN layer
		:type h2: torch.tensor
		:return: Pooled embeddings
		:rtype: torch.tensor
		"""
		h = torch.cat([h1.mean(dim=-2), h2.mean(dim=-2)], dim=-1)
		return self.prelu(self.linear(h))  # Eq. (4)


class Discriminator(nn.Module):
	"""Discriminator module

	:param in_channels: Number of features in the hidden GCN layers
	:type in_channels: int
	"""

	def __init__(self, in_channels: int):
		super(Discriminator, self).__init__()
		self.bilinear: nn.Bilinear = nn.Bilinear(in_channels, in_channels, 1)

	def forward(self, h1: torch.tensor, h2: torch.tensor, h3: torch.tensor, h4: torch.tensor, r1: torch.tensor,
				r2: torch.tensor) -> torch.tensor:
		"""Forward pass

		:param h1: Embeddings of the nodes of the original view after the GCN_real layers
		:type h1: torch.tensor
		:param h2: Embeddings of the nodes of the diffused view after the GCN_diff layers
		:type h2: torch.tensor
		:param h3: Embeddings of the shuffled (corrupted) nodes of the original view after the GCN_real layers
		:type h3: torch.tensor
		:param h4: Embeddings of the shuffled (corrupted) nodes of the diffused view after the GCN_diff layers
		:type h4: torch.tensor
		:param r1: Output of the readout layer for the original view
		:type r1: torch.tensor
		:param r2: Output of the readout layer for the diffused view
		:type r2: torch.tensor
		:return: Discriminator output of shape for each gcn layer: positive (real) and negative (corrupted) pairs
		:rtype: torch.tensor
		"""
		r1 = r1.expand_as(h1)
		r2 = r2.expand_as(h2)
		return torch.cat([
			self.bilinear(h2, r1).squeeze(),  # diffused nodes vs real pooled, should be positive
			self.bilinear(h1, r2).squeeze(),  # real nodes vs diffused pooled, should be positive
			self.bilinear(h4, r1).squeeze(),  # corrupted diffused nodes vs real pooled, should be negative
			self.bilinear(h3, r2).squeeze()  # corrupted real nodes vs diffused pooled, should be negative
		], dim=-1)


class MVGRLModel(nn.Module):
	"""Multi-View Graph Representation Learning (MVGRL) model

	:param in_channels: Number of input features
	:type in_channels: int
	:param latent_dim: Dimension of the latent space
	:type latent_dim: int
	"""

	def __init__(self, in_channels: int, latent_dim: int, dropout: int):
		super(MVGRLModel, self).__init__()
		self.gcn_real: GCN = GCN(in_channels, latent_dim, dropout)
		self.gcn_diff: GCN = GCN(in_channels, latent_dim, dropout)
		self.readout: Readout = Readout(latent_dim)
		self.discriminator: Discriminator = Discriminator(latent_dim)

	def forward(self, x: torch.tensor, edge_index: torch.tensor, diff_edge_index: torch.tensor, diff_edge_weight: torch.tensor, corrupted_idx: torch.tensor = None) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
		"""Forward pass

		:param x: Input features
		:type x: torch.tensor
		:param edge_index: Edge index tensor
		:type edge_index: torch.tensor
		:param diff_edge_index: Diffused edge index tensor
		:type diff_edge_index: torch.tensor
		:param diff_edge_weight: Diffused edge weight tensor
		:type diff_edge_weight: torch.tensor
		:param corrupted_idx: Corrupted index tensor
		:type corrupted_idx: torch.tensor
		:return: Discriminator output, readout output of the original view, readout output of the diffused view, embeddings of the nodes of the original view, embeddings of the nodes of the diffused view
		:rtype: tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]
		"""
		h1_real, h2_real = self.gcn_real(x, edge_index)
		h1_diff, h2_diff = self.gcn_diff(x, diff_edge_index, diff_edge_weight)
		if corrupted_idx is None:
			corrupted_idx = torch.randperm(x.size(0))
		h3_real, h4_real = self.gcn_real(x[corrupted_idx], edge_index)
		h3_diff, h4_diff = self.gcn_diff(x[corrupted_idx], diff_edge_index, diff_edge_weight)
		r1 = self.readout(h1_real, h2_real)
		r2 = self.readout(h1_diff, h2_diff)
		return self.discriminator(h2_real, h2_diff, h4_real, h4_diff, r1, r2), r1, r2, h1_real, h1_diff

	def encode(self, x: torch.tensor, edge_index: torch.tensor, diff_edge_index: torch.tensor, diff_edge_weight: torch.tensor) -> torch.tensor:
		"""Embedding function

		:param x: Input features
		:type x: torch.tensor
		:param edge_index: Edge index tensor
		:type edge_index: torch.tensor
		:param diff_edge_index: Diffused edge index tensor
		:type diff_edge_index: torch.tensor
		:param diff_edge_weight: Diffused edge weight tensor
		:type diff_edge_weight: torch.tensor
		:return: Node embeddings
		:rtype: torch.tensor
		"""
		_, h_real = self.gcn_real(x, edge_index)
		_, h_diff = self.gcn_diff(x, diff_edge_index, diff_edge_weight)
		return h_real + h_diff
