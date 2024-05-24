"""Adapted from https://github.com/kavehhassani/mvgrl/blob/master/node/train.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
	"""Graph Convolutional Network (GCN) with as single layer

	:param in_channels: Number of input features
	:type in_channels: int
	:param out_channels: Number of output features
	:type out_channels: int
	"""

	def __init__(self, in_channels: int, out_channels: int):
		super(GCN, self).__init__()
		self.conv1: GCNConv = GCNConv(in_channels, out_channels)
		self.conv2: GCNConv = GCNConv(out_channels, out_channels)
		self.prelu: nn.PReLU = nn.PReLU()

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
		h1 = self.prelu(self.conv1(x, edge_index, edge_weight))
		return h1, self.prelu(self.conv2(h1, edge_index, edge_weight))


class Projection(nn.Module):
	"""Projection layer

	:param latent_dim: Dimension of the latent space
	:type latent_dim: int
	"""

	def __init__(self, latent_dim: int):
		super(Projection, self).__init__()
		self.fc1: nn.Linear = nn.Linear(latent_dim, latent_dim)
		self.fc2: nn.Linear = nn.Linear(latent_dim, latent_dim)
		self.prelu: nn.PReLU = nn.PReLU()

	def forward(self, h: torch.tensor) -> torch.tensor:
		"""Forward pass

		:param h: Node embeddings
		:type h: torch.tensor
		:return: Projected embeddings
		:rtype: torch.tensor
		"""
		return self.prelu(self.fc2(self.prelu(self.fc1(h))))


class Readout(nn.Module):
	"""Readout function for a two-layer GCN model
	"""

	def __init__(self, latent_dim: int):
		super(Readout, self).__init__()
		self.fc = nn.Linear(latent_dim * 2, latent_dim)

	def forward(self, h1: torch.tensor, h2: torch.tensor) -> torch.tensor:
		"""Pooling layer

		:param h1: Node embeddings at the first GCN layer
		:type h1: torch.tensor
		:param h2: Node embeddings at the second GCN layer
		:type h2: torch.tensor
		:return: Pooled embeddings
		:rtype: torch.tensor
		"""
		return F.sigmoid(self.fc(torch.cat([h1.mean(dim=-2), h2.mean(dim=-2)], dim=-1)))


class Discriminator(nn.Module):
	"""Discriminator module

	:param in_channels: Number of features in the hidden GCN layers
	:type in_channels: int
	"""

	def __init__(self, in_channels: int):
		super(Discriminator, self).__init__()
		self.bilinear: nn.Bilinear = nn.Bilinear(in_channels, in_channels, 1)

	def forward(self, ha: torch.tensor, hb: torch.tensor, Ha: torch.tensor, Hb: torch.tensor, Ha_corrupted: torch.tensor, Hb_corrupted: torch.tensor) -> torch.tensor:
		""" Forward pass of the discriminator computer the MI between the two representations of the views

		:param ha: Graph embedding of the original view
		:type ha: torch.tensor
		:param hb: Graph embedding of the diffused view
		:type hb: torch.tensor
		:param Ha: Node embedding of the original view
		:type Ha: torch.tensor
		:param Hb: Node embedding of the diffused view
		:type Hb: torch.tensor
		:param Ha_corrupted: Node embedding of the corrupted original view
		:type Ha_corrupted: torch.tensor
		:param Hb_corrupted: Node embedding of the corrupted diffused view
		:type Hb_corrupted: torch.tensor
		:return: Discriminator output
		:rtype: torch.tensor
		"""
		ha = ha.expand_as(Ha)
		hb = hb.expand_as(Hb)
		return torch.cat(
			[
				self.bilinear(hb, Ha).squeeze(),
				self.bilinear(ha, Hb).squeeze(),
				self.bilinear(hb, Ha_corrupted).squeeze(),
				self.bilinear(ha, Hb_corrupted).squeeze()
			], dim=-1
		)


class MVGRLModel(nn.Module):
	"""Multi-View Graph Representation Learning (MVGRL) model

	:param in_channels: Number of input features
	:type in_channels: int
	:param latent_dim: Dimension of the latent space
	:type latent_dim: int
	"""

	def __init__(self, in_channels: int, latent_dim: int):
		super(MVGRLModel, self).__init__()
		self.gcn_real: GCN = GCN(in_channels, latent_dim)
		self.gcn_diff: GCN = GCN(in_channels, latent_dim)
		self.readout: Readout = Readout(latent_dim)
		self.projector_nodes: Projection = Projection(latent_dim)
		self.projector_graph: Projection = Projection(latent_dim)
		self.discriminator: Discriminator = Discriminator(latent_dim)

	def forward(self, x: torch.tensor, edge_index: torch.tensor, diff_edge_index: torch.tensor, diff_edge_weight: torch.tensor, corrupted_idx: torch.tensor = None):
		"""Forward pass, a=alpha (original view), b=beta (diffused view)

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
		"""
		# Graph and node embeddings
		h1a, h2a = self.gcn_real(x, edge_index)
		h1b, h2b = self.gcn_diff(x, diff_edge_index, diff_edge_weight)
		Ha = self.projector_nodes(h2a)
		Hb = self.projector_nodes(h2b)
		ha = self.projector_graph(self.readout(h1a, h2a))
		hb = self.projector_graph(self.readout(h1b, h2b))
		# Corrupted features embeddings
		if corrupted_idx is None:
			corrupted_idx = torch.randperm(x.size(0))
		h1a_corrupted, h2a_corrupted = self.gcn_real(x[corrupted_idx], edge_index)
		h1b_corrupted, h2b_corrupted = self.gcn_diff(x[corrupted_idx], diff_edge_index, diff_edge_weight)
		Ha_corrupted = self.projector_nodes(h2a_corrupted)
		Hb_corrupted = self.projector_nodes(h2b_corrupted)
		# Discriminator output
		disc_out = self.discriminator(ha, hb, Ha, Hb, Ha_corrupted, Hb_corrupted)
		return disc_out, ha + hb, Ha + Hb

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
		_, h2a = self.gcn_real(x, edge_index)
		_, h2b = self.gcn_diff(x, diff_edge_index, diff_edge_weight)
		return self.projector_nodes(h2a) + self.projector_nodes(h2b)

