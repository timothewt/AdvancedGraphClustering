import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
	"""Graph Convolutional Network Encoder for Graph Autoencoder

	:param in_channels: Number of input channels
	:type in_channels: int
	:param latent_dim: Latent dimension
	:type latent_dim: int
	:param dropout: Dropout rate
	:type dropout: float
	"""
	def __init__(self, in_channels: int, latent_dim: int, dropout: float = 0.):
		super(GCNEncoder, self).__init__()
		self.dropout: float = dropout

		self.conv1: GCNConv = GCNConv(in_channels, latent_dim * 2)
		self.conv2: GCNConv = GCNConv(latent_dim * 2, latent_dim)

	def forward(self, x: torch.tensor, edge_index: torch.tensor) -> torch.tensor:
		"""Forward pass

		:param x: Node features
		:type x: torch.tensor
		:param edge_index: Edge index
		:type edge_index: torch.tensor
		:return: Node embeddings
		:rtype: torch.tensor
		"""
		x = F.relu(self.conv1(x, edge_index))
		x = F.dropout(x, p=self.dropout, training=self.training)
		return self.conv2(x, edge_index)
