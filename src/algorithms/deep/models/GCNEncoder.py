import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
	"""Graph Convolutional Network Encoder for Graph Autoencoder

	:param in_channels: Number of input channels
	:type in_channels: int
	:param latent_dim: Latent dimension
	:type latent_dim: int
	:param activation: Activation function
	:type activation: nn.Module
	"""
	def __init__(self, in_channels: int, latent_dim: int, activation: nn.Module = nn.ReLU):
		super(GCNEncoder, self).__init__()

		self.conv1: GCNConv = GCNConv(in_channels, latent_dim * 2)
		self.conv2: GCNConv = GCNConv(latent_dim * 2, latent_dim)
		self.act: nn.Module = activation()

	def forward(self, x: torch.tensor, edge_index: torch.tensor) -> torch.tensor:
		"""Forward pass

		:param x: Node features
		:type x: torch.tensor
		:param edge_index: Edge index
		:type edge_index: torch.tensor
		:return: Node embeddings
		:rtype: torch.tensor
		"""
		x = self.act(self.conv1(x, edge_index))
		return self.conv2(x, edge_index)
