import torch
from torch import nn
from torch.nn import functional as F


class FCNet(nn.Module):
	"""Fully connected three-layer neural network

	:param in_channels: Number of input channels
	:type in_channels: int
	:param out_channels: Number of output channels
	:type out_channels: int
	"""
	def __init__(self, in_channels: int, out_channels: int, dropout: float = .0):
		super(FCNet, self).__init__()
		self.dropout: float = dropout

		self.fc1: nn.Linear = nn.Linear(in_channels, out_channels * 4)
		self.fc2: nn.Linear = nn.Linear(out_channels * 4, out_channels * 2)
		self.fc3: nn.Linear = nn.Linear(out_channels * 2, out_channels)

	def forward(self, x: torch.tensor) -> torch.tensor:
		x = F.relu(self.fc1(x))
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = F.relu(self.fc2(x))
		x = F.dropout(x, p=self.dropout, training=self.training)
		return torch.sigmoid(self.fc3(x))
