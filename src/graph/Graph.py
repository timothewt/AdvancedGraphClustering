import networkx as nx
import numpy as np
import matplotlib.cm as mplcm
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt


class Graph:
	"""Graph class to represent a graph using the adjacency matrix and features

    :param adj_matrix: Adjacency matrix of the graph of shape (n, n) where n is the number of nodes in the graph
    :type adj_matrix: np.ndarray
    :param edge_index: Edge index of the graph of shape (2, e) where e is the number of edges in the graph
    :type edge_index: np.ndarray
    :param edge_weight: Edge weight of the graph of shape (e,) where e is the number of edges in the graph
    :type edge_weight: np.ndarray
    :param features: Features of the nodes in the graph of shape (n, f) where f is the number of features per node
    :type features: np.ndarray
    :param labels: Labels of the nodes in the graph of shape (n,) where n is the number of nodes in the graph
    :type labels: np.ndarray
    :param directed: Boolean flag to indicate if the graph is directed or undirected
    :type directed: bool
    :param nx_graph: NetworkX graph object
    :type nx_graph: nx.Graph | nx.DiGraph
    """

	def __init__(self, adj_matrix: np.ndarray, directed: bool = False, features: np.ndarray = None, labels: np.ndarray = None):
		"""Constructor method
		"""
		self.adj_matrix: np.ndarray = adj_matrix
		self.edge_index: np.ndarray = np.argwhere(adj_matrix).transpose()
		self.edge_weight: np.ndarray = adj_matrix[self.edge_index.transpose()[:, 0], self.edge_index.transpose()[:, 1]]
		self.features: np.ndarray = features if features is not None else np.ones((adj_matrix.shape[0], 1))
		self.labels: np.ndarray = labels
		self.directed: bool = directed
		self.nx_graph: nx.Graph | nx.DiGraph = nx.DiGraph() if directed else nx.Graph()
		self._create_nx_graph()

	def _create_nx_graph(self) -> None:
		"""Adds nodes and edges to the NetworkX graph object
		"""
		self.nx_graph.add_nodes_from(range(self.features.shape[0]), features=self.features)
		self.nx_graph.add_edges_from(self.edge_index.transpose())
		for i, (u, v) in enumerate(self.edge_index.transpose()):
			self.nx_graph[u][v]['weight'] = self.edge_weight[i]

	def draw(self, clusters: list[int] = None, draw_labels: bool = False) -> None:
		"""Draws the graph using the NetworkX draw method. If clusters are provided, the nodes are colored based on the
		clusters based on https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib

		:param clusters: List of cluster labels for each node
		:type clusters: list[int]
		:param draw_labels: Boolean flag to indicate if the labels should be drawn
		:type draw_labels: bool
		"""
		if clusters is not None:
			num_colors = max(clusters) + 1
			cm = plt.get_cmap('gist_rainbow')
			cNorm = mplcolors.Normalize(vmin=0, vmax=num_colors - 1)
			scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
			color_map = [scalarMap.to_rgba(i) for i in range(num_colors)]
			colors = [color_map[clusters[i]] for i in range(len(clusters))]
		else:
			colors = "darkgray"
		node_size = 50 if len(self.features) < 50 else 5
		edge_width = 1 if len(self.features) < 50 else 0.75
		pos = nx.spring_layout(self.nx_graph)
		nx.draw_networkx_nodes(self.nx_graph, pos, edgecolors="darkgray", linewidths=.5, node_color=colors, node_size=node_size)
		nx.draw_networkx_edges(self.nx_graph, pos, width=edge_width, alpha=0.5, edge_color="gray")
		if draw_labels:
			nx.draw_networkx_labels(self.nx_graph, pos)
		plt.tight_layout()
		plt.axis("off")
		plt.show()

	def __str__(self):
		"""Returns the string representation of the graph object

		:return: String representation of the graph object (number of nodes and edges)
		:rtype: str
		"""
		return f"Graph with {self.adj_matrix.shape[0]} nodes and {self.edge_index.shape[1]} edges."

	def __repr__(self):
		"""Returns the string representation of the graph object

		:return: String representation of the graph object (number of nodes and edges)
		:rtype: str
		"""
		return self.__str__()

	def __getitem__(self, key: int) -> np.ndarray:
		"""Returns the adjacency matrix of the graph for the given key

		:param key: Key to access the adjacency matrix
		:type key: int
		:return: Adjacency matrix of the graph for the given key
		:rtype: np.ndarray
		"""
		return self.adj_matrix[key]
