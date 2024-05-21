from graph import Graph
from algorithms import AdaGAE, GAE, ARGA, Markov, Louvain, SBM, Spectral
import numpy as np


def main():
	"""Main function
	"""
	adj = np.array([
		[0, 0, 0, 1, 1, 0, 1, 0],
		[0, 0, 1, 0, 0, 0, 0, 1],
		[0, 1, 0, 0, 0, 1, 0, 0],
		[1, 0, 0, 0, 0, 0, 1, 0],
		[1, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 1, 0, 0, 0, 1, 1],
		[1, 0, 0, 1, 0, 1, 0, 0],
		[0, 1, 0, 0, 0, 1, 0, 0],
	])
	graph = Graph(adj)
	print(graph)
	sp = Louvain(graph)
	sp.run()
	print(sp.clusters)
	graph.draw(colors=["red" if cluster == 0 else "blue" for cluster in sp.clusters])


if __name__ == "__main__":
	main()
