import argparse
import os

import numpy as np

from graph import Graph
from algorithms import AdaGAE, GAE, ARGA, Markov, Louvain, SBM, Spectral


def main():
	"""Main function
	"""
	parser = argparse.ArgumentParser(description="Graph Embedding Algorithms")
	parser.add_argument("--algorithm", type=str, default="gae", help="Algorithm to use")
	# Dataset
	parser.add_argument("--dataset", type=str, help="Dataset to use (cora, citeseer, pubmed). If not provided, use custom dataset")
	# Custom Dataset
	parser.add_argument("--adj", type=str, help="Graph adjacency matrix as .csv (no header and index)")
	parser.add_argument("--features", type=str, help="Graph features matrix as .csv (no header and index)")
	parser.add_argument("--labels", type=str, help="Graph labels matrix as .csv (no header and index)")
	# Markov clustering parameters
	parser.add_argument("--expansion", type=float, default=2.0, help="Expand parameter for Markov clustering")
	parser.add_argument("--inflation", type=float, default=2.0, help="Inflation parameter for Markov clustering")
	parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for Markov clustering")
	# Spectral clustering parameters
	parser.add_argument("--num_clusters", type=int, default=3, help="Number of clusters for Spectral clustering and Deep Graph Clustering")
	# Deep parameters
	parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for Deep Graph Clustering")
	parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for Deep Graph Clustering")
	parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension for Deep Graph Clustering")
	parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for Deep Graph Clustering")
	# Output
	parser.add_argument("--output_path", type=str, default="../output", help="Output path to save the results")
	parser.add_argument("--draw_clusters", action="store_true", help="Draw clusters after clustering")

	args = parser.parse_args()

	if args.dataset:
		assert args.dataset in ["cora", "citeseer", "pubmed"], "Invalid dataset"
		adj = np.load(f"../data/{args.dataset}/adj.npy")
		features = np.load(f"../data/{args.dataset}/feat.npy").astype(np.float32)
		labels = np.load(f"../data/{args.dataset}/label.npy")
		graph = Graph(adj_matrix=adj, features=features, labels=labels)
	else:
		assert args.adj, "Adjacency matrix is required when dataset is not provided"
		adj = np.loadtxt(args.adj, delimiter=",").astype(np.uint8)
		features = np.loadtxt(args.features, delimiter=",").astype(np.float32) if args.features else None
		labels = np.loadtxt(args.labels, delimiter=",").astype(np.uint32) if args.labels else None
		graph = Graph(adj_matrix=adj, features=features, labels=labels)

	if args.algorithm == "gae":
		algo = GAE(graph, num_clusters=args.num_clusters, epochs=args.epochs, lr=args.lr, latent_dim=args.latent_dim, dropout=args.dropout)
	elif args.algorithm == "adagae":
		algo = AdaGAE(graph, num_clusters=args.num_clusters, epochs=args.epochs, lr=args.lr, latent_dim=args.latent_dim, dropout=args.dropout)
	elif args.algorithm == "arga":
		algo = ARGA(graph, num_clusters=args.num_clusters, epochs=args.epochs, lr=args.lr, latent_dim=args.latent_dim, dropout=args.dropout)
	elif args.algorithm == "markov":
		algo = Markov(graph, expansion=args.expansion, inflation=args.inflation, iterations=args.iterations)
	elif args.algorithm == "louvain":
		algo = Louvain(graph)
	elif args.algorithm == "sbm":
		algo = SBM(graph)
	elif args.algorithm == "spectral":
		algo = Spectral(graph, num_clusters=args.num_clusters)
	else:
		raise ValueError("Invalid algorithm")

	# Running the algorithm
	print("Running algorithm:", args.algorithm)
	algo.run()
	print("Done!")
	# TODO: Add evaluation metrics

	# Saving the resulting clustering
	if not os.path.exists(args.output_path):
		os.makedirs(args.output_path)
	output_file = os.path.abspath(f"{args.output_path}/{ds + '_' if (ds := args.dataset) is not None else ''}{args.algorithm}_clusters.csv")
	print(f"Saving the resulting clustering to {output_file}.")
	np.savetxt(output_file, np.stack([range(graph.adj_matrix.shape[0]), algo.clusters], axis=1), delimiter=",", fmt="%d",)

	# Optionally, draw the clusters
	if args.draw_clusters:
		print("Drawing clusters...")
		graph.draw(clusters=algo.clusters)


if __name__ == "__main__":
	main()
