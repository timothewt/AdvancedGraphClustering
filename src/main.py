import argparse
import os

from prettytable import PrettyTable
import numpy as np

from graph import Graph
from algorithms import GAE, ARGA, MVGRL, Markov, Louvain, Leiden, SBM_em, SBM_metropolis, Spectral, DCSBM


def main():
	"""Main function
	"""
	parser = argparse.ArgumentParser(description="Graph Embedding Algorithms", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# Algorithm
	parser.add_argument("--algo", type=str, default="gae", help="Algorithm to use (gae, arga, mvgrl, markov, louvain, leiden, dcsbm, sbm_em, sbm_metropolis, spectral)")

	# Dataset
	parser.add_argument("--dataset", type=str, help="Dataset to use (karateclub, cora, citeseer, uat). If not provided, use custom dataset")

	# Custom Dataset
	parser.add_argument("--adj", type=str, help="Graph adjacency matrix as .csv (no header and index)")
	parser.add_argument("--features", type=str, help="Graph features matrix as .csv (no header and index)")
	parser.add_argument("--labels", type=str, help="Graph labels matrix as .csv (no header and index)")

	# Markov clustering parameters
	parser.add_argument("--expansion", type=int, default=2, help="Expand parameter for Markov")
	parser.add_argument("--inflation", type=int, default=2, help="Inflation parameter for Markov ")
	parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for Markov and SBM")

	# For Deep, Spectral and SBM
	parser.add_argument("--num_clusters", type=int, default=7, help="Number of clusters for Spectral clustering and Deep Graph Clustering")

	# Deep parameters
	default_latent_dim = [32, 32, 128]  # Default latent dimensions for GAE, ARGA, and MVGRL
	parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for Deep Graph Clustering")
	parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Deep Graph Clustering")
	parser.add_argument("--latent_dim", type=int, default=default_latent_dim[0], help="Latent dimension for Deep Graph Clustering")
	parser.add_argument("--use_pretrained", action="store_true", help="Use a pretrained model for Deep Graph Clustering")
	parser.add_argument("--save_model", action="store_true", help="Save the model after training if use_pretrained is not specified")

	# Output
	parser.add_argument("--output_path", type=str, default="../output", help="Output path to save the results")
	parser.add_argument("--draw_clusters", action="store_true", help="Draw clusters after clustering")

	args = parser.parse_args()

	# Loading the dataset
	if args.dataset:
		assert args.dataset.lower() in ["karateclub", "cora", "citeseer", "uat"], "Invalid dataset"
		adj = np.load(f"../data/{args.dataset.lower()}/adj.npy")
		features = np.load(f"../data/{args.dataset.lower()}/feat.npy").astype(np.float32)
		labels = np.load(f"../data/{args.dataset.lower()}/label.npy")
		graph = Graph(adj_matrix=adj, features=features, labels=labels, dataset_name=args.dataset)
	else:
		assert args.adj, "Adjacency matrix is required when dataset is not provided"
		adj = np.loadtxt(args.adj, delimiter=",").astype(np.uint8)
		features = np.loadtxt(args.features, delimiter=",").astype(np.float32) if args.features else None
		labels = np.loadtxt(args.labels, delimiter=",").astype(np.uint32) if args.labels else None
		graph = Graph(adj_matrix=adj, features=features, labels=labels)

	# Ensuring that the pretrained folder exists
	if args.use_pretrained or args.save_model:
		os.mkdir("algorithms/deep/pretrained") if not os.path.exists("algorithms/deep/pretrained") else None

	# Instantiating the algorithm
	match args.algo.lower():
		case "gae":
			latent_dim = default_latent_dim[0] if args.use_pretrained else args.latent_dim
			algo = GAE(graph, num_clusters=args.num_clusters, epochs=args.epochs, lr=args.lr, latent_dim=latent_dim, use_pretrained=args.use_pretrained, save_model=args.save_model)
		case "arga":
			latent_dim = default_latent_dim[1] if args.use_pretrained else args.latent_dim
			algo = ARGA(graph, num_clusters=args.num_clusters, epochs=args.epochs, lr=args.lr, latent_dim=latent_dim, use_pretrained=args.use_pretrained, save_model=args.save_model)
		case "mvgrl":
			latent_dim = default_latent_dim[2] if args.use_pretrained else args.latent_dim
			algo = MVGRL(graph, num_clusters=args.num_clusters, epochs=args.epochs, lr=args.lr, latent_dim=latent_dim, use_pretrained=args.use_pretrained, save_model=args.save_model)
		case "markov":
			algo = Markov(graph, expansion=args.expansion, inflation=args.inflation, iterations=args.iterations)
		case "louvain":
			algo = Louvain(graph)
		case "leiden":
			algo = Leiden(graph)
		case "sbm_metropolis":
			algo = SBM_metropolis(graph, num_clusters=args.num_clusters, iterations=args.iterations)
		case "sbm_em":
			algo = SBM_em(graph, num_clusters=args.num_clusters, iterations=args.iterations)
		case "dcsbm":
			algo = DCSBM(graph, num_clusters=args.num_clusters, iterations=args.iterations)
		case "spectral":
			algo = Spectral(graph, num_clusters=args.num_clusters)
		case _:
			raise ValueError("Invalid algorithm")

	# Running the algorithm
	print("Running algorithm:", args.algo)
	algo.run()
	print("Done!\n")

	# Evaluating the clustering
	evaluation = algo.evaluate()
	print("  Evaluation:")
	table = PrettyTable(["Metric", "Value"])
	for metric, value in evaluation:
		table.add_row([metric, f"{value:.3}"])
	print(table, end="\n\n")

	# Saving the resulting clustering
	if not os.path.exists(args.output_path):
		os.makedirs(args.output_path)
	output_file = os.path.abspath(f"{args.output_path}/{ds + '_' if (ds := args.dataset) is not None else ''}{args.algo}_clusters.csv")
	print(f"Saving the resulting clustering to {output_file}.")
	np.savetxt(output_file, np.stack([range(graph.adj_matrix.shape[0]), algo.clusters], axis=1), delimiter=",", fmt="%d",)

	# Optionally, draw the clusters
	if args.draw_clusters:
		print("Drawing clusters...")
		graph.draw(clusters=algo.clusters)


if __name__ == "__main__":
	"""Example usage:
	$ python main.py --algo gae --dataset cora --num_clusters 7 --epochs 50 --lr 0.001 --latent_dim 32 --draw_clusters
	"""
	main()
