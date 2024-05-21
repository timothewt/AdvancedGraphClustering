# Advanced Graph Clustering 

This project focuses on the study and implementation of various graph clustering techniques, covering traditional techniques such as Spectral Clustering and Louvain Method, as well as deep graph clustering methods like Graph Autoencoders. The project aims to provide a comprehensive overview of graph clustering algorithms and their applications in network analysis.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Implemented Techniques](#implemented-techniques)
- [License](#license)

## Introduction

Graph clustering is an essential task in network analysis, aimed at partitioning a graph into meaningful groups or clusters. This project explores and implements several prominent graph clustering algorithms to analyze and understand complex networks.

## Installation

To use this project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/graph-clustering.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the implemented clustering techniques, execute the respective scripts provided in the project. Detailed usage instructions for each technique can be found in their corresponding documentation on the GitHub page.

Example usage:

```bash
cd src/
py main.py --algorithm spectral --adj adjacency_matrix.csv --features node_features.csv --num_clusters 3
```

Use the `--help` flag to see the available options for the script (hyperparameters vary based on the clustering technique):

```bash
py main.py --help
```

## Implemented Techniques
- Traditional Clustering Techniques:
  - Louvain Method
  - Markov Clustering Algorithm
  - Spectral Clustering
  - Stochastic Block Models
- Deep Graph Clustering:
  - Graph Autoencoder (GAE)
  - Adversarially Regularized Graph Autoencoder (ARGA)
  - Adaptative Graph Autoencoder (AdaGAE)

## License
This project is licensed under the MIT License.