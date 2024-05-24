# Advanced Graph Clustering 

This project focuses on the study and implementation of various graph clustering techniques, covering traditional techniques such as Spectral Clustering and Leiden Method, as well as deep graph clustering methods like Graph Autoencoders. The project aims to provide a comprehensive overview of graph clustering algorithms and their applications in network analysis.

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

Note: For Deep Graph Clustering techniques, the hyperparameters are only relevant when a pre-trained model is not used.

Example usage:

```bash
cd src/
py main.py --algo gae --dataset cora --num_clusters 7 --use_pretrained
```

Use the `--help` flag to see the available options for the script (hyperparameters vary based on the clustering technique):

```bash
py main.py --help
```

## Implemented Techniques
- Traditional Clustering Techniques:
  - Spectral Clustering [1]
  - Stochastic Block Models [2] (using PySBM from [9])
  - Markov Clustering Algorithm [3]
  - Louvain Algorithm [4]
  - Leiden Algorithm [5]
- Deep Graph Clustering:
  - Graph Autoencoder (GAE) [6]
  - Adversarially Regularized Graph Autoencoder (ARGA) [7]
  - Multi-view Graph Representation Learning (MVGRL) [8]

## Datasets

The project uses several benchmark datasets for evaluating the clustering techniques.
The datasets are available in the `data/` directory and include the following:

| Dataset    | Nodes  | Edges | Features  | Classes | Description |
|------------|--------|-------|-----------|---------|-------------|
| Cora       | 2708   | 5429  | 1433      | 7       | Citation network |
| Citeseer   | 3327   | 4732  | 3703      | 6       | Citation network |
| DBLP       | 4057   | 3528  | 334       | 4       | Co-authorship network |
| Karateclub | 34     | 78    | 34        | 4       | Social network |

## References
[1] Ulrike von Luxburg. A tutorial on spectral clustering. (arXiv:0711.0189), November 2007. doi:
10.48550/arXiv.0711.0189. URL http://arxiv.org/
abs/0711.0189. arXiv:0711.0189 [cs].

[2] Clement Lee and Darren J. Wilkinson. A review of stochastic block models and extensions for graph clustering. Applied Network Science, 4(11):1–50, December 2019. ISSN 2364-8228. doi: 10.1007/s41109-019-0232-2.

[3] Stijn Van Dongen. Graph clustering via a discrete
uncoupling process. SIAM Journal on Matrix Analysis
and Applications, 30(1):121–141, January 2008. ISSN
0895-4798. doi: 10.1137/040608635.

[4] Vincent D. Blondel, Jean-Loup Guillaume, Renaud
Lambiotte, and Etienne Lefebvre. Fast unfolding of
communities in large networks. Journal of Statistical
Mechanics: Theory and Experiment, 2008(10):P10008,
October 2008. ISSN 1742-5468. doi: 10.1088/1742-
5468/2008/10/P10008. arXiv:0803.0476 [cond-mat,
physics:physics].

[5] V. A. Traag, L. Waltman, and N. J. van Eck. From louvain to leiden: guaranteeing well-connected communities. Scientific Reports, 9(1):5233, March 2019. ISSN 2045-2322. Doi: 10.1038/s41598-019-41695-z.

[6] Thomas N. Kipf and Max Welling. Variational graph
auto-encoders. (arXiv:1611.07308), November 2016. doi:
10.48550/arXiv.1611.07308. URL http://arxiv.org/
abs/1611.07308. arXiv:1611.07308 [cs, stat].

[7] Shirui Pan, Ruiqi Hu, Guodong Long, Jing Jiang, Lina Yao, and Chengqi Zhang.
Adversarially regularized graph autoencoder for graph embedding.
(arXiv:1802.04407), January 2019.
doi: 10.48550/ arXiv.1802.04407.
URL http://arxiv.org/abs/ 1802.04407.
arXiv:1802.04407 [cs, stat].

[8] Kaveh Hassani and Amir Hosein Khasahmadi.
Contrastive multi-view representation learning on graphs.
(arXiv:2006.05582), June 2020. doi: 10.48550/arXiv.2006.05582.
URL http://arxiv.org/abs/2006.05582. arXiv:2006.05582 [cs, stat].

[9] Funke T, Becker T (2019) Stochastic block models: A comparison of variants and inference methods. PLoS ONE 14(4): e0215296. https://doi.org/10.1371/journal.pone.0215296

## License
This project is licensed under the MIT License.