def extract_clusters_from_communities_list(communities: list[list[int]]) -> list[int]:
	"""Extracts clusters from the output of the cdlib library / markov clustering algorithm

	:param communities: List of communities (list of nodes)
	:type communities: list[list[int]]
	:return: List of clusters
	:rtype: list
	"""
	clusters = {}
	for i, community in enumerate(communities):
		for node in community:
			clusters[node] = i

	return [v for _, v in sorted(clusters.items())]
