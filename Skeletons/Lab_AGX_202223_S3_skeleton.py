import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import louvain_communities, modularity

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def num_common_nodes(*arg):
    """
    Return the number of common nodes between a set of graphs.

    :param arg: (an undetermined number of) networkx graphs.
    :return: an integer, number of common nodes.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    if (len(arg) > 0) and (len(arg[0]) > 0): #Check that the input is valid    
        
        common_nodes = set(arg[0].nodes) 
        for graph in arg[1:]:
            nodes = set(graph.nodes)
            common_nodes = common_nodes.intersection(nodes)

    return len(common_nodes)
    # ----------------- END OF FUNCTION --------------------- #

def get_degree_distribution(g: nx.Graph) -> dict:
    """
    Get the degree distribution of the graph.

    :param g: networkx graph.
    :return: dictionary with degree distribution (keys are degrees, values are number of occurrences).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    degree_count = nx.degree_histogram(g)
    degree_distribution = {degree: count for degree, count in enumerate(degree_count)}
    return degree_distribution
    # ----------------- END OF FUNCTION --------------------- #


def get_k_most_central(g: nx.Graph, metric: str, num_nodes: int) -> list:
    """
    Get the k most central nodes in the graph.

    :param g: networkx graph.
    :param metric: centrality metric. Can be (at least) 'degree', 'betweenness', 'closeness' or 'eigenvector'.
    :param num_nodes: number of nodes to return.
    :return: list with the top num_nodes nodes with the specified centrality.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    if metric == "degree":
        most_central = sorted(nx.degree_centrality(g).items(), key=lambda item: item[1], reverse=True)[:num_nodes]
    elif metric == "betweeness":
        most_central = sorted(nx.betweenness_centrality(g).items(), key=lambda item: item[1], reverse=True)[:num_nodes]
    elif metric == "closeness":
        most_central = sorted(nx.closeness_centrality(g).items(), key=lambda item: item[1], reverse=True)[:num_nodes]
    elif metric == "eigenvector":
        most_central = sorted(nx.eigenvector_centrality(g).items(), key=lambda item: item[1], reverse=True)[:num_nodes]
    elif metric == "page_rank":
        most_central = sorted(nx.pagerank(g).items(), key=lambda item: item[1], reverse=True)[:num_nodes]
    
    return most_central
    # ----------------- END OF FUNCTION --------------------- #


def find_cliques(g: nx.Graph, min_size_clique: int) -> tuple:
    """
    Find cliques in the graph g with size at least min_size_clique.

    :param g: networkx graph.
    :param min_size_clique: minimum size of the cliques to find.
    :return: two-element tuple, list of cliques (each clique is a list of nodes) and
        list of nodes in any of the cliques.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    cliques = [clic for clic in nx.find_cliques(g) if len(clic) > min_size_clique]
    clique_nodes = list(set(node for clique in cliques for node in clique))

    return cliques, clique_nodes
    # ----------------- END OF FUNCTION --------------------- #


def detect_communities(g: nx.Graph, method: str) -> tuple:
    """
    Detect communities in the graph g using the specified method.

    :param g: a networkx graph.
    :param method: string with the name of the method to use. Can be (at least) 'givarn-newman' or 'louvain'.
    :return: two-element tuple, list of communities (each community is a list of nodes) and modularity of the partition.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    if method == "girvan_newman":
        communities = girvan_newman(g)
    
    elif method == "louvain":
        communities = louvain_communities(g)

    modularity_partition = modularity(g, communities)

    return communities, modularity_partition

    # ----------------- END OF FUNCTION --------------------- #


if __name__ == '__main__':
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    """
    All the functions in these file are used in the Report.ipynb file
    """
    pass
    # ------------------- END OF MAIN ------------------------ #
