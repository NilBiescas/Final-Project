import networkx as nx
import pandas as pd
from collections import defaultdict
import numpy as np

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
def cosine_similiarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def euclidean_similiarity(vector1, vector2):
    return 1 / (1 + np.linalg.norm(vector1 - vector2))

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    undirected_graph = nx.Graph()

    bidirectional_edges = []    #List of tuples. Each tuple represents and edge
    nodes_attributes    = {}    #Dictionary to store all the information of the nodes
    # Iterate over the edges of the directed graph
    for u, v in g.edges():       
        if g.has_edge(v, u):  # Check if both edges (v, u) and (u, v) exist
            bidirectional_edges.append((u, v))      #Store the nodes that have a bidirectional edge
            nodes_attributes[v] = g.nodes()[v]      #And their attributes
            nodes_attributes[u] = g.nodes()[u]
            
    undirected_graph.add_edges_from(bidirectional_edges)    #Add all the new nodes, with the edges
    nx.set_node_attributes(undirected_graph, nodes_attributes)  #Add the nodes attributes retrive earlier.
    nx.write_graphml_lxml(undirected_graph, out_filename)  # Write the undirected graph to a GraphML file
    return undirected_graph  # Return the undirected graph
    # ----------------- END OF FUNCTION --------------------- #

def prune_low_degree_nodes(g: nx.Graph, min_degree: int, out_filename: str) -> nx.Graph:
    """
    Prune a graph by removing nodes with degree < min_degree.

    :param g: a networkx graph.
    :param min_degree: lower bound value for the degree.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #

    # Create a list of nodes to remove based on their degree being less than min_degree
    remove_nodes = [id for id, degree in g.degree() if degree < min_degree]
    
    # Remove the nodes from the graph
    g.remove_nodes_from(remove_nodes)

    # Create a list of nodes with zero degree
    zero_degree_nodes = [id for id, degree in g.degree() if degree == 0]
    
    # Remove the nodes with zero degree from the graph
    g.remove_nodes_from(zero_degree_nodes)

    # ----------------- END OF FUNCTION --------------------- #
    nx.write_graphml_lxml(g, out_filename)  # Write the pruned graph to a GraphML file
    return g  # Return the pruned graph


def prune_low_weight_edges(g: nx.Graph, min_weight=None, min_percentile=None, out_filename: str = None) -> nx.Graph:
    """
    Prune a graph by removing edges with weight < threshold. Threshold can be specified as a value or as a percentile.
    :param g: a weighted networkx graph.
    :param min_weight: lower bound value for the weight.
    :param min_percentile: lower bound percentile for the weight.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    if ((min_weight == None) and (min_percentile == None)) or ((min_weight != None) and (min_percentile != None)):
        raise Exception  # Raise an exception if both min_weight and min_percentile are None or if both are not None.
    
    if min_percentile != None:
        sorted_weights = sorted([data["weight"] for _, _, data in g.edges(data=True)])  # Get a sorted list of edge weights
        pos = int((min_percentile * 100) * len(sorted_weights) - 1)  # Calculate the position based on percentile
        min_weight = sorted_weights[pos]  # Set the min_weight based on the calculated position in the sorted list

    # Find edges with weight less than min_weight and remove them from the graph
    remove_edges = [(u, v) for u, v, data in g.edges(data=True) if data["weight"] < min_weight]
    g.remove_edges_from(remove_edges)

    # Find nodes with zero degree and remove them from the graph
    zero_degree_nodes = [id for id, degree in g.degree() if degree == 0]
    g.remove_nodes_from(zero_degree_nodes)

    # Write the pruned graph to a file if out_filename is specified

    # Uncomment the line below to save the pruned graph as a graphml file
    nx.write_graphml_lxml(g, out_filename)
    return g
# ----------------- END OF FUNCTION --------------------- #


def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    mean_audio_features = {}  # Dictionary to store mean audio features for each artist

    # Convert string representations of dictionaries to actual dictionaries
    tracks_df["audio_feature"] = tracks_df["audio_feature"].apply(eval)
    tracks_df["song_data"] = tracks_df["song_data"].apply(eval)
    tracks_df["artists"] = tracks_df["artists"].apply(eval)
    tracks_df['albums'] = tracks_df['albums'].apply(eval)

    # Get unique artist names from the dataframe
    artist_names = tracks_df['artists'].apply(lambda x: x.get('name')).unique()

    for artist_name in artist_names:
        filtered_df = tracks_df[tracks_df['artists'].apply(lambda x: x.get('name') == artist_name)]
        artist_id = filtered_df["artists"].iat[0]['id']  # Get the artist ID from the first row

        grouped_audio_features = defaultdict(int)  # Defaultdict to store aggregated audio features
        num_songs = len(filtered_df)  # Number of songs for the artist

        for entry in filtered_df["audio_feature"].values:
            for key in entry:
                grouped_audio_features[key] += entry[key]  # Aggregate audio features

        # Calculate mean audio features
        mean_audio = {key: round(grouped_audio_features[key] / num_songs, 2) for key in grouped_audio_features}
        mean_audio_features[artist_id] = {"artist_id":artist_id, 
                                          "artist_name": artist_name,                              
                                          "danceability": mean_audio['danceability'],
                                          "energy": mean_audio['energy'],
                                          "loudness": mean_audio['loudness'],
                                          "speechiness": mean_audio['speechiness'],
                                          "acousticness": mean_audio['acousticness'],
                                          "instrumentalness":mean_audio['instrumentalness'],
                                          "liveness": mean_audio['liveness'],
                                          "valence": mean_audio['valence'],
                                          "tempo": mean_audio['tempo']
                                          }

    df = pd.DataFrame.from_dict(mean_audio_features, orient="index")
    return df
    # ----------------- END OF FUNCTION --------------------- #


def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> \
        nx.Graph:
    
    """
    Create a similarity graph from a dataframe with mean audio features per artist.

    :param artist_audio_features_df: dataframe with mean audio features per artist.
    :param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
    :param out_filename: name of the file that will be saved.
    :return: a networkx graph with the similarity between artists as edge weights.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    audio_features = ['danceability', 'energy', 'loudness',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo']
    
    edges_weights = []
    audio_features_mean = artist_audio_features_df[audio_features]
    artist_id = artist_audio_features_df['artist_id']

    for u in artist_id:
        vector1 = audio_features_mean.loc[u].values
        for v in artist_id:
            if v == u:
                continue
            vector2 = audio_features_mean.loc[v].values
            if similarity == 'cosine':
                similarity = cosine_similiarity(vector1, vector2)
            else:
                similarity = euclidean_similiarity(vector1, vector2)
            edges_weights.append((u, v, {"weight":similarity}))

    # Create an empty graph
    graph = nx.Graph()
    # Add nodes and weighted edges to the graph
    graph.add_edges_from(edges_weights)

    nx.write_graphml_lxml(graph, out_filename)
    return graph
    # ----------------- END OF FUNCTION --------------------- #




if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #

    # Read Graphs B and Graph D

    path_graph_B = '/Users/nbiescas/Desktop/Graphs/Graphs_data/Graph_B.graphml'
    path_graph_D = '/Users/nbiescas/Desktop/Graphs/Graphs_data/Graph_D.graphml'
    Graph_B = nx.read_graphml(path_graph_B)
    Graph_D = nx.read_graphml(path_graph_D)


    #Obtantion of the undirected graphs

    Undirected_graph_B = retrieve_bidirectional_edges(Graph_B, "Undirected_graph_B.graphml")
    Undirected_graph_D = retrieve_bidirectional_edges(Graph_D, "Undirected_graph_D.graphml")

    # Obtain the two dataframes of each graph
    Pandas_Graph_B = pd.read_csv('/Users/nbiescas/Desktop/Graphs/Graphs_data/Pandas_Graph_B.csv', index_col="song_id")
    Pandas_Graph_D = pd.read_csv('/Users/nbiescas/Desktop/Graphs/Graphs_data/Pandas_Graph_D.csv', index_col="song_id")

    # Compute the mean audio features for each graph
    artist_audio_features_graph_B = compute_mean_audio_features(Pandas_Graph_B)
    artist_audio_features_graph_D = compute_mean_audio_features(Pandas_Graph_D)

    # Obtain the similiarity graphs for each case the B and D
    CompleteGraph_B           = create_similarity_graph(artist_audio_features_graph_B, similarity = 'cosine', out_filename = "CompleteGraph_B.graphml") 
    CompleteGraph_D           = create_similarity_graph(artist_audio_features_graph_D, similarity = 'cosine', out_filename = "CompleteGraph_D.graphml") 

    # Finally obtain the weighted graphs
    Undirected_graph_B_weights = prune_low_weight_edges(CompleteGraph_B, min_weight = 0.45, out_filename='Weighted_Graph_B.graphml')
    Undirected_graph_D_weights = prune_low_weight_edges(CompleteGraph_D, min_weight = 0.31, out_filename='Weighted_Graph_D.graphml')

    # ------------------- END OF MAIN ------------------------ #
