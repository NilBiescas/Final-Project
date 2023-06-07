import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import networkx as nx
import pandas as pd


# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
CLIENT_ID     = "6b32ea69e8d8402889219cc1e430ea87"
CLIENT_SECRET = "ee17f7d8e0f249eab6ede4181c92086f"

auth_manager = SpotifyClientCredentials (
            client_id     = CLIENT_ID ,
            client_secret = CLIENT_SECRET )

sp = spotipy.Spotify( auth_manager = auth_manager )

def BFS(artist_id, graph, Queue):
    for friend_artist in (sp.artist_related_artists(artist_id)["artists"]):
        graph.add_edge(artist_id, friend_artist["id"])
        if friend_artist["id"] not in Queue:
            Queue.append(friend_artist["id"])
    return graph, Queue

def DFS(artist_id, graph, stack, nodes_visitats):
    friends_node = []
    for friend_artist in (sp.artist_related_artists(artist_id)["artists"]):
        graph.add_edge(artist_id, friend_artist["id"])
        if (friend_artist["id"]) not in nodes_visitats:
            friends_node.append(friend_artist["id"])
    stack = friends_node + stack
    return graph, stack

def add_properties(graph):
    nodes_data = {}
    for node_id in graph.nodes():
        artist = sp.artist(node_id)
        properties = {
            'name': artist['name'],
            'id': artist['id'],
            'followers': artist['followers']['total'],
            'popularity': artist['popularity'],
            'genres': str(artist['genres'])
        }
        nodes_data[node_id] = properties

    nx.set_node_attributes(graph, nodes_data)
    return graph

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #


def search_artist(artist_name: str) -> str:
    """
    Search for an artist in Spotify.

    :param artist_name: name to search for.
    :return: spotify artist id.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    results = sp.search(q=artist_name, type='artist')
    if results['artists']["total"] != 0:
        artist = results["artists"]["items"][0]
        artist_id = artist["id"]
        return artist_id
    return None
    # ----------------- END OF FUNCTION --------------------- #


def crawler(seed: str, max_nodes_to_crawl: int, strategy: str = "BFS", out_filename: str = "g.graphml") -> nx.DiGraph:
    """
    Crawl the Spotify artist graph, following related artists.

    :param seed: starting artist id.
    :param max_nodes_to_crawl: maximum number of nodes to crawl.
    :param strategy: BFS or DFS.
    :param out_filename: name of the graphml output file.
    :return: networkx directed graph.

    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    last_crawled = None
    graph = nx.DiGraph()
    crawled_nodes = 0
    if strategy == "BFS":
        Queue = [seed]
        while crawled_nodes < max_nodes_to_crawl:
            graph, Queue  = BFS(Queue[crawled_nodes], graph, Queue)
            crawled_nodes = crawled_nodes + 1
        last_crawled = Queue[crawled_nodes]
        
    elif strategy == "DFS":
        Stack = [seed]
        nodes_visitats = set()
        while (crawled_nodes < max_nodes_to_crawl) and len(Stack) != 0:
            node = Stack.pop(0)    
            nodes_visitats.add(node)      
            graph, Stack  = DFS(node, graph, Stack, nodes_visitats)
            crawled_nodes = crawled_nodes + 1
        last_crawled = node
        
    graph = add_properties(graph)
    nx.write_graphml_lxml(graph, out_filename)
    return graph, last_crawled
    # ----------------- END OF FUNCTION --------------------- #


def get_track_data(graphs: list, out_filename: str) -> pd.DataFrame:
    """
    Get track data for each visited artist in the graph.

    :param graphs: a list of graphs with artists as nodes.
    :param out_filename: name of the csv output file.
    :return: pandas dataframe with track data.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    data = {}

    total_artist = {artist for graph in graphs 
                    for artist in graph.nodes if graph.out_degree(artist) > 0}
    for artist in total_artist:
        
        top_tracks = sp.artist_top_tracks(artist, country='ES')

        for track in top_tracks["tracks"]:

            audio = sp.audio_features(track["id"])

            song_data = {"id": track["id"], 
                         "duration_ms": track["duration_ms"], 
                         "name": track["name"], 
                         "popularity": track["popularity"]}

            audio_feature = {"danceability": audio[0]["danceability"], 
                             "energy": audio[0]["energy"], 
                             "loudness": audio[0]["loudness"], 
                             "speechiness": audio[0]["speechiness"], 
                             "acousticness": audio[0]["acousticness"], 
                             "instrumentalness": audio[0]["instrumentalness"], 
                             "liveness": audio[0]["liveness"], 
                             "valence": audio[0]["valence"], 
                            "tempo": audio[0]["tempo"]
                            }
            
            albums = {"id": track["album"]["id"], 
                      "name": track["album"]["name"], 
                      "release_date": track["album"]["release_date"]
                      }
            
            artists = {"id": artist, 
                       "name": sp.artist(artist)["name"]
                       }
            
            data[track["id"]] = {"song_data": song_data, "audio_feature": audio_feature, "albums": albums, "artists": artists}
        
    Data = pd.DataFrame.from_dict(data, orient = "index")
    Data.to_csv(out_filename)

    return Data
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #

    #Artist_id  = search_artist("Drake", sp)
    #Graph_B, _ = crawler(Artist_id, 200, "BFS", "Graph_B.graphml")
    #Graph_D, _ = crawler(Artist_id, 200, "DFS", "Graph_D.graphml")
    #
    #D_dataframe = get_track_data([Graph_B, Graph_D], 'D.csv') 
    #
    #Artist_id = search_artist("French Montana", sp)
    #Graph_H, Last_Crawled_id   = crawler(Artist_id, 200, "BFS", "Graph_H.graphml")
    #Graph_fb = crawler(Last_Crawled_id, 200, strategy='BFS', out_filename = "Graph_fb.graphml")

    DataFrame                = pd.read_csv('/Users/nbiescas/Desktop/Graphs/Graphs_data/D.csv', index_col="song_id")
    number_songs  = len(DataFrame.index)
    number_artist = len(DataFrame.artists.unique())
    number_albums = len(DataFrame.albums.unique())
    pass
    # ------------------- END OF MAIN ------------------------ #
