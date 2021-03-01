import pandas as pd
import networkx as nx
import numpy as np
import os
'''
code main goal: make a graph with labels and make a knowledge-graph to the classes.
~_~_~ Graph ~_~_~
Graph nodes: movies
Graph edges: given 2 movies, an edge determined if a cast member play in both of the movies.
Label: the genre of the movie. We treat multi genre as different label. For example: Drama-Comedy and Action-Comedy
treat as different labels.
~_~_~ Knowledge-Graph ~_~_~
Knowledge-Graph nodes: classes that represented by genres types.
Knowledge-Graph edges: Jaccard similarity, which means Intersection over Union, donate weight edges between the classes. 
                       For example: Drama-Comedy and Action-Comedy interception is Comedy (donate 1)
                       The union is Drama, Action, Comedy (donate 3)
                       Thus, there is an edge with 1/3 weight between those classes.
'''


class DataCsvToGraph(object):
    """
    Class that read and clean the data
    For IMDb data set we download 2 csv file
    IMDb movies.csv includes 81273 movies with attributes: title, year, genre , etc.
    IMDb title_principles.csv includes 38800 movies and 175715 cast names that play among the movies.
    """
    def __init__(self, data_paths):
        self.data_paths = data_paths

    @staticmethod
    def drop_columns(df, arr):
        for column in arr:
            df = df.drop(column, axis=1)
        return df

    def clean_data_cast(self: None) -> object:
        """
        Clean 'IMDb title_principals.csv' data.
        :return: Data-Frame with cast ('imdb_name_id') and the movies ('imdb_title_id') they play.
        """
        data = pd.read_csv(self.data_paths['cast'])
        clean_column = ['ordering', 'category', 'job', 'characters']
        data = self.drop_columns(data, clean_column)
        data = data.sort_values('imdb_name_id')
        data = pd.DataFrame.dropna(data)
        return data

    def clean_data_genre(self):
        """
        Clean 'IMDb movies.csv' data.
        :return: Data-Frame with movies ('imdb_title_id') and their genre as label ('genre')
        """
        data = pd.read_csv(self.data_paths['genre'])
        clean_columns = list(data.columns)
        clean_columns.remove('imdb_title_id')
        clean_columns.remove('genre')
        for column in clean_columns:
            data = data.drop(column, axis=1)
        data = data.sort_values('imdb_title_id')
        data = pd.DataFrame.dropna(data)
        return data

    def bi_graph(self):
        """
        Build bipartite graph.
        Nodes: movies and cast members.
        Edges: if a cast member play in movie there is an edge between them
        :return: Bipartite Graph
        """
        graph_data = self.clean_data_cast()
        gnx = nx.from_pandas_edgelist(graph_data, source='imdb_title_id', target='imdb_name_id')
        gnx = gnx.to_undirected()
        return gnx


class MoviesGraph(DataCsvToGraph):
    """
    class that inherit from 'DataCsvToGraph' class.
    The goal is to make the final graph of movies as we mention in the code main goal above.
    Also we associate every node to his label.
    """
    def __init__(self, data_paths):
        super().__init__(data_paths)
        self.is_self_second_neighbors = False

    @staticmethod
    def create_dict_neighbors(bi_gnx):
        """
        create a dictionary when key==node and value==set_of_neighbors
        """
        nodes = list(bi_gnx.nodes())
        neighbors_dict = {}
        for i in range(len(nodes)):
            node = nodes[i]
            neighbors_dict.update({node: set(bi_gnx[node])})
        return neighbors_dict

    def create_dict_second_neighbors(self, bi_gnx):
        """
        create a dictionary when key==node and value==set_of_two order_neighbors
        """
        dict_neighbors = self.create_dict_neighbors(bi_gnx)
        nodes = list(bi_gnx.nodes())
        second_neighbors_dict = {}
        for i in range(len(nodes)):
            node = nodes[i]
            first_neighbors_node_i = dict_neighbors[node]
            if len(first_neighbors_node_i) > 0:
                second_neighbors_i = set([])
                for j in range(len(first_neighbors_node_i)):
                    second_neighbor = dict_neighbors[list(first_neighbors_node_i)[j]]
                    second_neighbors_i.update(second_neighbor)
                if not self.is_self_second_neighbors:
                    if node in second_neighbors_i:
                        second_neighbors_i.remove(node)
                second_neighbors_dict.update({node: second_neighbors_i})
            else:
                second_neighbors_dict.update({node: set([])})
        return second_neighbors_dict

    def create_graph(self):
        """
        create the movies graph from the bipartite graph.
        make an edge between every second neighbors of every node except from self edges.
        Thus we have an edge between movies if a cast member play in both and unnecessary edges between cast members
        if they participate in the same movie.
        We then remove all the cast nodes and all their adjacent edges.
        :return: The final movies graph.
        """
        if os.path.exists('pkl/IMDb_graph.gpickle'):
            G = nx.read_gpickle('pkl/IMDb_graph.gpickle')
        else:
            G = self.bi_graph()
            nodes = list(G.nodes())
            dict_second_neighbors = self.create_dict_second_neighbors(G)
            for node in nodes:
                if len(dict_second_neighbors[node]) > 0:
                    for second_neighbor in dict_second_neighbors[node]:
                        G.add_edge(node, second_neighbor)
            bi_graph_nodes_df = self.clean_data_cast()
            bi_graph_nodes_df = bi_graph_nodes_df.drop('imdb_title_id', axis=1)
            bi_graph_nodes_df = bi_graph_nodes_df.drop_duplicates()
            unnecessary_nodes = bi_graph_nodes_df['imdb_name_id']
            for node in unnecessary_nodes:
                G.remove_node(node)
            nx.write_gpickle(G, 'pkl/IMDb_graph.gpickle')
        return G

    def create_basic_labels(self, final_gnx):
        """
        create relevant labels for the nodes in the final graph. the labels still in the original interface.
        There are 20 different labels (genres), e.g. Drama, Comedy...
        :param final_gnx:
        :return: relevant labels
        """
        if os.path.exists('pkl/labels_data.pkl'):
            labels_data = pd.read_pickle('pkl/labels_data.pkl')
        else:
            df_all_labels = self.clean_data_genre()
            nodes = final_gnx.nodes()
            labels_data = pd.DataFrame()
            for node in nodes:
                mask = (df_all_labels.imdb_title_id == node)
                labels_data = labels_data.append(df_all_labels[mask])
            labels_data = labels_data.sort_values('imdb_title_id')
            labels_data.to_pickle('pkl/labels_data.pkl')
        labels_data = labels_data.reset_index(drop=True)
        return labels_data

    def labels2int(self, final_gnx):
        """
        add to the labels data, 2 important shapes:
        array_label: each array_label is a vector that consists of the 20 genres each genre represented as
        number between 1 to 20 e.g. drama = [0]; comedy = [3]; drama, comedy = [0,3]
        int_label: represent every different combination of genres as different label.
        :param final_gnx:
        :return: array_label, int_label
        """
        if os.path.exists('pkl/final_labels_data.pkl'):
            final_labels_data = pd.read_pickle('pkl/final_labels_data.pkl')
        else:
            final_labels_data = self.create_basic_labels(final_gnx)
            new_column = np.zeros(len(final_labels_data['genre']))
            final_labels_data['array_label'] = new_column
            final_labels_data['int_label'] = new_column
            dict_label2array = {}
            genres = final_labels_data['genre']
            num_label = 0
            label_index = 0
            new_ind = True
            unique_labels = np.array([]).astype(int)
            for i in range(len(genres)):
                labels = genres[i].split(", ")
                array_int_labels = np.array([])
                array_int_labels = array_int_labels.astype(int)
                for label in labels:
                    if dict_label2array.get(label) is None:
                        dict_label2array.update({label: num_label})
                        num_label += 1
                    array_int_labels = np.append(array_int_labels, dict_label2array[label])
                array_int_labels = np.sort(array_int_labels)
                s = str(array_int_labels[0])
                if len(array_int_labels) > 0:
                    for j in range(len(array_int_labels)):
                        if j > 0:
                            s = ",".join((s, str(array_int_labels[j])))
                final_labels_data.loc[i, 'array_label'] = s
                for k in unique_labels:
                    compare_label = np.array(final_labels_data['array_label'][k].split(",")).astype(int)
                    if len(compare_label) != len(array_int_labels):
                        new_ind = True
                    elif np.equal(compare_label, array_int_labels).all():
                        final_labels_data.loc[i, 'int_label'] = str(final_labels_data['int_label'][k])
                        new_ind = False
                        break
                if new_ind:
                    final_labels_data.loc[i, 'int_label'] = str(label_index)
                    unique_labels = np.append(unique_labels, i)
                    label_index += 1
            final_labels_data.to_pickle('pkl/final_labels_data.pkl')
        return final_labels_data

    @staticmethod
    def compute_jaccard_similarity_score(x, y):
        """
        Jaccard Similarity J (A,B) = | Intersection (A,B) | /
                                        | Union (A,B) |
        """
        intersection_cardinality = len(set(x).intersection(set(y)))
        union_cardinality = len(set(x).union(set(y)))
        return intersection_cardinality / float(union_cardinality)

    def create_knowledge_graph(self, final_labels_data):
        """
        create the knowledge graph from the classes.
        we use jaccard similarity score for every 2 nodes based on their array_label.
        For example: drama comedy and drama crime will have a weighted edge and the value will be
        the jaccard similarity.
        :param final_labels_data:
        :return: knowledge_graph
        """
        if os.path.exists('pkl/IMDb_knowledge_graph.gpickle'):
            knowledge_graph = nx.read_gpickle('pkl/IMDb_knowledge_graph.gpickle')
        else:
            clean_column = ['genre', 'imdb_title_id']
            final_labels_data = self.drop_columns(final_labels_data, clean_column)
            final_labels_data = final_labels_data.drop_duplicates('int_label')
            final_labels_data = final_labels_data.reset_index(drop=True)
            nodes = final_labels_data['int_label']
            knowledge_graph = nx.Graph()
            knowledge_graph.add_nodes_from(nodes)
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if i != j:
                        compare_label_i = np.array(final_labels_data['array_label'][i].split(",")).astype(int)
                        compare_label_j = np.array(final_labels_data['array_label'][j].split(",")).astype(int)
                        jaccard_similarity = self.compute_jaccard_similarity_score(compare_label_i, compare_label_j)
                        if jaccard_similarity > 0.3:
                            edge = [(nodes[i], nodes[j], jaccard_similarity)]
                            knowledge_graph.add_weighted_edges_from(edge)
            nx.write_gpickle(knowledge_graph, 'pkl/IMDb_knowledge_graph.gpickle')
        return knowledge_graph

    def create_labels_graph(self, final_labels_data):
        """
        create a graph of connections between every node and his class.
        :param final_labels_data:
        :return: labels_graph
        """
        if os.path.exists('pkl/IMDb_labels_graph.gpickle'):
            labels_graph = nx.read_gpickle('pkl/IMDb_labels_graph.gpickle')
        else:
            clean_column = ['genre', 'array_label']
            final_labels_data = self.drop_columns(final_labels_data, clean_column)
            labels_graph = nx.from_pandas_edgelist(final_labels_data, source='imdb_title_id', target='int_label')
            labels_graph = labels_graph.to_undirected()
            nx.write_gpickle(labels_graph, 'pkl/IMDb_labels_graph.gpickle')
        return labels_graph

    def create_multi_graph(self, final_gnx, knowledge_graph, final_labels_data):
        """
        unite 3 graphs: final_graph, knowledge_graph, labels_graph into 1 multi-graph.
        :param final_gnx:
        :param knowledge_graph:
        :param final_labels_data:
        :return: multi_graph
        """
        if os.path.exists('pkl/IMDb_multi_graph1.gpickle'):
            multi_graph = nx.read_gpickle('pkl/IMDb_multi_graph.gpickle')
        else:
            multi_graph = nx.MultiGraph()
            labels_graph = self.create_labels_graph(final_labels_data)
            labels_edges = labels_graph.edges
            movies_edges = final_gnx.edges
            classes_edges = knowledge_graph.edges
            movies_nodes = final_gnx.nodes
            classes_nodes = knowledge_graph.nodes
            multi_graph.add_nodes_from(movies_nodes, key='movies')
            multi_graph.add_nodes_from(classes_nodes, key='classes')
            multi_graph.add_edges_from(movies_edges, key='movies_edges')
            multi_graph.add_edges_from(classes_edges, key='classes_edges')
            multi_graph.add_edges_from(labels_edges, key='labels_edges')
            for edge in classes_edges:
                dict_weight = knowledge_graph.get_edge_data(edge[0], edge[1])
                weight = dict_weight.get('weight')
                if weight is not None:
                    multi_graph[edge[0]][edge[1]][0]['weight'] = weight
            nx.write_gpickle(multi_graph, 'pkl/IMDb_multi_graph1.gpickle')
        return multi_graph


def main():
    dict_paths = {'cast': 'data_set/IMDb title_principals.csv', 'genre': 'data_set/IMDb movies.csv'}
    IMDb = MoviesGraph(dict_paths)
    gnx = IMDb.create_graph()
    labels = IMDb.labels2int(gnx)
    print(labels.head())
    print(labels.shape)
    knowledge_gnx = IMDb.create_knowledge_graph(labels)
    multi_gnx = IMDb.create_multi_graph(gnx, knowledge_gnx, labels)
    print(len(multi_gnx.edges))
    print(len(multi_gnx.nodes))
    print(len(gnx.edges))
    print(len(gnx.nodes))
    print(len(knowledge_gnx.nodes))
    print(len(knowledge_gnx.edges))
    return multi_gnx


main()

