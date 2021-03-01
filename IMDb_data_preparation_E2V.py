import pandas as pd
import networkx as nx
import numpy as np
import os
import random
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
        if os.path.exists('pkl_e2v/data_cast_movie.pkl'):
            data = pd.read_csv(self.data_paths['cast'])
            clean_column = ['ordering', 'category', 'job', 'characters']
            data = self.drop_columns(data, clean_column)
            data = data.sort_values('imdb_name_id')
            data = pd.DataFrame.dropna(data)
            keys = data
            keys = keys.drop('imdb_name_id', axis=1)
            data = pd.read_pickle('pkl_e2v/data_cast_movie.pkl')
            data['tmp'] = keys['imdb_title_id']
        else:
            data = pd.read_csv(self.data_paths['cast'])
            clean_column = ['ordering', 'category', 'job', 'characters']
            data = self.drop_columns(data, clean_column)
            data = data.sort_values('imdb_name_id')
            data = pd.DataFrame.dropna(data)
            keys = data.drop_duplicates('imdb_title_id')
            keys = keys.drop('imdb_name_id', axis=1)
            keys = keys.to_dict('list')
            keys = keys['imdb_title_id']
            for i in range(len(keys)):
                name = 't' + str(i)
                cond = data != keys[i]
                data = data.where(cond, name)
            data.to_pickle('pkl_e2v/data_cast_movie.pkl')
            data = pd.read_csv(self.data_paths['cast'])
            clean_column = ['ordering', 'category', 'job', 'characters']
            data = self.drop_columns(data, clean_column)
            data = data.sort_values('imdb_name_id')
            data = pd.DataFrame.dropna(data)
            keys = data
            keys = keys.drop('imdb_name_id', axis=1)
            data = pd.read_pickle('pkl_e2v/data_cast_movie.pkl')
            data['tmp'] = keys['imdb_title_id']
        return data

    def clean_data_genre(self):
        """
        Clean 'IMDb movies.csv' data.
        :return: Data-Frame with movies ('imdb_title_id') and their genre as label ('genre')
        """
        data = pd.read_csv(self.data_paths['genre'])
        renames = self.clean_data_cast()
        renames = renames.drop('imdb_name_id', axis=1)
        renames = renames.drop_duplicates('imdb_title_id')
        renames = renames.reset_index(drop=True)
        original = renames.to_dict('index')
        dict_translate_original_name = {}
        for i in range(len(original)):
            dict_translate_original_name[original[i]['tmp']] = original[i]['imdb_title_id']
        for index, row in data.iterrows():
            if dict_translate_original_name.get(data['imdb_title_id'][index]):
                data.loc[index, 'imdb_title_id'] = dict_translate_original_name[data['imdb_title_id'][index]]
            # else:
            #     data.drop(data.index[index])
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
    def __init__(self, data_paths, movies_nodes_precentage):
        super().__init__(data_paths)
        self.is_self_second_neighbors = False
        self.movies_nodes_precentage = movies_nodes_precentage

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
        if os.path.exists('pkl_e2v/IMDb_graph.gpickle'):
            G = nx.read_gpickle('pkl_e2v/IMDb_graph.gpickle')
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
            bi_graph_nodes_df = bi_graph_nodes_df.drop('tmp', axis=1)
            bi_graph_nodes_df = bi_graph_nodes_df.drop_duplicates()
            unnecessary_nodes = bi_graph_nodes_df['imdb_name_id']
            for node in unnecessary_nodes:
                G.remove_node(node)
            nx.write_gpickle(G, 'pkl_e2v/IMDb_graph.gpickle')
        return G

    def create_basic_labels(self, final_gnx):
        """
        create relevant labels for the nodes in the final graph. the labels still in the original interface.
        There are 20 different labels (genres), e.g. Drama, Comedy...
        :param final_gnx:
        :return: relevant labels
        """
        if os.path.exists('pkl_e2v/labels_data.pkl'):
            labels_data = pd.read_pickle('pkl_e2v/labels_data.pkl')
        else:
            df_all_labels = self.clean_data_genre()
            nodes = final_gnx.nodes()
            labels_data = pd.DataFrame()
            for node in nodes:
                mask = (df_all_labels.imdb_title_id == node)
                labels_data = labels_data.append(df_all_labels[mask])
            labels_data = labels_data.sort_values('imdb_title_id')
            labels_data.to_pickle('pkl_e2v/labels_data.pkl')
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
        if os.path.exists('pkl_e2v/final_labels_data.pkl'):
            final_labels_data = pd.read_pickle('pkl_e2v/final_labels_data.pkl')
        else:
            final_labels_data = self.create_basic_labels(final_gnx)
            new_column = np.zeros(len(final_labels_data['genre']))
            final_labels_data['array_label'] = new_column
            final_labels_data['int_label'] = new_column
            final_labels_data['name_label'] = new_column
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
                        final_labels_data.loc[i, 'name_label'] = 'c' + str(final_labels_data['int_label'][k])
                        new_ind = False
                        break
                if new_ind:
                    final_labels_data.loc[i, 'int_label'] = str(label_index)
                    final_labels_data.loc[i, 'name_label'] = 'c' + str(label_index)
                    unique_labels = np.append(unique_labels, i)
                    label_index += 1
            final_labels_data.to_pickle('pkl_e2v/final_labels_data.pkl')
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

    def create_knowledge_graph(self, final_labels_data, threshold):
        """
        create the knowledge graph from the classes.
        we use jaccard similarity score for every 2 nodes based on their array_label.
        For example: drama comedy and drama crime will have a weighted edge and the value will be
        the jaccard similarity.
        :param final_labels_data:
        :return: knowledge_graph
        """
        if os.path.exists('pkl_e2v/IMDb_knowledge_graph.gpickle') and os.path.exists('pkl_e2v/IMDb_knowledge_data.pkl'):
            knowledge_graph = nx.read_gpickle('pkl_e2v/IMDb_knowledge_graph.gpickle')
            knowledge_data = pd.read_pickle('pkl_e2v/IMDb_knowledge_data.pkl')
        else:
            clean_column = ['genre', 'imdb_title_id']
            final_labels_data = self.drop_columns(final_labels_data, clean_column)
            final_labels_data = final_labels_data.drop_duplicates('name_label')
            final_labels_data = final_labels_data.reset_index(drop=True)
            nodes = final_labels_data['name_label']
            # knowledge_data = nodes
            # knowledge_data['name_label2'] = np.zeros(len(knowledge_data['name_label']))
            # knowledge_data.append()
            knowledge_data = pd.DataFrame(columns=['node1', 'node2', 'weight'])
            knowledge_graph = nx.Graph()
            knowledge_graph.add_nodes_from(nodes)
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if i != j:
                        compare_label_i = np.array(final_labels_data['array_label'][i].split(",")).astype(int)
                        compare_label_j = np.array(final_labels_data['array_label'][j].split(",")).astype(int)
                        jaccard_similarity = self.compute_jaccard_similarity_score(compare_label_i, compare_label_j)
                        if jaccard_similarity > threshold:
                            edge = [(nodes[i], nodes[j], jaccard_similarity)]
                            knowledge_graph.add_weighted_edges_from(edge)
                            knowledge_data.loc[len(knowledge_data)] = [nodes[i], nodes[j], jaccard_similarity]
            nx.write_gpickle(knowledge_graph, 'pkl_e2v/IMDb_knowledge_graph.gpickle')
            pd.to_pickle(knowledge_data, 'pkl_e2v/IMDb_knowledge_data.pkl')
        return knowledge_graph, knowledge_data

    def create_labels_graph(self, final_labels_data):
        """
        create a graph of connections between every node and his class.
        :param final_labels_data:
        :return: labels_graph
        """
        if os.path.exists('pkl_e2v/IMDb_labels_graph.gpickle'):
            labels_graph = nx.read_gpickle('pkl_e2v/IMDb_labels_graph.gpickle')
        else:
            clean_column = ['genre', 'array_label', 'int_label']
            final_labels_data = self.drop_columns(final_labels_data, clean_column)
            labels_graph = nx.from_pandas_edgelist(final_labels_data, source='imdb_title_id', target='name_label')
            labels_graph = labels_graph.to_undirected()
            nx.write_gpickle(labels_graph, 'pkl_e2v/IMDb_labels_graph.gpickle')
        return labels_graph

    def create_multi_graph(self, final_gnx, knowledge_graph, final_labels_data):
        """
        unite 3 graphs: final_graph, knowledge_graph, labels_graph into 1 multi-graph.
        :param final_gnx:
        :param knowledge_graph:
        :param final_labels_data:
        :return: multi_graph
        """
        if os.path.exists('pkl_e2v/IMDb_multi_graph1.gpickle'):
            multi_graph = nx.read_gpickle('pkl_e2v/IMDb_multi_graph1.gpickle')
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
            classes_edges = np.array(classes_edges)
            movies_edges = np.array(movies_edges)
            labels_edges = np.array(labels_edges)
            edges = np.concatenate((classes_edges, movies_edges), axis=0)
            edges = np.concatenate((edges, labels_edges), axis=0).T
            edges_data = pd.DataFrame()
            edges_data['edge1'] = edges[0]
            edges_data['edge2'] = edges[1]
            edges_data.to_csv('pkl_e2v/edgelist_e2v.txt', header=None, index=None, sep=' ', mode='a')
            nx.write_gpickle(multi_graph, 'pkl_e2v/IMDb_multi_graph1.gpickle')
        return multi_graph

    def mutli_graph_edge_list(self):
        pd.read_pickle('final_labels_data.pkl')

    def weighted_multi_graph(self, final_gnx, knowledge_graph, final_labels_data, weights_dict, ):
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
        movies_weight = weights_dict['movies_edges']
        labels_weight = weights_dict['labels_edges']
        for edge in movies_edges:
            multi_graph[edge[0]][edge[1]][0]['weight'] = movies_weight
        for edge in labels_edges:
            multi_graph[edge[0]][edge[1]][0]['weight'] = labels_weight
        movies_nodes = np.array(movies_nodes)
        random.Random(4).shuffle(movies_nodes)
        if self.movies_nodes_precentage < 1:
            for movie_node in movies_nodes[0:int(len(movies_nodes)*(1 - self.movies_nodes_precentage))]:
                multi_graph.remove_node(movie_node)
        return multi_graph

    def weighted_graph(self, final_gnx, knowledge_graph, final_labels_data, weights_dict):
        weighted_graph = nx.Graph()
        labels_graph = self.create_labels_graph(final_labels_data)
        classes_nodes = knowledge_graph.nodes
        movies_nodes = final_gnx.nodes
        labels_edges = labels_graph.edges
        movies_edges = final_gnx.edges
        classes_edges = knowledge_graph.edges
        weighted_graph.add_nodes_from(movies_nodes, key='movies')
        weighted_graph.add_nodes_from(classes_nodes, key='classes')
        weighted_graph.add_edges_from(movies_edges, key='movies_edges')
        weighted_graph.add_edges_from(classes_edges, key='classes_edges')
        weighted_graph.add_edges_from(labels_edges, key='labels_edges')
        for edge in classes_edges:
            dict_weight = knowledge_graph.get_edge_data(edge[0], edge[1])
            weight = dict_weight.get('weight')
            if weight is not None:
                weighted_graph.add_edge(edge[0], edge[1], weight=weight, key='classes_edges')
        movies_weight = weights_dict['movies_edges']
        labels_weight = weights_dict['labels_edges']
        for edge in movies_edges:
            weighted_graph.add_edge(edge[0], edge[1], weight=movies_weight, key='movies_edges')
        for edge in labels_edges:
            weighted_graph.add_edge(edge[0], edge[1], weight=labels_weight, key='labels_edges')
        movies_nodes = np.array(movies_nodes)
        random.Random(4).shuffle(movies_nodes)
        if self.movies_nodes_precentage < 1:
            for movie_node in movies_nodes[0:int(len(movies_nodes)*(1 - self.movies_nodes_precentage))]:
                weighted_graph.remove_node(movie_node)
        return weighted_graph


def main():
    dict_paths = {'cast': 'data_set/IMDb title_principals.csv', 'genre': 'data_set/IMDb movies.csv'}
    IMDb = MoviesGraph(dict_paths, 1)
    gnx = IMDb.create_graph()
    labels = IMDb.labels2int(gnx)
    print(labels.head())
    print(labels.shape)
    knowledge_gnx, knowledge_data = IMDb.create_knowledge_graph(labels, 0.3)
    multi_gnx = IMDb.create_multi_graph(gnx, knowledge_gnx, labels)
    print(len(multi_gnx.edges))
    print(len(multi_gnx.nodes))
    print(len(gnx.edges))
    print(len(gnx.nodes))
    print(len(knowledge_gnx.nodes))
    print(len(knowledge_gnx.edges))
    return multi_gnx


if __name__ == '__main__':
    main()
