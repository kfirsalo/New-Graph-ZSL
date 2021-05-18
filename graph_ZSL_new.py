from node2vec import Node2Vec
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
import argparse
from numpy import linalg as la
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import model_selection as sk_ms
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from statistics import harmonic_mean
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import chain
from utils import set_gpu
from utlis_graph_zsl import hist_plot, plot_confusion_matrix, plots_2measures_vs_parameter
import torch
from torch.backends import cudnn

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.deterministic = True
random.seed(seed)

class GraphImporter:
    """
    class that responsible to import or create the relevant graph
    """
    def __init__(self, args):
        self.dataset = args.dataset
        self.graph_percentage = args.graph_percentage
        self.threshold = args.threshold
        self.args = args

    def import_imdb_multi_graph(self, weights):
        """
        Make our_imdb multi graph using class
        :param weights:
        :return:
        """
        from IMDb_data_preparation_E2V import MoviesGraph
        weights_dict = {'movies_edges': weights[0], 'labels_edges': weights[1]}
        dict_paths = {'cast': 'data_set/IMDb title_principals.csv', 'genre': 'data_set/IMDb movies.csv'}
        imdb = MoviesGraph(dict_paths, self.args.graph_percentage)
        gnx = imdb.create_graph()
        labels = imdb.labels2int(gnx)
        knowledge_gnx, knowledge_data = imdb.create_knowledge_graph(labels, self.threshold)
        multi_gnx = imdb.weighted_multi_graph(gnx, knowledge_gnx, labels, weights_dict)
        return multi_gnx

    def import_imdb_weighted_graph(self, weights):
        from IMDb_data_preparation_E2V import MoviesGraph
        weights_dict = {'movies_edges': weights[0], 'labels_edges': weights[1]}
        dict_paths = {'cast': 'data_set/IMDb title_principals.csv', 'genre': 'data_set/IMDb movies.csv'}
        imdb = MoviesGraph(dict_paths, self.graph_percentage)
        gnx = imdb.create_graph()
        labels = imdb.labels2int(gnx)
        knowledge_gnx, knowledge_data = imdb.create_knowledge_graph(labels, float(self.threshold))
        weighted_graph = imdb.weighted_graph(gnx, knowledge_gnx, labels, weights_dict)
        return weighted_graph

    def import_graph(self):
        graph = nx.MultiGraph()
        data_path = self.dataset + '.txt'
        path = os.path.join(self.dataset, data_path)
        with open(path, 'r') as f:
            for line in f:
                items = line.strip().split()
                att1 = str(items[0][0])
                att2 = str(items[1][0])
                graph.add_node(items[0], key=att1)
                graph.add_node(items[1], key=att2)
                sort_att = np.array([att1, att2])
                sort_att = sorted(sort_att)
                graph.add_edge(items[0], items[1], key=str(sort_att[0]) + str(sort_att[1]))
        return graph

    def import_data_graph(self, final_graph_weights, specific_split, att_weight):
        from images_graph_creator_all import FinalGraphCreator, ImagesEmbeddings, define_graph_args
        weights_dict = {'classes_edges': final_graph_weights[0], 'labels_edges': final_graph_weights[1]}
        dict_paths, radius = define_graph_args(self.args.dataset)
        graph_preparation = ImagesEmbeddings(dict_paths, self.args)
        embeds_matrix, dict_image_embed, dict_image_class = graph_preparation.images_embed_calculator()
        dict_idx_image_class = {i: dict_image_class[image]
                                for i, image in enumerate(list(dict_image_class.keys()))}
        final_graph_creator = FinalGraphCreator(dict_paths, embeds_matrix, dict_image_embed,
                                                dict_idx_image_class, self.args.images_nodes_percentage, self.args)
        image_graph = final_graph_creator.create_image_graph(radius)
        kg, dict_class_nodes_translation = final_graph_creator.imagenet_knowledge_graph()
        kg = final_graph_creator.attributed_graph(kg, dict_class_nodes_translation, att_weight, radius)
        seen_classes, unseen_classes = final_graph_creator.seen_classes, final_graph_creator.unseen_classes
        seen_classes = [dict_class_nodes_translation[c] for c in seen_classes]
        unseen_classes = [dict_class_nodes_translation[c] for c in unseen_classes]
        split = {'seen': seen_classes, 'unseen': unseen_classes}
        labels_graph = final_graph_creator.create_labels_graph(dict_class_nodes_translation)
        final_graph = final_graph_creator.weighted_graph(image_graph, kg, labels_graph, weights_dict)
        print("image graph edges & nodes: ", len(image_graph.edges), "&", len(image_graph.nodes))
        print("knowledge graph edges & nodes: ", len(kg.edges), "&", len(kg.nodes))
        print("labels graph edges & nodes: ", len(labels_graph.edges), "&", len(labels_graph.nodes))
        print("final graph edges & nodes: ", len(final_graph.edges), "&", len(final_graph.nodes))


        nx.write_gpickle(final_graph, f'{self.args.dataset}/train/{self.args.dataset}_graph')
        if specific_split:
            return final_graph, split
        else:
            split = None
            return final_graph, split

class EmbeddingCreator(object):
    def __init__(self, graph=None, dimension=None, args=None):
        self.dataset = args.dataset
        self.dim = dimension
        self.graph = graph

    def create_node2vec_embeddings(self):
        path1 = os.path.join(self.dataset, 'Node2Vec_embedding.pickle')
        # path2 = os.path.join(self.dataset, 'Node2Vec_embedding.csv')
        if os.path.exists(path1):
            with open(path1, 'rb') as handle:
                dict_embeddings = pickle.load(handle)
        # elif os.path.exists(path2):
        #     embedding_df = pd.read_csv(path2)
        #     dict_embeddings = embedding_df.to_dict(orient='list')
        #     with open(path2, 'wb') as handle:
        #         pickle.dump(dict_embeddings, handle, protocol=3)
        # else:
        #     node2vec = Node2Vec(self.graph, dimensions=16, walk_length=30, num_walks=200, workers=1)
        #     model = node2vec.fit()
        #     nodes = list(self.graph.nodes())
        #     dict_embeddings = {}
        #     for i in range(len(nodes)):
        #         dict_embeddings.update({nodes[i]: np.asarray(model.wv.get_vector(nodes[i]))})
        #     with open(path1, 'wb') as handle:
        #         pickle.dump(dict_embeddings, handle, protocol=3)

        node2vec = Node2Vec(self.graph, dimensions=self.dim, walk_length=80, num_walks=16, workers=2, seed=0)
        model = node2vec.fit()
        nodes = list(self.graph.nodes())
        dict_embeddings = {}
        for i in range(len(nodes)):
            dict_embeddings.update({nodes[i]: np.asarray(model.wv.get_vector(str(nodes[i])))})
        with open(path1, 'wb') as handle:
             pickle.dump(dict_embeddings, handle, protocol=3)
        return dict_embeddings

    def create_event2vec_embeddings(self):
        data_path = self.dataset + '_e2v_embeddings.txt'
        path = os.path.join(self.dataset, data_path)
        cond = 0
        dict_embeddings = {}
        with open(path, 'r') as f:
            for line in f:
                if cond == 1:
                    items = line.strip().split()
                    dict_embeddings[items[0]] = items[1:]
                cond = 1
        return dict_embeddings

    def create_ogre_embeddings(self, user_initial_nodes_choice=None):
        from oger_embedding.for_nni import StaticEmbeddings
        if user_initial_nodes_choice is not None:
            static_embeddings = StaticEmbeddings(name='pw', graph=self.graph, dim=self.dim,
                                                 choose='user_choice', initial_nodes=user_initial_nodes_choice)
        else:
            static_embeddings = StaticEmbeddings(name='pw', graph=self.graph, dim=self.dim)
        dict_embeddings = static_embeddings.dict_embedding
        return dict_embeddings


class TopKRanker(OneVsRestClassifier):
    """
    Linear regression with one-vs-rest classifier
    """

    def predict_kfir(self, x, top_k_list):
        assert x.shape[0] == len(top_k_list)
        probs = super(TopKRanker, self).predict_proba(x)
        prediction = np.zeros((x.shape[0], self.classes_.shape[0]))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
            for label in labels:
                prediction[i, int(label)] = 1
        return prediction, probs


class EdgesPreparation:
    def __init__(self, graph, args, split, dict_image_class=None):
        self.args = args
        # self.multi_graph = multi_graph
        self.split = split
        self.graph = graph
        self.label_edges = self.make_label_edges()
        self.unseen_edges, self.test_edges, self.dict_test_edges, self.dict_train_edges, self.dict_unseen_edges\
            = self.train_test_unseen_split()
        self.a = 1

    def make_label_edges(self):
        """
        Make a list with all the edge from type "labels_edges", i.e. edges between a movie and its class.
        :return: list with labels_edges
        """
        data_path = self.args.dataset + '_true_edges.pickle'
        nodes = list(self.graph.nodes)
        label_edges = []
        for node in nodes:
            if str(node)[0] == 'c':
                info = self.graph._adj[node]
                neighs = list(info.keys())
                for neigh in neighs:
                    if info[neigh]['key'] == 'labels_edges':
                        label_edges.append([node, neigh])
        try:
            with open(os.path.join(self.args.dataset, data_path), 'wb') as handle:
                pickle.dump(label_edges, handle, protocol=3)
        except:
            pass
        return label_edges

    @staticmethod
    def label_edges_classes_ordered(edge_data):
        """
        Make a dict of classes and their labels_edges they belong to. For every label_edge
        there is only one class it belongs to.
        :return: a dict of classes and their labels_edges
        """
        dict_class_label_edge = {}
        for edge in edge_data:
            if edge[0][0] == 'c':
                label = edge[0]
            else:
                label = edge[1]
            if dict_class_label_edge.get(label) is not None:
                edges = dict_class_label_edge[label]
                edges.append(edge)
                dict_class_label_edge[label] = edges
            else:
                dict_class_label_edge.update({label: [edge]})
        return dict_class_label_edge

    def train_test_unseen_split(self):  # unseen edges
        ratio = self.args.ratio[0]
        dict_true_edges = self.label_edges_classes_ordered(self.label_edges)
        classes = list(dict_true_edges.keys())
        for i, k in enumerate(sorted(dict_true_edges, key=lambda x: len(dict_true_edges[x]), reverse=True)):
            classes[i] = k
        seen_classes = classes[:int(self.args.seen_percentage * len(classes))]
        unseen_classes = classes[int(self.args.seen_percentage * len(classes)):]
        if self.split is not None:
            seen_classes = self.split['seen']
            unseen_classes = self.split['unseen']
        # unseen_classes.append(classes[0])
        unseen_edges, seen_edges, train_edges, test_edges = [], [], [], []
        for c in unseen_classes:
            # class_edges = list(self.graph.edges(c))
            # for edge in class_edges:
            #     self.graph[edge[0]][edge[1]]['weight'] *= 10
            for edge in dict_true_edges[c]:
                unseen_edges.append(edge)
        for c in seen_classes:
            seen_edges_c = []
            for edge in dict_true_edges[c]:
                seen_edges.append(edge)
                seen_edges_c.append(edge)
            random.Random(4).shuffle(seen_edges_c)
            train_edges_c = seen_edges_c[:int(ratio * len(seen_edges_c))]
            test_edges_c = seen_edges_c[int(ratio * len(seen_edges_c)):]
            for edge in train_edges_c:
                train_edges.append(edge)
            if len(test_edges_c) > 0:
                for edge in test_edges_c:
                    test_edges.append(edge)
        # unseen_edges = [dict_true_edges[c] for c in unseen_classes]
        # seen_edges = [dict_true_edges[c] for c in seen_classes]
        # random.Random(4).shuffle(seen_edges)
        # train_edges = seen_edges[:int(ratio * len(seen_edges))]
        # test_edges = seen_edges[int(ratio * len(seen_edges)):]
        dict_train_edges = self.label_edges_classes_ordered(train_edges)
        dict_test_edges = self.label_edges_classes_ordered(test_edges)
        dict_unseen_edges = self.label_edges_classes_ordered(unseen_edges)
        # for c in unseen_classes:
        #     unseen_edges.append(dict_true_edges[c])
        return unseen_edges, test_edges, dict_train_edges, dict_test_edges, dict_unseen_edges

    def seen_graph(self):
        graph = self.graph
        for edge in self.unseen_edges:
            graph.remove_edge(edge[0], edge[1])
        for edge in self.test_edges:
            graph.remove_edge(edge[0], edge[1])
        return graph

    def ogre_initial_nodes(self):
        train_classes = list(self.dict_train_edges.keys())
        train_nodes = train_classes.copy()
        for c in train_classes:
            train_nodes.append(self.dict_train_edges[c][0][1])
            # try:
            #     train_nodes.append(self.dict_train_edges[c][1][1])
            # except:
            #     continue
        return train_nodes

    def make_false_label_edges(self, dict_class_label_edge):
        """
        Make a dictionary of classes and false 'labels_edges' i.e. 'labels_edges' that do not exist.
        The number of false 'labels_edges' for each class in the dictionary is false_per_true times the true
        'labels_edges' of the class.
        In addition, to be more balance the function take randomly false 'labels_edges' but the number of
        false 'label_edges' corresponding to each class is similar.
        # We need the false 'labels_edges' to be a false instances to the classifier.
        :param dict_class_label_edge
        :return: a dict of classes and their false labels_edges.
        """
        data_path = self.args.dataset + '_false_edges_balanced_{}.pickle'.format(self.args.false_per_true)
        dict_class_false_edges = {}
        labels = list(dict_class_label_edge.keys())
        false_labels = []
        for label in labels:
            for edge in dict_class_label_edge[label]:
                if edge[0][0] == 'c':
                    label = edge[0]
                    movie = edge[1]
                else:
                    label = edge[1]
                    movie = edge[0]
                if len(false_labels) < self.args.false_per_true + 1:
                    false_labels = list(set(labels) - set([label]))  # The set makes every run different edges.
                else:
                    false_labels = list(set(false_labels) - set([label]))
                indexes = random.sample(range(1, len(false_labels)), self.args.false_per_true)
                # random.Random(4).shuffle(false_labels)
                # false_labels = false_labels[:self.args.false_per_true + 1]
                for i, index in enumerate(indexes):
                    if dict_class_false_edges.get(label) is None:
                        dict_class_false_edges[label] = [[movie, false_labels[index]]]
                    else:
                        edges = dict_class_false_edges[label]
                        edges.append([movie, false_labels[index]])
                        dict_class_false_edges[label] = edges
                false_labels = list(np.delete(np.array(false_labels), indexes))
        try:
            with open(os.path.join(self.args.dataset, data_path), 'wb') as handle:
                pickle.dump(dict_class_false_edges, handle, protocol=3)
        except:
            pass
        return dict_class_false_edges


class Classifier:
    def __init__(self, dict_train_true, dict_train_false, dict_test_true, dict_unseen_edges,
                 dict_projections, embedding, args, linear_classifier=True):
        self.args = args
        self.embedding = embedding
        self.dict_true_edges = dict_train_true
        self.dict_false_edges = dict_train_false
        self.dict_test_true = dict_test_true
        self.dict_unseen_edges = dict_unseen_edges
        self.norm = set(args.norm)
        self.dict_projections = dict_projections
        self.linear_classifier = linear_classifier

    def edges_embeddings(self, edges):
        embed_edges_0 = [self.dict_projections[edge[0]] for edge in edges]
        embed_edges_1 = [self.dict_projections[edge[1]] for edge in edges]
        # embed_false_edges_0 = [self.dict_projections[edge[0]] for edge in false_edges]
        # embed_false_edges_1 = [self.dict_projections[edge[1]] for edge in false_edges]
        embeddings = np.array([[*edge0, *edge1] for edge0, edge1 in zip(embed_edges_0, embed_edges_1)])
        # false_embeddings = np.array([[*edge0, *edge1] for edge0, edge1 in zip(embed_false_edges_0, embed_false_edges_1)])
        # y_true_edge = np.column_stack((np.ones(shape=(len(true_edges), 1)),
        #                                np.zeros(shape=(len(true_edges), 1)))).astype(int)
        # y_false_edge = np.column_stack((np.zeros(shape=(len(false_edges), 1)),
        #                                 np.ones(shape=(len(false_edges), 1)))).astype(int)
        return embeddings

    def edges_distance(self, edges):
        """
        Calculate the distance of an edge. Take the vertices of the edge and calculate the distance between their
        embeddings.
        We use to calculate The distance with L1, l2, Cosine Similarity.
        :param edge: the edge we want to find its distance.
        :return: The distance
        """
        embed_edges_0 = [self.dict_projections[edge[0]] for edge in edges]
        embed_edges_1 = [self.dict_projections[edge[1]] for edge in edges]
        if self.norm == set('L1 Norm'):
            norms = la.norm(np.subtract(embed_edges_0, embed_edges_1), 1, axis=1)
        elif self.norm == set('L2 Norm'):
            norms = la.norm(np.subtract(embed_edges_0, embed_edges_1), 2, axis=1)
        elif self.norm == set('cosine'):
            all_norms = cosine_similarity(embed_edges_0, embed_edges_1)
            norms = [math.acos(all_norms[i, i]) for i in range(len(all_norms))]
        else:
            raise ValueError(f"Wrong name of norm, {self.norm}")
        final_norms = np.array(norms).reshape(-1, 1)
        return final_norms

    def edge_distance(self, edge):
        """
        Calculate the distance of an edge. Take the vertices of the edge and calculate the distance between their
        embeddings.
        We use to calculate The distance with L1, l2, Cosine Similarity.
        :param edge: the edge we want to find its distance.
        :return: The distance
        """
        try:
            embd1 = np.array(self.dict_projections[edge[0]]).astype(float)
            embd2 = np.array(self.dict_projections[edge[1]]).astype(float)
        except:
            embd1 = np.ones(self.args.embedding_dimension).astype(float)
            embd2 = np.zeros(self.args.embedding_dimension).astype(float)
            pass
        if self.norm == set('L1 Norm'):
            norm = la.norm(np.subtract(embd1, embd2), 1)
        elif self.norm == set('L2 Norm'):
            norm = la.norm(np.subtract(embd1, embd2), 1)
        elif self.norm == set('cosine'):
            norm = math.acos(cosine_similarity(embd1.reshape(1, -1), embd2.reshape(1, -1))[0])
        else:
            raise ValueError(f"Wrong name of norm, {self.norm}")
        return norm

    def calculate_classifier_value(self, true_edges, false_edges):
        """
        Create x and y for Logistic Regression Classifier.
        self.dict_projections: A dictionary of all nodes embeddings, where keys==nodes and values==embeddings
        :param true_edges: A list of true edges.
        :param false_edges: A list of false edges.
        :return: x_true/x_false - The feature matrix for logistic regression classifier, of true/false edge.
        The i'th row is the norm score calculated for each edge.
                y_true_edge/y_false_edge - The edges labels, [1,0] for true/ [0,1] for false.
                Also the edge of the label is concatenate to the label.
        """
        if self.linear_classifier:
            x_true = self.edges_distance(true_edges)
            x_false = self.edges_distance(false_edges)
        else:
            x_true = self.edges_embeddings(true_edges)
            x_false = self.edges_embeddings(false_edges)
        # x_true, x_false = np.array(norms_true).reshape(-1, 1), np.array(norms_false).reshape(-1, 1)
        y_true_edge = np.column_stack((np.ones(shape=(len(true_edges), 1)),
                                       np.zeros(shape=(len(true_edges), 1)))).astype(int)
        y_false_edge = np.column_stack((np.zeros(shape=(len(false_edges), 1)),
                                        np.ones(shape=(len(false_edges), 1)))).astype(int)
        return x_true, x_false, y_true_edge, y_false_edge

    def calculate_by_single_norm(self, true_edges, false_edges):
        x_true, x_false = np.zeros(shape=(len(true_edges), 1)), np.zeros(shape=(len(false_edges), 1))
        y_true_edge, y_false_edge = np.zeros(shape=(len(true_edges), 4)).astype(int), \
            np.zeros(shape=(len(false_edges), 4)).astype(int)
        for i, edge in enumerate(true_edges):
            norm = self.edge_distance(edge)
            x_true[i, 0] = norm
            # y_true_edge[i, 2] = edge[0]
            # y_true_edge[i, 3] = edge[1]
            y_true_edge[i, 0] = str(1)
        for i, edge in enumerate(false_edges):
            norm = self.edge_distance(edge)
            x_false[i, 0] = norm
            # y_false_edge[i, 2] = edge[0]
            # y_false_edge[i, 3] = edge[1]
            y_false_edge[i, 1] = str(1)
        return x_true, x_false, y_true_edge, y_false_edge

    @staticmethod
    def train_edge_classification(x_train, y_train):
        """
        train  the classifier with the train set.
        :param x_train: The features' edge- norm (train set).
        :param y_train: The edges labels- 0 for true, 1 for false (train set).
        :return: The classifier
        """
        model = LogisticRegression()
        parameters = {"penalty": ["l2"], "C": [0.01, 0.1, 1]}
        model = TopKRanker(GridSearchCV(model, param_grid=parameters, cv=2, scoring='balanced_accuracy', n_jobs=1, verbose=0, pre_dispatch='n_jobs'))
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def create_keras_model(input_shape, hidden_layer_size):
        """
        :param input_shape: tuple - shape of a single sample (2d, 1)
        """
        from keras.models import Sequential
        import tensorflow as tf
        from keras.layers import Dense, Activation
        import keras
        model = Sequential()

        model.add(Dense(hidden_layer_size, activation="relu", input_shape=(input_shape, )))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        opti = keras.optimizers.Adam(learning_rate=0.01)

        model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])
        model.summary()

        return model

    @staticmethod
    def keras_model_fit(model, x_train, y):
        y_train = y.reshape((-1, 1))
        model.fit(x_train, y_train, epochs=5)
        print("done fitting")
        return model

    @staticmethod
    def keras_model_predict(model, x_test):
        y_pred = model.predict_proba(x_test)
        return y_pred

    @staticmethod
    def concat_data(x_true, x_false, y_true_edge, y_false_edge):
        """
        split the data into rain and test for the true edges and the false one.
        :param ratio: determine the train size.
        :return: THe split data
        """
        x_train, y_train = np.concatenate((x_true, x_false), axis=0),\
                                np.concatenate((y_true_edge, y_false_edge), axis=0)
        # y_train = np.array([y_train_edge.T[0].reshape(-1, 1), y_train_edge.T[1].reshape(-1, 1)]).T.reshape(-1,
        #                                                                                                    2).astype(
        #     int)
        return x_train, y_train

    def train(self):
        """
        Prepare the data for train, also train the classifier and make the test data divide by classes.
        :return: The classifier and dict_class_movie_test
        """
        path1 = os.path.join(self.args.dataset, f'train/classifier23_{self.embedding}_{self.args.norm}.pkl')
        path2 = os.path.join(self.args.dataset, f'train/dict_{self.embedding}_{self.args.norm}.pkl')
        classes = list(self.dict_true_edges.keys())
        for i, k in enumerate(sorted(self.dict_true_edges, key=lambda x: len(self.dict_true_edges[x]), reverse=True)):
            classes[i] = k
        dict_class_movie_test = {}
        x_train_all, y_train_all = np.array([]), np.array([])
        train_classes = list(self.dict_true_edges.keys())
        test_classes = list(self.dict_test_true.keys())
        unseen_classes = list(self.dict_unseen_edges.keys())
        classif2 = None
        for c in train_classes:
            if set(self.args.embedding) != set('OGRE'):
                x_true, x_false, y_true_edge, y_false_edge = \
                    self.calculate_classifier_value(self.dict_true_edges[c], self.dict_false_edges[c])
            else:
                x_true, x_false, y_true_edge, y_false_edge = \
                    self.calculate_by_single_norm(self.dict_true_edges[c], self.dict_false_edges[c])
            x_train, y_train = self.concat_data(x_true, x_false, y_true_edge, y_false_edge)
            if len(x_train_all) > 0:
                x_train_all = np.concatenate((x_train_all, x_train), axis=0)
                y_train_all = np.concatenate((y_train_all, y_train), axis=0)
            else:
                x_train_all = x_train
                y_train_all = y_train
        shuff = np.c_[x_train_all.reshape(len(x_train_all), -1), y_train_all.reshape(len(y_train_all), -1)]
        random.Random(4).shuffle(shuff)
        x_train_all = shuff.T[0].reshape(-1, 1)
        y_train_all = np.array([shuff.T[1].reshape(-1, 1), shuff.T[2].reshape(-1, 1)]).T.reshape(-1, 2).astype(
            int)
        if self.linear_classifier:
            classif2 = self.train_edge_classification(np.array(x_train_all), np.array(y_train_all))
        else:
            classif2 = self.create_keras_model(len(x_train_all[0]), int(self.args.enbedding_dimension/2))
        for c in test_classes:
            dict_movie_edge = {}
            for edge in self.dict_test_true[c]:
                if edge[0][0] == 'c':
                    movie = edge[1]
                else:
                    movie = edge[0]
                dict_movie_edge[movie] = edge
            dict_class_movie_test[c] = dict_movie_edge.copy()
        for c in unseen_classes:
            dict_movie_edge = {}
            for edge in self.dict_unseen_edges[c]:
                if edge[0][0] == 'c':
                    movie = edge[1]
                else:
                    movie = edge[0]
                dict_movie_edge[movie] = edge
            dict_class_movie_test[c] = dict_movie_edge.copy()
        # if not os.path.exists(os.path.join('Graph-ZSL', self.args.dataset)):
        #     os.makedirs(os.path.join('Graph-ZSL', self.args.dataset))
        with open(path2, 'wb') as fid:
            pickle.dump(dict_class_movie_test, fid)
        return classif2, dict_class_movie_test

    @staticmethod
    def predict_edge_classification(classif2, x_test):
        """
        With the test data make
        :param classif2:
        :param x_test:
        :return:
        """
        top_k_list = list(np.ones(len(x_test)).astype(int))
        prediction, probs = classif2.predict_kfir(x_test, top_k_list)
        if np.argmax(x_test) != np.argmax(probs.T[0]):
            print('stop')
        return prediction, probs

    def evaluate(self, classif2, dict_class_movie_test):
        # evaluate
        classes = list(dict_class_movie_test.keys())
        pred_true = []
        pred = []
        for i, k in enumerate(sorted(dict_class_movie_test, key=lambda x: len(dict_class_movie_test[x]), reverse=True)):
            classes[i] = k
        num_classes = len(classes)
        dict_measures = {'acc': {}, 'precision': {}}
        dict_class_measures = {}
        for c in classes:
            class_movies = list(dict_class_movie_test[c].keys())
            count = 0
            for m in class_movies:
                edges = np.array([np.repeat(m, num_classes), classes]).T
                class_test = np.zeros(shape=(len(edges), 1))
                if self.linear_classifier:
                    if set(self.args.embedding) != set('OGRE'):
                        class_test = self.edges_distance(edges)
                    else:
                        for i, edge in enumerate(edges):
                            norm = self.edge_distance(edge)
                            class_test[i, 0] = norm
                    _, probs = self.predict_edge_classification(classif2, class_test)
                else:
                    class_test = self.edges_embeddings(edges)
                    probs = self.keras_model_predict(class_test)
                pred_index = np.argmax(probs.T[0])
                # pred_index = np.argmax(class_test)
                prediction = edges[pred_index]
                real_edge = list(dict_class_movie_test[c][m])
                pred_true.append(c)
                if prediction[0][0] == 'c':
                    pred.append(prediction[0])
                else:
                    pred.append(prediction[1])
                if prediction[0] == real_edge[0]:
                    if prediction[1] == real_edge[1]:
                        count += 1
                elif prediction[1] == real_edge[0]:
                    if prediction[0] == real_edge[1]:
                        count += 1
            accuracy = count / len(class_movies)
            dict_measures['acc'] = accuracy
            dict_class_measures[c] = dict_measures.copy()
        with open(os.path.join(self.args.dataset, f'dict_class_measures_{self.embedding}_{self.args.norm}.pkl'),
                  'wb') as handle:
            pickle.dump(dict_class_measures, handle, protocol=3)
        # TODO dict class measures for every ratio
        return dict_class_measures, pred, pred_true

    def evaluate_for_hist(self, classif2, dict_class_movie_test, split=None):
        # evaluate
        classes = list(dict_class_movie_test.keys())
        hist_real_unseen_pred = np.zeros(len(classes))
        hist_real_unseen_first_unseen = np.zeros(len(classes))
        pred_true = []
        pred = []
        # for i, k in enumerate(sorted(dict_class_movie_test, key=lambda x: len(dict_class_movie_test[x]), reverse=True)):
        #     classes[i] = k
        num_classes = len(classes)
        seen_flag = np.zeros(int(self.args.seen_percentage*len(classes)))
        unseen_flag = np.ones(len(classes)-int(self.args.seen_percentage*len(classes)))
        classes_flag = np.concatenate((seen_flag, unseen_flag))
        dict_measures = {'acc': {}, 'precision': {}}
        dict_class_measures = {}
        for i, c in enumerate(classes):
            class_movies = list(dict_class_movie_test[c].keys())
            count = 0
            for m in class_movies:
                edges = np.array([np.repeat(m, num_classes), classes]).T
                class_test = np.zeros(shape=(len(edges), 1))
                if set(self.args.embedding) != set('OGRE'):
                    class_test = self.edges_distance(edges)
                else:
                    for j, edge in enumerate(edges):
                        norm = self.edge_distance(edge)
                        class_test[j, 0] = norm
                # _, probs = self.predict_edge_classification(classif2, class_test)
                # pred_index = np.argmax(probs.T[0])
                class_norm_test = np.column_stack((np.column_stack((class_test, classes)), classes_flag))
                sorted_class_norm = class_norm_test[np.argsort(class_norm_test[:, 0])]
                # if set(self.args.norm) == set('cosine'):
                #     sorted_class_norm = np.flip(sorted_class_norm)
                #     sort_classes = sorted_class_norm.T[0]
                # else:
                sort_classes = sorted_class_norm.T[1]
                sort_norm = sorted_class_norm.T[0].astype(float)
                sort_classes_flag = sorted_class_norm.T[2].astype(float)
                # class_test[::-1].sort(axis=0)
                prediction = np.array([m, sort_classes[0]])
                # prediction = edges[pred_index]
                real_edge = list(dict_class_movie_test[c][m])
                pred_true.append(c)
                if i > int(self.args.seen_percentage*len(classes)):
                    place = np.where(sort_classes == c)[0][0]
                    hist_real_unseen_pred[place] += 1
                place = np.where(sort_classes_flag == 1)[0][0]
                if self.args.unseen_weight_advantage*sort_norm[place] < sort_norm[0]:
                    pred.append(sort_classes[place])
                else:
                    pred.append(sort_classes[0])
                # pred.append(sort_classes[0])
                # if prediction[0][0] == 'c':
                #     pred.append(prediction[0])
                # else:
                #     pred.append(prediction[1])
                if prediction[0] == real_edge[0]:
                    if prediction[1] == real_edge[1]:
                        count += 1
                elif prediction[1] == real_edge[0]:
                    if prediction[0] == real_edge[1]:
                        count += 1
            accuracy = count / len(class_movies)
            dict_measures['acc'] = accuracy
            dict_class_measures[c] = dict_measures.copy()
        with open(os.path.join(self.args.dataset, f'dict_class_measures_{self.embedding}_{self.args.norm}.pkl'),
                  'wb') as handle:
            pickle.dump(dict_class_measures, handle, protocol=3)
        # TODO dict class measures for every ratio
        return dict_class_measures, pred, pred_true, hist_real_unseen_pred

    def hist_plot_for_unseen_dist_eval(self, distances):
        title = 'Histogram Of The Distance Between \n Unseen Label Norm And Predicted Norm'
        x_label = f'Distance, limit:{len(distances)}'
        y_label = 'Count'
        hist_plot(distances, title, x_label, y_label)
        plt.savefig(f'{self.args.dataset}/plots/hist_distance_real_unseen-prediction_'
                    f'{self.embedding}_{self.args.norm}_{int(100*self.args.seen_percentage)}_seen_percent')

    def confusion_matrix_maker(self, dict_class_measures, dict_class_movie_test,  pred, pred_true, split=None):
        conf_matrix = confusion_matrix(pred_true, pred, labels=list(dict_class_measures.keys()))
        if split is not None:
            seen_classes = split['seen']
            unseen_classes = split['unseen']
        seen_true_count = 0
        seen_count = 0
        unseen_true_count = 0
        unseen_count = 0
        seen_idx = []
        unseen_idx = []
        classes = list(dict_class_movie_test.keys())
        for i, c in enumerate(classes):
            if len(set([c]).intersection(set(self.dict_unseen_edges.keys()))) > 0:
                unseen_idx.append(i)
            else:
                seen_idx.append(i)
        for i in seen_idx:
            seen_true_count += conf_matrix[i][i]
            for j in range(len(classes)):
                seen_count += conf_matrix[i][j]
        for i in unseen_idx:
            unseen_true_count += conf_matrix[i][i]
            for j in range(len(conf_matrix)):
                unseen_count += conf_matrix[i][j]
        # seen_number = int(self.args.seen_percentage * len(conf_matrix))
        # for i in range(len(conf_matrix))[:seen_number]:
        #     seen_true_count += conf_matrix[i][i]
        #     for j in range(len(conf_matrix)):
        #         seen_count += conf_matrix[i][j]
        # for i in range(len(conf_matrix))[seen_number:]:
        #     unseen_true_count += conf_matrix[i][i]
        #     for j in range(len(conf_matrix)):
        #         unseen_count += conf_matrix[i][j]
        seen_accuracy = seen_true_count / seen_count
        unseen_accuracy = unseen_true_count / unseen_count
        harmonic_mean_ = harmonic_mean([seen_accuracy, unseen_accuracy])
        binary_conf_matrix = np.array([[seen_true_count, seen_count - seen_true_count],
                                       [unseen_count - unseen_true_count, unseen_true_count]])
        binary_conf_matrix = normalize(binary_conf_matrix, norm="l1") # to add
        print(f'accuracy all seen: {seen_accuracy}')
        print(f'accuracy all unseen: {unseen_accuracy}')
        print(f'Harmonic Mean all: {harmonic_mean_}')
        return harmonic_mean_, seen_accuracy, unseen_accuracy, conf_matrix, binary_conf_matrix

    def plot_confusion_matrix_all_classes(self, conf_matrix, binary_conf_matrix=None):
        title = f'Confusion Matrix, ZSL OUR_IMDB \n' \
                f'{self.embedding} {self.args.norm} {int(100 * self.args.seen_percentage)} Percent Seen'
        x_title = f"True Labels {int(100 * self.args.seen_percentage)}/{100 - int(100 * self.args.seen_percentage)}" \
                  f" (seen/unseen)"
        y_title = f"Predicted Labels"
        save_path = f'{self.args.dataset}/plots/confusion_matrix_{self.embedding}_{self.args.norm}' \
                    f'_{int(100 * self.args.seen_percentage)}_seen_percent'
        plot_confusion_matrix(conf_matrix, title, x_title, y_title, save_path)


def define_args(params):
    print(params)
    weights = np.array([params['weights_movie_movie'], params['weights_movie_class']]).astype(float)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest="dataset", help=' Name of the dataset', type=str, default=params['dataset']) # our_imdb, awa2, cub, lad
    parser.add_argument('--threshold', default=params['threshold'])
    parser.add_argument('--norm', default=params['norma_types'])  # cosine / L2 Norm / L1 Norm
    parser.add_argument('--embedding', default=params['embedding_type'])  # Node2Vec / Event2Vec / OGRE
    # embedding = params[2]
    parser.add_argument('--false_per_true', default=10)
    parser.add_argument('--ratio', default=[0.8])
    parser.add_argument('--seen_percentage', default=float(params['seen_percentage']))
    parser.add_argument('--embedding_dimension', default=int(params['embedding_dimensions']))
    parser.add_argument('--unseen_weight_advantage', default=0.9)
    parser.add_argument('--graph_percentage', default=1)
    if params['dataset'] == 'awa2' or params['dataset'] == 'cub' or params['dataset'] == 'lad':
        parser.add_argument('--attributes_edges_weight', default=params['attributes_edges_weight'])

        parser.add_argument('--images_nodes_percentage', default=1.0)
    # embedding_dimension = params[3].astype(int)
    args = parser.parse_args()
    return args, weights


def obj_func_grid(params, specific_split=True, split=None):
    """
    Main Function for link prediction task.
    :return:
    """
    args, weights = define_args(params)
    np.random.seed(0)
    # ratio_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    graph_maker = GraphImporter(args)
    # multi_graph = graph_maker.import_imdb_multi_graph(weights)
    if args.dataset == 'our_imdb':
        weighted_graph = graph_maker.import_imdb_weighted_graph(weights)
    elif args.dataset == 'awa2' or args.dataset == 'cub' or args.dataset == 'lad':
        awa2_att_weight = params['attributes_edges_weight']
        weighted_graph, split = graph_maker.import_data_graph(weights, specific_split, awa2_att_weight)
    else:
        raise ValueError(f"Wrong name of DataSet, {args.dataset}")
    edges_preparation = EdgesPreparation(weighted_graph, args, split)
    # dict_true_edges = edges_preparation.label_edges_classes_ordered(edges_preparation.label_edges)
    # dict_false_edges = edges_preparation.make_false_label_edges(dict_true_edges)
    dict_train_true = edges_preparation.dict_train_edges
    dict_test_true = edges_preparation.dict_test_edges
    dict_train_false = edges_preparation.make_false_label_edges(dict_train_true)
    dict_unseen_edges = edges_preparation.dict_unseen_edges
    graph = edges_preparation.seen_graph()
    embeddings_maker = EmbeddingCreator(graph, args.embedding_dimension, args)
    if args.embedding == 'Node2Vec':
        dict_embeddings = embeddings_maker.create_node2vec_embeddings()
    elif args.embedding == 'Event2Vec':
        dict_embeddings = embeddings_maker.create_event2vec_embeddings()
    elif args.embedding == 'OGRE':
        initial_nodes = edges_preparation.ogre_initial_nodes()
        dict_embeddings = embeddings_maker.create_ogre_embeddings(user_initial_nodes_choice=initial_nodes)
    else:
        raise ValueError(f"Wrong name of embedding, {args.embedding}")
    classifier = Classifier(dict_train_true, dict_train_false, dict_test_true, dict_unseen_edges,
                            dict_embeddings, args.embedding, args)
    classif, dict_class_movie_test = classifier.train()
    dict_class_measures_node2vec, pred, pred_true, hist_real_unseen_pred = classifier.evaluate_for_hist(classif, dict_class_movie_test)
    # classifier.hist_plot_for_unseen_dist_eval(hist_real_unseen_pred)
    _harmonic_mean, seen_accuracy, unseen_accuracy, conf_matrix, binary_conf_matrix = classifier.confusion_matrix_maker(
        dict_class_measures_node2vec, dict_class_movie_test, pred, pred_true, split)
    # classifier.plot_confusion_matrix_all_classes(conf_matrix, binary_conf_matrix)
    try:
        values = pd.read_csv('our_imdb/train/grid_search.csv')
        df1 = pd.DataFrame(np.column_stack((params.reshape(1, 8),
                                            np.array([_harmonic_mean, seen_accuracy, unseen_accuracy]).reshape(1, 3))),
                           columns=['movie_weights', 'labels_weights', 'embedding_type',
                                    'embedding_dimension', 'norma_type', 'class_edges_threshold', 'seen_percentage',
                                    'acc', 'seen_acc', 'unseen_acc', 'dataset'])
        frames1 = [values, df1]
        values = pd.concat(frames1, axis=0, names=['class_edges_threshold', 'movie_weights', 'labels_weights',
                                                   'embedding_type',
                                                   'embedding_dimension', 'norma_type', 'acc', 'seen_acc',
                                                   'unseen_acc', 'dataset'], sort=False)
    except:
        values = pd.DataFrame(np.column_stack((params.reshape(1, 8),
                                            np.array([_harmonic_mean, seen_accuracy, unseen_accuracy]).reshape(1, 3))),
                              columns=['movie_weights', 'labels_weights', 'embedding_type',
                                       'embedding_dimension', 'norma_type', 'class_edges_threshold', 'seen_percentage',
                                       'acc', 'seen_acc', 'unseen_acc', 'dataset'])
    values.to_csv('our_imdb/train/grid_search.csv', index=None)
    return _harmonic_mean, seen_accuracy, unseen_accuracy


if __name__ == '__main__':
    seen_accuracies, unseen_accuracies = [], []
    parameters = {
        "dataset": ['cub'],  # 'awa2', 'our_imdb'
        "embedding_type": ["Node2Vec"],
        "embedding_dimensions": [128],
        "weights_movie_class": [30],
        "weights_movie_movie": [1],
        # "weights_movie_class": np.logspace(-2, 3, 6),
        # "weights_movie_movie": np.logspace(-2, 3, 6),
        "norma_types": ['cosine'],
        "threshold": [0.3],
        "seen_percentage": [0.4],
        # "seen_percentage": np.linspace(0.1, 0.9, 9)
        "attributes_edges_weight": [100]  # 100 is the best for now
    }
    num = 0
    for dataset in parameters["dataset"]:
        for e_type in parameters["embedding_type"]:
            for dim in parameters["embedding_dimensions"]:
                for w_m_c in parameters["weights_movie_class"]:
                    for w_m_m in parameters["weights_movie_movie"]:
                        for norma_type in parameters["norma_types"]:
                            for threshold in parameters["threshold"]:
                                for per in parameters["seen_percentage"]:
                                    for att in parameters["attributes_edges_weight"]:
                                        param = np.array([dataset, e_type, dim, w_m_c, w_m_m, norma_type, threshold, per, att])
                                        dict_param = {p: param[i] for i, p in enumerate(list(parameters.keys()))}
                                        print(f'iteration number {num}')
                                        num += 1
                                        acc, seen_acc, unseen_acc = obj_func_grid(dict_param)
                                        seen_accuracies.append(seen_acc*100)
                                        unseen_accuracies.append(unseen_acc*100)
                                        # print("all accuracy: ", acc)
    dict_measures = {"unseen_accuracy": unseen_accuracies, "seen_accuracy": seen_accuracies}
    plots_2measures_vs_parameter(dict_measures, parameters["seen_percentage"], 'seen Percentage', parameters['dataset'],
                                 'Zero Shot Learning', "Accuracy", parameters["norma_types"][0],
                                 parameters["embedding_type"][0])
