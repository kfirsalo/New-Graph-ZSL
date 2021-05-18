import json
import multiprocessing
from datetime import datetime

from node2vec import Node2Vec
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
from sklearn.preprocessing import normalize
from statistics import harmonic_mean
import argparse
from numpy import linalg as la
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import model_selection as sk_ms
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import chain
from utils import set_gpu
from utlis_graph_zsl import hist_plot, plot_confusion_matrix, plots_2measures_vs_parameter, grid
from IMDb_data_preparation_E2V import MoviesGraph
import torch
from torch.backends import cudnn

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.deterministic = True
random.seed(seed)

# HEADER = ['movie_weights',
#           'labels_weights',
#           'embedding_type',
#           'embedding_dimension',
#           'norma_type',
#           'class_edges_threshold',
#           'seen_percentage',
#           'dataset',
#           'awa2_attributes_weight',
#           'harmonic_mean',
#           'seen_acc',
#           'unseen_acc']

HEADER = ['weights_movie_movie',
          'weights_movie_class',
          'embedding_type',
          'embedding_dimensions',
          'norma_types',
          'threshold',
          'seen_percentage',
          'data_name',
          'awa2_attributes_weight',
          'harmonic_mean',
          'seen_acc',
          'unseen_acc']


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
        weights_dict = {'movies_edges': weights[0], 'labels_edges': weights[1]}
        dict_paths = {'cast': 'data_set/IMDb title_principals.csv', 'genre': 'data_set/IMDb movies.csv'}
        imdb = MoviesGraph(dict_paths, self.args.graph_percentage)
        gnx = imdb.create_graph()
        labels = imdb.labels2int(gnx)
        knowledge_gnx, knowledge_data = imdb.create_knowledge_graph(labels, self.threshold)
        multi_gnx = imdb.weighted_multi_graph(gnx, knowledge_gnx, labels, weights_dict)
        return multi_gnx

    def import_imdb_weighted_graph(self, weights):
        weights_dict = {'movies_edges': weights[0], 'labels_edges': weights[1]}
        dict_paths = {'cast': 'data_set/IMDb title_principals.csv', 'genre': 'data_set/IMDb movies.csv'}
        imdb = MoviesGraph(dict_paths, self.args.graph_percentage)
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
        embeds_matrix, dict_image_embed, dict_image_class, val_images = graph_preparation.images_embed_calculator()
        dict_val_image_class = {image: dict_image_class[image] for image in val_images}
        dict_idx_image_class, dict_val_edges = {}, {}
        for i, image in enumerate(list(dict_image_class.keys())):
            dict_idx_image_class[str(i)] = dict_image_class[image]
            if dict_val_image_class.get(image) is not None:
                dict_val_edges[str(i)] = dict_image_class[image]
        final_graph_creator = FinalGraphCreator(dict_paths, embeds_matrix, dict_image_embed,
                                                dict_idx_image_class, self.args.images_nodes_percentage, self.args)
        image_graph = final_graph_creator.create_image_graph(radius)
        kg, dict_class_nodes_translation = final_graph_creator.imagenet_knowledge_graph()
        dict_val_edges = {img: dict_class_nodes_translation[dict_val_edges[img]] for img in list(dict_val_edges.keys())}
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
            return final_graph, dict_val_edges, split
        else:
            split = None
            return final_graph, dict_val_edges, split


class EmbeddingCreator(object):
    def __init__(self, graph=None, dimension=None, args=None):
        self.dataset = args.dataset
        self.dim = dimension
        self.graph = graph

    def create_node2vec_embeddings(self):
        # path1 = os.path.join(self.dataset, 'Node2Vec_embedding.pickle')
        # path2 = os.path.join(self.dataset, 'Node2Vec_embedding.csv')
        # if os.path.exists(path1):
        #     with open(path1, 'rb') as handle:
        #         dict_embeddings = pickle.load(handle)
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

        node2vec = Node2Vec(self.graph, dimensions=self.dim, walk_length=80, num_walks=16, workers=2)
        model = node2vec.fit()
        nodes = list(self.graph.nodes())
        dict_embeddings = {}
        for i in range(len(nodes)):
            dict_embeddings.update({nodes[i]: np.asarray(model.wv.get_vector(str(nodes[i])))})
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
        from StaticGraphEmbeddings.our_embeddings_methods.static_embeddings import StaticEmbeddings
        if user_initial_nodes_choice is not None:
            static_embeddings = StaticEmbeddings(self.dataset, self.graph, initial_size=100, initial_method="node2vec", method="OGRE", H=user_initial_nodes_choice,
                                                 dim=self.dim, choose="degrees", regu_val=0, weighted_reg=False, epsilon=0.1, file_tags=None)
        else:
            static_embeddings = StaticEmbeddings(self.dataset, self.graph, dim=self.dim)
        dict_embeddings = static_embeddings.dict_embedding
        return dict_embeddings


class EdgesPreparation:
    def __init__(self, graph, dict_val_edges, args, split=None):
        self.args = args
        # self.multi_graph = multi_graph
        self.split = split
        self.graph = graph
        self.label_edges = self.make_label_edges()
        self.unseen_edges, self.test_edges, self.dict_test_edges, self.dict_train_edges, self.dict_unseen_edges \
            = self.train_test_unseen_split(dict_val_edges)

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

    def train_test_unseen_split(self, _dict_val_edges):  # unseen edges
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
        for c in seen_classes:  # maybe not fair that I take train label edges that different from the train images I use to the ResNet50
            seen_edges_c = []
            for edge in dict_true_edges[c]:
                seen_edges.append(edge)
                seen_edges_c.append(edge)
                if _dict_val_edges is not None:  # deal with the case that I make dev set before (for the ResNet50)
                    if edge[0][0] == "c":
                        image = edge[1]
                    else:
                        image = edge[0]
                    if _dict_val_edges.get(image) is None:
                        train_edges.append(edge)
                    else:
                        test_edges.append(edge)
            if _dict_val_edges is None:  # other wise take 20 percent dev randomly
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

    def ogre_initial_nodes(self, gnx):
        train_classes = list(self.dict_train_edges.keys())
        train_nodes = train_classes.copy()
        for c in train_classes:
            train_nodes.append(self.dict_train_edges[c][0][1])
            # try:
            #     train_nodes.append(self.dict_train_edges[c][1][1])
            # except:
            #     continue
        intial_graph = gnx.subgraph(train_nodes)
        return intial_graph


class Classifier:
    def __init__(self, dict_train_true, dict_test_true, dict_unseen_edges,
                 dict_projections, embedding, split, args):
        self.args = args
        self.split = split
        self.embedding = embedding
        self.dict_true_edges = dict_train_true
        self.dict_test_true = dict_test_true
        self.dict_unseen_edges = dict_unseen_edges
        self.norm = set(args.norm)
        self.dict_projections = dict_projections

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
            try:
                all_norms = cosine_similarity(embed_edges_0, embed_edges_1)
                norms = []
                for i in range(len(all_norms)):
                    if np.abs(all_norms[i, i]) <= 1:
                        norms.append(math.acos(all_norms[i, i]))
                    elif all_norms[i, i] > 1:
                        norms.append(math.acos(1))
                    elif all_norms[i, i] < -1:
                        norms.append(math.acos(-1))
                # norms = [math.acos(all_norms[i, i]) if np.abs(all_norms[i, i]) < 1 else math.acos(1) for i in range(len(all_norms))]
            except:
                print('a')
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
        x_true = self.edges_distance(true_edges)
        x_false = self.edges_distance(false_edges)
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
    def concat_data(x_true, x_false, y_true_edge, y_false_edge):
        """
        split the data into rain and test for the true edges and the false one.
        :param ratio: determine the train size.
        :return: THe split data
        """
        x_train, y_train = np.concatenate((x_true, x_false), axis=0), \
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
        path2 = os.path.join(self.args.dataset, f'train/dict_{self.embedding}_{self.args.norm}.pkl')
        classes = list(self.dict_true_edges.keys())
        # for i, k in enumerate(sorted(self.dict_true_edges, key=lambda x: len(self.dict_true_edges[x]), reverse=True)):
        #     classes[i] = k
        dict_class_movie_test = {}
        test_classes = list(self.dict_test_true.keys())
        unseen_classes = list(self.dict_unseen_edges.keys())
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
        return dict_class_movie_test

    def evaluate(self, dict_class_movie_test):
        # evaluate
        classes = list(dict_class_movie_test.keys())
        pred_true = []
        pred = []
        # for i, k in enumerate(sorted(dict_class_movie_test, key=lambda x: len(dict_class_movie_test[x]), reverse=True)):
        #     classes[i] = k
        num_classes = len(classes)
        dict_measures = {'acc': {}, 'precision': {}}
        dict_class_measures = {}
        for c in classes:
            class_movies = list(dict_class_movie_test[c].keys())
            count = 0
            for m in class_movies:
                edges = np.array([np.repeat(m, num_classes), classes]).T
                class_test = np.zeros(shape=(len(edges), 1))
                # if set(self.args.embedding) != set('OGRE'):
                class_test = self.edges_distance(edges)
                # else:
                #     for i, edge in enumerate(edges):
                #         norm = self.edge_distance(edge)
                #         class_test[i, 0] = norm
                # _, probs = self.predict_edge_classification(classif2, class_test)
                # pred_index = np.argmax(probs.T[0])
                pred_index = np.argmax(class_test)
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

    def count_seen_num(self, classes=None):
        if self.split is not None:
            seen_classes = self.split['seen']
            unseen_classes = self.split['unseen']
            seen_num = len(seen_classes)
            unseen_num = len(unseen_classes)
        else:
            seen_num = int(self.args.seen_percentage*len(classes))
            unseen_num = len(classes)-int(self.args.seen_percentage*len(classes))
        return seen_num, unseen_num

    def evaluate_for_hist(self, dict_class_movie_test):
        # evaluate
        classes = list(dict_class_movie_test.keys())
        # for i, k in enumerate(sorted(dict_class_movie_test, key=lambda x: len(dict_class_movie_test[x]), reverse=True)):
        #     classes[i] = k
        hist_real_unseen_pred = np.zeros(len(classes))
        hist_real_unseen_first_unseen = np.zeros(len(classes))
        pred_true = []
        pred = []
        # for i, k in enumerate(sorted(dict_class_movie_test, key=lambda x: len(dict_class_movie_test[x]), reverse=True)):
        #     classes[i] = k
        num_classes = len(classes)
        seen_num, unseen_num = self.count_seen_num(classes)
        seen_flag = np.zeros(seen_num)
        unseen_flag = np.ones(unseen_num)

        classes_flag = np.concatenate((seen_flag, unseen_flag))
        dict_measures = {'acc': {}, 'precision': {}}
        dict_class_measures = {}
        for i, c in enumerate(classes):
            class_movies = list(dict_class_movie_test[c].keys())
            count = 0
            for m in class_movies:
                edges = np.array([np.repeat(m, num_classes), classes]).T
                class_test = np.zeros(shape=(len(edges), 1))
                # if set(self.args.embedding) != set('OGRE'):
                class_test = self.edges_distance(edges)
                # else:
                #     for j, edge in enumerate(edges):
                #         norm = self.edge_distance(edge)
                #         class_test[j, 0] = norm
                # _, probs = self.predict_edge_classification(classif2, class_test)
                # pred_index = np.argmax(probs.T[0])
                try:
                    class_norm_test = np.column_stack((np.column_stack((class_test, classes)), classes_flag))
                except:
                    print('a')
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
                if i > seen_num:
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
        plt.close()

    def confusion_matrix_maker(self, dict_class_measures, pred, pred_true):
        conf_matrix = confusion_matrix(pred_true, pred, labels=list(dict_class_measures.keys()))
        seen_true_count = 0
        binary_seen_true_count = 0
        seen_count = 0
        unseen_true_count = 0
        binary_unseen_true_count = 0
        unseen_count = 0
        seen_number = int(self.args.seen_percentage * len(conf_matrix))
        classes = list(dict_class_measures.keys())
        seen_idx = []
        unseen_idx = []
        for i, c in enumerate(classes):
            if len(set([c]).intersection(set(self.dict_unseen_edges.keys()))) > 0:
                unseen_idx.append(i)
            else:
                seen_idx.append(i)
        for i in seen_idx:
            seen_true_count += conf_matrix[i][i]
            for j in range(len(classes)):
                seen_count += conf_matrix[i][j]
            for j in seen_idx:
                binary_seen_true_count += conf_matrix[i][j]
        for i in unseen_idx:
            unseen_true_count += conf_matrix[i][i]
            for j in range(len(conf_matrix)):
                unseen_count += conf_matrix[i][j]
            for j in unseen_idx:
                binary_unseen_true_count += conf_matrix[i][j]
        # for i in range(len(conf_matrix))[:seen_number]:
        #     seen_true_count += conf_matrix[i][i]
        #     for j in range(len(conf_matrix)):
        #         seen_count += conf_matrix[i][j]k
        # for i in range(len(conf_matrix))[seen_number:]:
        #     unseen_true_count += conf_matrix[i][i]
        #     for j in range(len(conf_matrix)):
        #         unseen_count += conf_matrix[i][j]
        seen_accuracy = seen_true_count / seen_count
        unseen_accuracy = unseen_true_count / unseen_count
        harmonic_mean_ = harmonic_mean([seen_accuracy, unseen_accuracy])
        # seen_unseen_conf_matrix = np.array([[seen_true_count, seen_count - seen_true_count],
        #                                [unseen_count - unseen_true_count, unseen_true_count]])
        binary_conf_matrix = np.array([[binary_seen_true_count, seen_count - binary_seen_true_count],
                                       [unseen_count - binary_unseen_true_count, binary_unseen_true_count]])
        binary_conf_matrix = normalize(binary_conf_matrix, norm="l1") # to add
        print(f'accuracy all seen: {seen_accuracy}')
        print(f'accuracy all unseen: {unseen_accuracy}')
        print(f'Harmonic Mean all: {harmonic_mean_}')
        return harmonic_mean_, seen_accuracy, unseen_accuracy, conf_matrix, binary_conf_matrix

    def plot_confusion_matrix_all_classes(self, conf_matrix, binary_conf_matrix=None):
        title = f'Confusion Matrix, ZSL {self.args.dataset} \n' \
                f'{self.embedding} {self.args.norm} {int(100 * self.args.seen_percentage)} Percent Seen'
        x_title = f"True Labels {int(100 * self.args.seen_percentage)}/{100 - int(100 * self.args.seen_percentage)}" \
                  f" (seen/unseen)"
        y_title = f"Predicted Labels"
        save_path = f'{self.args.dataset}/plots/confusion_matrix_{self.embedding}_{self.args.norm}'\
                    f'_{int(100 * self.args.seen_percentage)}_seen_percent'
        # plot_confusion_matrix(conf_matrix, title, x_title, y_title, save_path)
        if binary_conf_matrix is not None:
            y_binary = "True Seen/Unseen"
            x_binary = "Predicted Seen/Unseen"
            binary_title = "Binary " + title
            save_path_binary = f'{self.args.dataset}/plots/binary_confusion_matrix_{self.embedding}_{self.args.norm}'\
                               f'_{int(100 * self.args.seen_percentage)}_seen_percent'
            plot_confusion_matrix(binary_conf_matrix, binary_title, x_binary, y_binary, save_path_binary, vmax=None,
                                  vmin=None, cmap=None)


from dataclasses import dataclass
@dataclass
class InventoryItem:
    """Class for keeping track of an item in inventory."""
    dataset: str
    threshold: float
    norm: str
    embedding: str
    false_per_true: str
    norm: str


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
    parser.add_argument('--graph_percentage', default=0.1)
    if params['dataset'] == 'awa2' or params['dataset'] == 'cub' or params['dataset'] == 'lad':
        parser.add_argument("--train_percentage", help="train percentage from the seen images", default=90)

        parser.add_argument('--attributes_edges_weight', default=params['attributes_edges_weight'])

        parser.add_argument('--images_nodes_percentage', default=0.15)
    # embedding_dimension = params[3].astype(int)
    args = parser.parse_args()
    return args, weights


def obj_func_grid(params, specific_split=True, split=None):  # split False or True
    """
    Main Function for link prediction task.
    :return:
    """
    args, weights = define_args(params)
    np.random.seed(0)
    # ratio_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    graph_maker = GraphImporter(args)
    # multi_graph = graph_maker.import_imdb_multi_graph(weights)
    dict_val_edges = None
    if args.dataset == 'our_imdb':
        weighted_graph = graph_maker.import_imdb_weighted_graph(weights)
    elif args.dataset == 'awa2' or args.dataset == 'cub' or args.dataset == 'lad':
        awa2_att_weight = params['attributes_edges_weight']
        weighted_graph, dict_val_edges, split = graph_maker.import_data_graph(weights, specific_split, awa2_att_weight)
    else:
        raise ValueError(f"Wrong name of DataSet, {args.dataset}")
    edges_preparation = EdgesPreparation(weighted_graph, dict_val_edges, args, split)
    # dict_true_edges = edges_preparation.label_edges_classes_ordered(edges_preparation.label_edges)
    # dict_false_edges = edges_preparation.make_false_label_edges(dict_true_edges)
    dict_train_true = edges_preparation.dict_train_edges
    dict_test_true = edges_preparation.dict_test_edges
    dict_unseen_edges = edges_preparation.dict_unseen_edges
    graph = edges_preparation.seen_graph()
    embeddings_maker = EmbeddingCreator(graph, args.embedding_dimension, args)
    if args.embedding == 'Node2Vec':
        dict_embeddings = embeddings_maker.create_node2vec_embeddings()
    elif args.embedding == 'Event2Vec':
        dict_embeddings = embeddings_maker.create_event2vec_embeddings()
    elif args.embedding == 'OGRE':
        initial_nodes = edges_preparation.ogre_initial_nodes(graph)
        dict_embeddings = embeddings_maker.create_ogre_embeddings(user_initial_nodes_choice=initial_nodes)
    else:
        raise ValueError(f"Wrong name of embedding, {args.embedding}")
    classifier = Classifier(dict_train_true, dict_test_true, dict_unseen_edges,
                            dict_embeddings, args.embedding, split, args)
    dict_class_movie_test = classifier.train()
    dict_class_measures_node2vec, pred, pred_true, hist_real_unseen_pred = classifier.evaluate_for_hist(dict_class_movie_test)
    # classifier.hist_plot_for_unseen_dist_eval(hist_real_unseen_pred)
    _harmonic_mean, seen_accuracy, unseen_accuracy, conf_matrix, binary_conf_matrix = classifier.confusion_matrix_maker(
        dict_class_measures_node2vec, pred, pred_true)
    classifier.plot_confusion_matrix_all_classes(conf_matrix, binary_conf_matrix)
    return _harmonic_mean, seen_accuracy, unseen_accuracy


def flatten_dict(d):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key + "." + subkey, subvalue
            else:
                yield key, value
    return dict(items())


def config_to_str(config):
    config = flatten_dict(config)
    return [str(config.get(k, "--")) for k in HEADER]


def run_grid(grid_params, res_dir, now):
    grid_params = grid_params if type(grid_params) is dict else json.load(open(grid_params, "rt"))
    res_filename = os.path.join(res_dir, f"{grid_params['dataset'][0]}_grid_{now}.csv")
    out = open(res_filename, "wt")
    out.write(f"{','.join(HEADER)}\n")
    for config in grid(grid_params):
        param = {p: config[i] for i, p in enumerate(list(grid_params.keys()))}
        harmonic_mean_, seen_acc, unseen_acc = obj_func_grid(param)
        table_row = config_to_str(param)
        table_row[HEADER.index('harmonic_mean')] = str(harmonic_mean_)
        table_row[HEADER.index('seen_acc')] = str(seen_acc)
        table_row[HEADER.index('unseen_acc')] = str(unseen_acc)
        out.write(f"{','.join(table_row)}\n")
    out.close()


def main():
    seen_accuracies, unseen_accuracies = [], []
    parameters = {
        "dataset": ['our_imdb'],  # 'awa2', 'our_imdb'
        "embedding_type": ["Node2Vec"],
        "embedding_dimensions": [32, 64, 128, 256],
        # "weights_movie_class": [1],
        # "weights_movie_movie": [1],
        "weights_movie_class": np.logspace(-2, 3, 6),
        "weights_movie_movie": np.logspace(-2, 3, 6),
        "norma_types": ['cosine'],
        "threshold": [0.3, 0.6, 0.9],
        "seen_percentage": [0.8],
        # "seen_percentage": np.linspace(0.1, 0.9, 9)
        "attributes_edges_weight": [100]  # 100 is the best for now
    }
    num = 0
    for param in grid(parameters):
        dict_param = {p: param[i] for i, p in enumerate(list(parameters.keys()))}
        # param = np.array([w_m_m, w_m_c, e_type, dim, norma_type, threshold, per, data, w_att])
        print(f'iteration number {num}')
        num += 1
        harmonic_mean_, seen_acc, unseen_acc = obj_func_grid(dict_param)
        seen_accuracies.append(seen_acc*100)
        unseen_accuracies.append(unseen_acc*100)
        # print("all accuracy: ", acc)
    dict_measures = {"unseen_accuracy": unseen_accuracies, "seen_accuracy": seen_accuracies}
    plots_2measures_vs_parameter(dict_measures, parameters["seen_percentage"], 'seen Percentage', 'our_imdb',
                                 'Zero Shot Learning', "Accuracy", parameters["norma_types"][0],
                                 parameters["embedding_type"][0])


if __name__ == '__main__':
    res_dir = "C:\\Users\\kfirs\\lab\\Zero Shot Learning\\New-Graph-ZSL\\grid_results"
    # now = datetime.now().strftime("%d%m%y_%H%M%S")
    now = "29_04_21"
    # parameters = {
    #     "dataset": ['our_imdb'],  # 'awa2', 'our_imdb'
    #     "embedding_type": ["Node2Vec"],
    #     "embedding_dimensions": [32, 64, 128, 256],
    #     # "weights_movie_class": [1],
    #     # "weights_movie_movie": [1],
    #     "weights_movie_class": np.logspace(-2, 3, 6),
    #     "weights_movie_movie": np.logspace(-2, 3, 6),
    #     "norma_types": ['cosine'],
    #     "threshold": [0.3, 0.6, 0.9],
    #     "seen_percentage": [0.8],
    #     # "seen_percentage": np.linspace(0.1, 0.9, 9)
    #     "attributes_edges_weight": [100]  # 100 is the best for now
    # }
    parameters = {
        "dataset": ['cub'],  # 'awa2', 'our_imdb', 'cub', 'lad'
        "embedding_type": ["Node2Vec"],
        "embedding_dimensions": [128],
        "weights_movie_class": [30],
        "weights_movie_movie": [1],
        "norma_types": ['cosine'],  # 'cosine', "L2 Norm", "L1 Norm"
        "threshold": [0.3],
        "seen_percentage": [0.8],
        # "seen_percentage": np.linspace(0.1, 0.9, 9)
        "attributes_edges_weight": [100]  # 100 is the best for now
    }
    processes = []
    parameters_by_procesess = []
    for data in parameters["dataset"]:
        # for w_m_c in parameters["weights_movie_class"]:
        param_by_parameters = parameters.copy()
        param_by_parameters["dataset"] = [data]
            # param_by_parameters["weights_movie_class"] = [w_m_c]
        parameters_by_procesess.append(param_by_parameters)
    for i in range(len(parameters_by_procesess)):
        proc = multiprocessing.Process(target=run_grid, args=(parameters_by_procesess[i], res_dir, now, ))
        processes.append(proc)
        proc.start()
    for p in processes:
        p.join()

