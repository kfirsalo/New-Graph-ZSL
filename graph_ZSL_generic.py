import json
import multiprocessing
from datetime import datetime
from pathlib import Path
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
import nni
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import chain
from utils import set_gpu
from utlis_graph_zsl import hist_plot, plot_confusion_matrix, plots_2measures_vs_parameter, grid, nested_nni_to_dict, \
    replace_max
from IMDb_data_preparation_E2V import MoviesGraph
import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
from gem.embedding.gf import GraphFactorization
from gem.embedding.hope import HOPE
from draw_unseen_graph import DrawUnseenGraph

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

# HEADER = ['instance_edges_weight',
#           'label_edges_weight',
#           'embedding_type',
#           'embedding_dimension',
#           'norm_type',
#           'kg_jacard_similarity_threshold',
#           'seen_percentage',
#           'data_name',
#           'awa2_attributes_weight',
#           'harmonic_mean',
#           'seen_acc',
#           'unseen_acc',
#           'seen_only_acc',
#           'unseen_only_acc',
#           'seen_count',
#           'unseen_count']
HEADER = ['dataset',
          'embedding_type',
          'graph_percentage',
          'embedding_dimension',
          'ogre_second_neighbor_advantage',
          'label_edges_weight',
          'instance_edges_weight',
          'attributes_edges_weight',
          'kg_jaccard_similarity_threshold(imdb)',
          'seen_percentage(imdb)',
          'link_prediction_type',
          'score_maximize',
          'optimizer',
          'lr',
          'epochs',
          'loss',
          'norm_type',
          'logistic_regression_regularization',
          'false_per_true_edges',
          'seen_weight_advantage',
          'harmonic_mean',
          'seen_acc',
          'unseen_acc',
          'seen_only_acc',
          'unseen_only_acc',
          'seen_count',
          'unseen_count']


class GraphImporter:
    """
    class that responsible to import or create the relevant graph
    """

    def __init__(self, args):
        self.dataset = args.dataset
        self.graph_percentage = args.graph_percentage
        self.kg_jacard_similarity_threshold = args.kg_jacard_similarity_threshold
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
        knowledge_gnx, knowledge_data = imdb.create_knowledge_graph(labels, self.kg_jacard_similarity_threshold)
        multi_gnx = imdb.weighted_multi_graph(gnx, knowledge_gnx, labels, weights_dict)
        return multi_gnx

    def import_imdb_weighted_graph(self, weights):
        weights_dict = {'movies_edges': weights[0], 'labels_edges': weights[1]}
        dict_paths = {'cast': 'data_set/IMDb title_principals.csv', 'genre': 'data_set/IMDb movies.csv'}
        imdb = MoviesGraph(dict_paths, self.args.graph_percentage)
        gnx = imdb.create_graph()
        labels = imdb.labels2int(gnx)
        knowledge_gnx, knowledge_data = imdb.create_knowledge_graph(labels, float(self.kg_jacard_similarity_threshold))
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

    def import_data_graph(self, final_graph_weights, specific_split, att_weight, gephi_display=False):
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
        if self.args.dataset == "awa2_w_imagenet":
            dict_class_name = final_graph_creator.dict_class_name
            dict_val_edges = {img: dict_class_nodes_translation[dict_class_name[dict_val_edges[img]]] for img in
                              list(dict_val_edges.keys())}
        else:
            dict_val_edges = {img: dict_class_nodes_translation[dict_val_edges[img]] for img in
                              list(dict_val_edges.keys())}
        kg = final_graph_creator.attributed_graph(kg, dict_class_nodes_translation, att_weight, radius)
        seen_classes, unseen_classes = final_graph_creator.seen_classes, final_graph_creator.unseen_classes
        seen_classes = [dict_class_nodes_translation[c] for c in seen_classes]
        unseen_classes = [dict_class_nodes_translation[c] for c in unseen_classes]
        if gephi_display:
            from kg2gephi import KG2Gephi
            relevant_kg = kg.subgraph(set(chain(seen_classes, unseen_classes))).copy()
            if self.args.dataset == "awa2_w_imagenet":
                nodes_translate = {node: final_graph_creator.dict_name_class[dict_class_nodes_translation[node]] for
                                   node in list(relevant_kg.nodes())}
            else:
                nodes_translate = dict_class_nodes_translation
            kg2gephi = KG2Gephi(relevant_kg, seen_classes, unseen_classes)
            edges_path = Path(f"{self.args.dataset}/plots/gephi/kg_jaccard_edges.csv")
            edges_path.parent.mkdir(parents=True, exist_ok=True)
            nodes_path = Path(f"{self.args.dataset}/plots/gephi/kg_jaccard_nodes.csv")
            nodes_path.parent.mkdir(parents=True, exist_ok=True)
            kg2gephi.extract_kg_csv(edges_path=edges_path, nodes_path=nodes_path, nodes_translate=nodes_translate)
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
        self.args = args
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
        # from StaticGraphEmbeddings.our_embeddings_methods.static_embeddings import StaticEmbeddings
        from OGRE.our_embeddings_methods.static_embeddings import StaticEmbeddings
        if user_initial_nodes_choice is not None:
            static_embeddings = StaticEmbeddings(self.dataset, self.graph, initial_size=100, initial_method="node2vec",
                                                 method="OGRE", H=user_initial_nodes_choice,
                                                 dim=self.dim, choose="degrees", regu_val=0, weighted_reg=False,
                                                 epsilon=self.args.ogre_second_neighbor_advantage, file_tags=None)
        else:
            static_embeddings = StaticEmbeddings(self.dataset, self.graph, dim=self.dim)
        dict_embeddings = static_embeddings.list_dicts_embedding[0]
        return dict_embeddings

    def create_hope_embeddings(self):
        hope = HOPE(d=128, beta=0.01)
        x, _ = hope.learn_embedding(self.graph, is_weighted=True)
        nodes = list(self.graph.nodes())
        dict_embeddings = {}
        for i in range(len(nodes)):
            dict_embeddings[nodes[i]] = x[i]
        return dict_embeddings


class EdgesPreparation:
    def __init__(self, graph, dict_val_edges, args, split=None):
        self.args = args
        # self.multi_graph = multi_graph
        self.split = split
        self.graph = graph
        self.label_edges = self.make_label_edges()
        self.unseen_edges, self.train_edges, self.test_edges, self.dict_test_edges, self.dict_train_edges, self.dict_unseen_edges \
            = self.train_test_unseen_split(dict_val_edges)

    def make_label_edges(self, graph=None):
        """
        Make a list with all the edge from type "labels_edges", i.e. edges between a movie and its class.
        :return: list with labels_edges
        """
        if graph is None:
            graph = self.graph
        data_path = self.args.dataset + '_true_edges.pickle'
        nodes = list(graph.nodes)
        label_edges = []
        for node in nodes:
            if str(node)[0] == 'c':
                info = graph._adj[node]
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
                edge = [edge[1], edge[0]]
            #     label = edge[0]
            # else:
            #     label = edge[1]
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
        return unseen_edges, train_edges, test_edges, dict_train_edges, dict_test_edges, dict_unseen_edges

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

    def graph_without_unseen_classes(self, gnx):
        initial_graph = gnx.copy()
        unseen_classes = list(self.dict_unseen_edges.keys())
        unseen_classes = unseen_classes.copy()
        for c in unseen_classes:
            initial_graph.remove_node(c)
        return initial_graph

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
        data_path = self.args.dataset + '_false_edges_balanced_{}.pickle'.format(self.args.false_per_true_edges)
        dict_class_false_edges = {}
        labels = list(dict_class_label_edge.keys())
        false_labels = []
        for label in labels:
            for edge in dict_class_label_edge[label]:
                if edge[0][0] == 'c':
                    edge = [edge[1], edge[0]]
                #     label = edge[0]
                #     movie = edge[1]
                # else:
                label = edge[1]
                movie = edge[0]
                if len(false_labels) < self.args.false_per_true_edges + 1:
                    false_labels = list(set(labels) - set([label]))  # The set makes every run different edges.
                else:
                    false_labels = list(set(false_labels) - set([label]))
                indexes = random.sample(range(1, len(false_labels)), self.args.false_per_true_edges)
                # random.Random(4).shuffle(false_labels)
                # false_labels = false_labels[:self.args.false_per_true_edges + 1]
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
    def __init__(self, dict_train_true, dict_test_true, dict_unseen_edges,
                 dict_projections, embedding, split, args):
        self.args = args
        self.split = split
        self.embedding = embedding
        self.dict_true_edges = dict_train_true
        self.dict_test_true = dict_test_true
        self.dict_unseen_edges = dict_unseen_edges
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
        if set(self.args.norm) == set('L1 Norm'):
            norms = la.norm(np.subtract(embed_edges_0, embed_edges_1), 1, axis=1)
        elif set(self.args.norm) == set('L2 Norm'):
            norms = la.norm(np.subtract(embed_edges_0, embed_edges_1), 2, axis=1)
        elif set(self.args.norm) == set('cosine'):
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
            raise ValueError(f"Wrong name of norm, {self.args.norm}")
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
        if set(self.args.norm) == set('L1 Norm'):
            norm = la.norm(np.subtract(embd1, embd2), 1)
        elif set(self.args.norm) == set('L2 Norm'):
            norm = la.norm(np.subtract(embd1, embd2), 1)
        elif set(self.args.norm) == set('cosine'):
            norm = math.acos(cosine_similarity(embd1.reshape(1, -1), embd2.reshape(1, -1))[0])
        else:
            raise ValueError(f"Wrong name of norm, {self.args.norm}")
        return norm

    def edges_embeddings(self, edges):
        embed_edges_0 = [self.dict_projections[edge[0]] for edge in edges]
        embed_edges_1 = [self.dict_projections[edge[1]] for edge in edges]
        embeddings = np.array([[*edge0, *edge1] for edge0, edge1 in zip(embed_edges_0, embed_edges_1)])
        return embeddings

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
        if self.args.link_prediction_type == "norm_linear_regression":
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

    def ml_train(self, dict_false_edges):
        classes = list(self.dict_true_edges.keys())
        # for i, k in enumerate(sorted(self.dict_true_edges, key=lambda x: len(self.dict_true_edges[x]), reverse=True)):
        #     classes[i] = k
        dict_class_movie_test = {}
        x_train_all, y_train_all = np.array([]), np.array([])
        train_classes = list(self.dict_true_edges.keys())
        test_classes = list(self.dict_test_true.keys())
        unseen_classes = list(self.dict_unseen_edges.keys())
        classif2 = None
        for c in train_classes:
            x_true, x_false, y_true_edge, y_false_edge = \
                self.calculate_classifier_value(self.dict_true_edges[c], dict_false_edges[c])
            x_train, y_train = self.concat_data(x_true, x_false, y_true_edge, y_false_edge)
            if len(x_train_all) > 0:
                x_train_all = np.concatenate((x_train_all, x_train), axis=0)
                y_train_all = np.concatenate((y_train_all, y_train), axis=0)
            else:
                x_train_all = x_train
                y_train_all = y_train
        shuff = np.c_[x_train_all.reshape(len(x_train_all), -1), y_train_all.reshape(len(y_train_all), -1)]
        # random.Random(4).shuffle(shuff)
        np.random.shuffle(shuff)
        # if self.linear_classifier:
        #     x_train_all = shuff.T[0].reshape(-1, 1)
        #     y_train_all = np.array([shuff.T[1].reshape(-1, 1), shuff.T[2].reshape(-1, 1)]).T.reshape(-1, 2).astype(
        #         int)
        x_train_all = shuff[:, :-2]
        y_train_all = shuff[:, -2:].astype(int)
        if self.args.link_prediction_type == "norm_linear_regression":
            from link_prediction_models import train_edge_classification
            classif2 = train_edge_classification(np.array(x_train_all), np.array(y_train_all),
                                                 solver=self.args.regression_solver)
        elif self.args.link_prediction_type == "embedding_neural_network":
            from link_prediction_models import create_keras_model, keras_model_fit, EmbeddingLinkPredictionDataset, \
                EmbeddingLinkPredictionNetwork, TrainLinkPrediction
            # classif2 = create_keras_model(len(x_train_all[0]), int(self.args.embedding_dimension / 2))
            # classif2 = keras_model_fit(classif2, x_train_all, y_train_all)
            x_path = Path("save_data_graph/lad/final_graph_embeddings.npy")
            y_path = Path("save_data_graph/lad/final_graph_labels.npy")
            np.save(x_path, x_train_all)
            np.save(y_path, y_train_all)
            dataset = EmbeddingLinkPredictionDataset(x_train_all, y_train_all)
            train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)),
                                                                         len(dataset) - int(0.8 * len(dataset))])
            train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
            net = EmbeddingLinkPredictionNetwork(len(x_train_all[0]), int(self.args.embedding_dimension / 2),
                                                 lr=self.args.embedding_nn_lr,
                                                 weight_decay=self.args.embedding_nn_weight_decay,
                                                 dropout_prob=self.args.embedding_nn_dropout_prob,
                                                 optimizer=self.args.embedding_nn_optimizer,
                                                 loss=self.args.embedding_nn_loss,
                                                 pos_weight=self.args.false_per_true_edges)
            train_lp = TrainLinkPrediction(net, epochs=self.args.embedding_nn_epochs,
                                           train_loader=train_loader, val_loader=val_loader)
            classif2 = train_lp.train()

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
        return dict_class_movie_test, classif2

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
            seen_num = int(self.args.seen_percentage * len(classes))
            unseen_num = len(classes) - int(self.args.seen_percentage * len(classes))
        return seen_num, unseen_num

    def evaluate_for_hist(self, dict_class_movie_test, advantage=1.0, classif=None):
        # evaluate
        advantage1 = advantage if self.args.link_prediction_type == "norm_argmin" else 1.0
        advantage2 = 1.0 if self.args.link_prediction_type == "norm_argmin" else advantage
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
                # class_test = self.edges_distance(edges)
                if self.args.link_prediction_type == "norm_argmin":
                    class_test = self.edges_distance(edges)
                elif self.args.link_prediction_type == "norm_linear_regression":
                    from link_prediction_models import predict_edge_classification
                    class_test = self.edges_distance(edges)
                    probs = -predict_edge_classification(classif, class_test)[1].T[0]
                elif self.args.link_prediction_type == "embedding_neural_network":
                    from link_prediction_models import keras_model_predict, EmbeddingLinkPredictionDataset, \
                        TrainLinkPrediction
                    embed_test = self.edges_embeddings(edges)
                    test_set = EmbeddingLinkPredictionDataset(embed_test, labels=np.zeros(len(embed_test)))
                    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
                    lp = TrainLinkPrediction(classif, epochs=0, test_loader=test_loader)
                    # probs = -keras_model_predict(classif, embed_test)
                    probs = lp.test()
                # else:
                #     for j, edge in enumerate(edges):
                #         norm = self.edge_distance(edge)
                #         class_test[j, 0] = norm
                # _, probs = self.predict_edge_classification(classif2, class_test)
                # pred_index = np.argmax(probs.T[0])
                try:
                    if self.args.link_prediction_type == "norm_argmin":
                        class_norm_test = np.column_stack((np.column_stack((class_test, classes)), classes_flag))
                    else:
                        class_norm_test = np.column_stack((np.column_stack((probs, classes)), classes_flag))
                except:
                    print('a')
                new_order = np.argsort(class_norm_test[:, 0].astype(float))
                sorted_class_norm = class_norm_test[new_order]
                # if set(self.args.norm) == set('cosine'):
                #     sorted_class_norm = np.flip(sorted_class_norm)
                #     sort_classes = sorted_class_norm.T[0]
                # else:
                sort_classes = sorted_class_norm.T[1]
                sort_norm = sorted_class_norm.T[0].astype(float)
                sort_classes_flag = sorted_class_norm.T[2].astype(float)
                # class_test[::-1].sort(axis=0)
                # if self.args.link_prediction_type == "norm_argmin":
                #     c_pred = sort_classes[0]
                # elif self.args.link_prediction_type == "norm_linear_regression"\
                #         or self.args.link_prediction_type == "embedding_neural_network":
                #     pred_index = np.argmax(probs.T[0])
                #     c_pred = edges[pred_index][1]
                c_pred = sort_classes[0]
                prediction = np.array([m, c_pred])
                # prediction = edges[pred_index]
                real_edge = list(dict_class_movie_test[c][m])
                pred_true.append(c)
                if i > seen_num:
                    place = np.where(sort_classes == c)[0][0]
                    hist_real_unseen_pred[place] += 1
                place = np.where(sort_classes_flag == 1)[0][0]
                if advantage1 * sort_norm[place] < advantage2 * sort_norm[0]:
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
        # TODO dict class measures for every ratio
        return dict_class_measures, pred, pred_true, hist_real_unseen_pred

    def hist_plot_for_unseen_dist_eval(self, distances):
        title = 'Histogram Of The Distance Between \n Unseen Label Norm And Predicted Norm'
        x_label = f'Distance, limit:{len(distances)}'
        y_label = 'Count'
        hist_plot(distances, title, x_label, y_label)
        plt.savefig(f'{self.args.dataset}/plots/hist_distance_real_unseen-prediction_'
                    f'{self.embedding}_{self.args.norm}_{int(100 * self.args.seen_percentage)}_seen_percent')
        plt.close()

    def confusion_matrix_maker(self, dict_class_measures, pred, pred_true):
        conf_matrix = confusion_matrix(pred_true, pred, labels=list(dict_class_measures.keys()))
        unseen_conf_matrix = np.zeros(shape=conf_matrix.shape)
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
                unseen_conf_matrix[i][j] = conf_matrix[i][j]
            for j in unseen_idx:
                binary_unseen_true_count += conf_matrix[i][j]
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
        s_o_acc = binary_seen_true_count / seen_count
        u_o_acc = binary_unseen_true_count / unseen_count
        measures = {"seen_accuracy": seen_accuracy, "unseen_accuracy": unseen_accuracy, "harmonic_mean": harmonic_mean_,
                    "seen_only_accuracy": s_o_acc, "unseen_only_accuracy": u_o_acc, "seen_count": seen_count,
                    "unseen_count": unseen_count}
        # seen_unseen_conf_matrix = np.array([[seen_true_count, seen_count - seen_true_count],
        #                                [unseen_count - unseen_true_count, unseen_true_count]])
        binary_conf_matrix = np.array([[binary_seen_true_count, seen_count - binary_seen_true_count],
                                       [unseen_count - binary_unseen_true_count, binary_unseen_true_count]])
        binary_conf_matrix = normalize(binary_conf_matrix, norm="l1")  # to add
        print(f"Total Seen Examples: {seen_count} || Total Unseen Examples: {unseen_count}")
        print(
            f'Seen Accuracy: {seen_accuracy} || Unseen Accuracy: {unseen_accuracy} || Harmonic Mean: {harmonic_mean_}')
        print(f'Seen Only Accuracy: {s_o_acc} || Unseen Only Accuracy: {u_o_acc}')
        return measures, conf_matrix, binary_conf_matrix, unseen_conf_matrix

    def plot_confusion_matrix_all_classes(self, conf_matrix, binary_conf_matrix=None, unseen_conf_matrix=None):
        # title = f'Confusion Matrix, ZSL {self.args.dataset} \n' \
        #         f'{self.embedding} {self.args.norm} {int(100 * self.args.seen_percentage)} Percent Seen'
        # x_title = f"True Labels {int(100 * self.args.seen_percentage)}/{100 - int(100 * self.args.seen_percentage)}" \
        #           f" (seen/unseen)"
        x_title = "True Labels"
        y_title = "Predicted Labels"
        save_path = f'{self.args.dataset}/plots/confusion_matrix_{self.embedding}_{self.args.link_prediction_type}_{self.args.norm}'
        title = None
        conf_matrix = normalize(conf_matrix)
        plot_confusion_matrix(conf_matrix, title, x_title, y_title, save_path, vmax=None, vmin=None)
        if binary_conf_matrix is not None:
            y_binary = "True Seen/Unseen"
            x_binary = "Predicted Seen/Unseen"
            # binary_title = "Binary " + title
            binary_title = None
            save_path_binary = f'{self.args.dataset}/plots/binary_confusion_matrix_{self.embedding}_{self.args.link_prediction_type}_{self.args.norm}'
            plot_confusion_matrix(binary_conf_matrix, binary_title, x_binary, y_binary, save_path_binary, vmax=None,
                                  vmin=None, cmap=None)
        if unseen_conf_matrix is not None:
            y_unseen = "True Seen/Unseen"
            x_unseen = "Predicted Seen/Unseen"
            # unseen_title = "Unseen " + title
            unseen_title = None
            save_path_unseen = f'{self.args.dataset}/plots/unseen_confusion_matrix_{self.embedding}_{self.args.link_prediction_type}_{self.args.norm}'
            plot_confusion_matrix(unseen_conf_matrix, unseen_title, x_unseen, y_unseen, save_path_unseen, vmax=None,
                                  vmin=None)


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
    weights = np.array([params['instance_edges_weight'], params['label_edges_weight']]).astype(float)
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_edges_weight', default=params['instance_edges_weight'])
    parser.add_argument('--label_edges_weight', default=params['label_edges_weight'])
    parser.add_argument('--graph_percentage', default=0.3)
    parser.add_argument('--dataset', dest="dataset", help=' Name of the dataset', type=str,
                        default=params['dataset'])  # our_imdb, awa2, cub, lad
    parser.add_argument('--kg_jacard_similarity_threshold', default=params['kg_jacard_similarity_threshold'])
    parser.add_argument('--embedding', default=params['embedding_type'])  # Node2Vec / Event2Vec / OGRE
    # embedding = params[2]
    parser.add_argument('--ratio', default=[0.8])
    parser.add_argument('--seen_percentage', default=float(params['seen_percentage']))
    parser.add_argument('--embedding_dimension', default=int(params['embedding_dimension']))
    parser.add_argument('--seen_weight_advantage', default=params["seen_advantage"])
    parser.add_argument('--link_prediction_type', default=params["link_prediction_type"])
    if params["embedding_type"] == "OGRE":
        parser.add_argument('--ogre_second_neighbor_advantage', default=params["ogre_second_neighbor_advantage"])
        parser.add_argument('--ogre_initial_graph', default=params["ogre_initial_graph"])
    if params['dataset'] == 'awa2' or params['dataset'] == 'cub' or params['dataset'] == 'lad':
        parser.add_argument("--train_percentage", help="train percentage from the seen images", default=90)

        parser.add_argument('--attributes_edges_weight', default=params['attributes_edges_weight'])

        parser.add_argument('--images_nodes_percentage', default=0.3)
    if params["link_prediction_type"] == "embedding_neural_network":
        parser.add_argument('--false_per_true_edges', default=params["false_per_true_edges"])
        parser.add_argument('--embedding_nn_optimizer', default=params["embedding_nn_optimizer"])
        parser.add_argument('--embedding_nn_lr', default=params["embedding_nn_lr"])
        parser.add_argument('--embedding_nn_weight_decay', default=params["embedding_nn_weight_decay"])
        parser.add_argument('--embedding_nn_dropout_prob', default=params["embedding_nn_dropout_prob"])
        parser.add_argument('--embedding_nn_epochs', default=params["embedding_nn_epochs"])
        parser.add_argument('--embedding_nn_loss', default=params["embedding_nn_loss"])
        parser.add_argument('--norm', default=None)
    elif params["link_prediction_type"] == "norm_linear_regression":
        parser.add_argument('--false_per_true_edges', default=params["false_per_true_edges"])
        parser.add_argument('--score_maximize', default="balanced_accuracy")
        parser.add_argument('--regression_solver', default=params["regression_solver"])
        parser.add_argument('--norm', default=params['norm_type'])  # cosine / L2 Norm / L1 Norm
    elif params["link_prediction_type"] == "norm_argmin":
        parser.add_argument('--norm', default=params['norm_type'])  # cosine / L2 Norm / L1 Norm

    # embedding_dimension = params[3].astype(int)
    args = parser.parse_args()
    params["graph_percentage"] = args.graph_percentage

    return args, weights, params


def obj_func_grid(params, file=None, specific_split=True, split=None, draw=False, classif=None,
                  is_nni=True):  # split False or True
    """
    Main Function for link prediction task.
    :return:
    """
    args, weights, params = define_args(params)
    np.random.seed(0)
    # ratio_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    graph_maker = GraphImporter(args)
    # multi_graph = graph_maker.import_imdb_multi_graph(weights)
    dict_val_edges = None
    if args.dataset == 'our_imdb':
        weighted_graph = graph_maker.import_imdb_weighted_graph(weights)
    elif args.dataset == 'awa2' or args.dataset == 'cub' or args.dataset == 'lad':
        awa2_att_weight = params['attributes_edges_weight']
        weighted_graph, dict_val_edges, split = graph_maker.import_data_graph(weights, specific_split, awa2_att_weight,
                                                                              gephi_display=True)
    else:
        raise ValueError(f"Wrong name of DataSet, {args.dataset}")
    edges_preparation = EdgesPreparation(weighted_graph, dict_val_edges, args, split)
    # dict_true_edges = edges_preparation.label_edges_classes_ordered(edges_preparation.label_edges)
    # dict_false_edges = edges_preparation.make_false_label_edges(dict_true_edges)
    dict_train_true = edges_preparation.dict_train_edges
    dict_test_true = edges_preparation.dict_test_edges
    dict_unseen_edges = edges_preparation.dict_unseen_edges
    if args.link_prediction_type == "embedding_neural_network" or args.link_prediction_type == "norm_linear_regression":
        dict_train_false = edges_preparation.make_false_label_edges(dict_train_true)
    graph = edges_preparation.seen_graph()
    embeddings_maker = EmbeddingCreator(graph, args.embedding_dimension, args)
    if args.embedding == 'Node2Vec':
        dict_embeddings = embeddings_maker.create_node2vec_embeddings()
    elif args.embedding == 'hope':
        dict_embeddings = embeddings_maker.create_hope_embeddings()
    elif args.embedding == 'Event2Vec':
        dict_embeddings = embeddings_maker.create_event2vec_embeddings()
    elif args.embedding == 'OGRE':
        if args.ogre_initial_graph == "label_edges":
            initial_nodes = edges_preparation.ogre_initial_nodes(graph)
        elif args.ogre_initial_graph == "instance_nodes":
            initial_nodes = edges_preparation.graph_without_unseen_classes(graph)
        else:
            raise ValueError("Wrong initial graph key")
        dict_embeddings = embeddings_maker.create_ogre_embeddings(user_initial_nodes_choice=initial_nodes)
    else:
        raise ValueError(f"Wrong name of embedding, {args.embedding}")
    if draw:
        draw_unseen_graph = DrawUnseenGraph(dict_embeddings, dict_train_true, dict_test_true, dict_unseen_edges,
                                            kind="tsne", dataset=args.dataset, args=args)
        draw_unseen_graph.draw_graph()
        draw_unseen_graph = DrawUnseenGraph(dict_embeddings, dict_train_true, dict_test_true, dict_unseen_edges,
                                            kind="pca", dataset=args.dataset, args=args)
        draw_unseen_graph.draw_graph()
    classifier = Classifier(dict_train_true, dict_test_true, dict_unseen_edges,
                            dict_embeddings, args.embedding, split, args)
    if args.link_prediction_type == "norm_argmin":
        dict_class_movie_test = classifier.train()
    else:
        dict_class_movie_test, classif = classifier.ml_train(dict_train_false)
        from graph_ZSL_new import MLClassifier
        classifier1 = MLClassifier(dict_train_true, dict_train_false, dict_test_true, dict_unseen_edges,
                                   dict_embeddings, args.embedding, args, linear_classifier=False)
        # classif, dict_class_movie_test = classifier1.train()
    all_measures = None
    local_max_harmonic_mean = 0
    best_advantage = None
    for advantage in args.seen_weight_advantage:
        # dict_class_measures_node2vec, pred, pred_true = classifier1.evaluate(classif, dict_class_movie_test)
        dict_class_measures_node2vec, pred, pred_true, hist_real_unseen_pred = classifier.evaluate_for_hist(
            dict_class_movie_test, advantage=advantage, classif=classif)
        # # classifier.hist_plot_for_unseen_dist_eval(hist_real_unseen_pred)
        measures, conf_matrix, binary_conf_matrix, unseen_conf_matrix = classifier.confusion_matrix_maker(
            dict_class_measures_node2vec, pred, pred_true)
        table_params = params.copy()
        table_params["seen_weight_advantage"] = advantage
        table_row = config_to_str(table_params)
        if file is not None:
            update_results(file, table_row, measures)
        if all_measures is None:
            all_measures = {key: [measures[key]] for key in list(measures.keys())}
        else:
            [all_measures[key].append(measures[key]) for key in list(measures.keys())]
        if draw:
            classifier.plot_confusion_matrix_all_classes(conf_matrix, binary_conf_matrix, unseen_conf_matrix)

        current_harmonic_mean = measures["harmonic_mean"]
        if is_nni:
            nni.report_intermediate_result(current_harmonic_mean)
        local_max_harmonic_mean, change = replace_max(local_max_harmonic_mean, current_harmonic_mean,
                                                      report_change=True)
        if change:
            best_advantage = advantage
    return all_measures, local_max_harmonic_mean, best_advantage


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


def run_grid(grid_params, res_dir, now, all_measures=None, ignore_params: list = None, add_to_exist_file=False,
             is_nni=True):
    grid_params = grid_params if type(grid_params) is dict else json.load(open(grid_params, "rt"))
    res_filename = os.path.join(res_dir, f"{grid_params['dataset'][0]}_grid_{now}.csv")
    if add_to_exist_file:
        out = open(res_filename, "a")
    else:
        out = open(res_filename, "wt")
    out.write(f"{','.join(HEADER)}\n")
    for config in grid(grid_params, ignore_params=ignore_params):
        config_keys = list(grid_params.keys())
        config_keys.remove(ignore_params[0])
        param = {p: config[i] for i, p in enumerate(config_keys)}
        if ignore_params is not None:
            for p in ignore_params:
                param[p] = grid_params[p]
        all_measures, max_h_m, best_advantage = obj_func_grid(param, out, is_nni=is_nni)
    out.close()
    return all_measures, max_h_m, best_advantage


def update_results(file, table_row, measures):
    table_row[HEADER.index('harmonic_mean')] = str(measures["harmonic_mean"])
    table_row[HEADER.index('seen_acc')] = str(measures["seen_accuracy"])
    table_row[HEADER.index('unseen_acc')] = str(measures["unseen_accuracy"])
    table_row[HEADER.index('seen_only_acc')] = str(measures["seen_only_accuracy"])
    table_row[HEADER.index('unseen_only_acc')] = str(measures["unseen_only_accuracy"])
    table_row[HEADER.index('seen_count')] = str(measures["seen_count"])
    table_row[HEADER.index('unseen_count')] = str(measures["unseen_count"])
    file.write(f"{','.join(table_row)}\n")


def main():
    seen_accuracies, unseen_accuracies = [], []
    parameters = {
        "dataset": ['our_imdb'],  # 'awa2', 'our_imdb'
        "embedding_type": ["Node2Vec"],
        "embedding_dimension": [32, 64, 128, 256],
        # "label_edges_weight": [1],
        # "instance_edges_weight": [1],
        "label_edges_weight": np.logspace(-2, 3, 6),
        "instance_edges_weight": np.logspace(-2, 3, 6),
        "norm_type": ['cosine'],
        "kg_jacard_similarity_threshold": [0.3, 0.6, 0.9],
        "seen_percentage": [0.8],
        # "seen_percentage": np.linspace(0.1, 0.9, 9)
        "attributes_edges_weight": [100],  # 100 is the best for now
        "link_prediction_type": ["norm_argmin"]
    }
    num = 0
    for param in grid(parameters):
        dict_param = {p: param[i] for i, p in enumerate(list(parameters.keys()))}
        # param = np.array([w_m_m, w_m_c, e_type, dim, norma_type, kg_jacard_similarity_threshold, per, data, w_att])
        print(f'iteration number {num}')
        num += 1
        harmonic_mean_, seen_acc, unseen_acc = obj_func_grid(dict_param)
        seen_accuracies.append(seen_acc * 100)
        unseen_accuracies.append(unseen_acc * 100)
        # print("all accuracy: ", acc)
    dict_measures = {"unseen_accuracy": unseen_accuracies, "seen_accuracy": seen_accuracies}
    plots_2measures_vs_parameter(dict_measures, parameters["seen_percentage"], 'seen Percentage', 'our_imdb',
                                 'Zero Shot Learning', "Accuracy", parameters["norm_type"][0],
                                 parameters["embedding_type"][0])


# if __name__ == '__main__':
#     res_dir = "C:\\Users\\kfirs\\lab\\Zero Shot Learning\\New-Graph-ZSL\\grid_results"
#     # now = datetime.now().strftime("%d%m%y_%H%M%S")
#     now = "08_08_21"
#     # parameters = {
#     #     "dataset": ['our_imdb'],  # 'awa2', 'our_imdb'
#     #     "embedding_type": ["Node2Vec"],
#     #     "embedding_dimension": [32, 64, 128, 256],
#     #     # "label_edges_weight": [1],
#     #     # "instance_edges_weight": [1],
#     #     "label_edges_weight": np.logspace(-2, 3, 6),
#     #     "instance_edges_weight": np.logspace(-2, 3, 6),
#     #     "norm_type": ['cosine'],
#     #     "kg_jacard_similarity_threshold": [0.3, 0.6, 0.9],
#     #     "seen_percentage": [0.8],
#     #     # "seen_percentage": np.linspace(0.1, 0.9, 9)
#     #     "attributes_edges_weight": [100]  # 100 is the best for now
#     # }
#     parameters = {
#         "dataset": ['cub'],  # 'our_imdb', 'awa2', 'cub', 'lad'
#         "embedding_type": ["Node2Vec"],  # "Node2Vec", "OGRE", "hope"
#         "embedding_dimension": [128],  # 128
#         "label_edges_weight": [100],  # 30
#         "instance_edges_weight": [100],  # 1
#         # "label_edges_weight": np.logspace(0, 2, 3).astype(int),
#         # "instance_edges_weight": np.logspace(0, 2, 3).astype(int),
#         "norm_type": ['cosine'],  # 'cosine', "L2 Norm", "L1 Norm"
#         "kg_jacard_similarity_threshold": [0.3],
#         "seen_percentage": [0.8],
#         # "seen_advantage": np.linspace(0, 1.0, 11),
#         "seen_advantage": [0.7],
#         # "seen_percentage": np.linspace(0.1, 0.9, 9)
#         "attributes_edges_weight": [1],  # 100 is the best for now
#         # "attributes_edges_weight": np.logspace(0, 2, 3).astype(int),  # 100 is the best for now
#         "link_prediction_type": ["norm_linear_regression"]  # "norm_argmin", "norm_linear_regression", "embedding_neural_network"
#     }
# if "OGRE" in parameters["embedding_type"]:
#     parameters.update({"ogre_second_neighbor_advantage": [0.01]})  # 0.1
#     parameters.update({"ogre_initial_graph": ["instance_nodes"]})  # "label_edges", "instance_nodes"
# if "embedding_neural_network" in parameters["link_prediction_type"] or "norm_linear_regression" in parameters["link_prediction_type"]:
#     from graph_ZSL_new import obj_func_grid as obj
#     parameters.update({"regression_solver": "lbfgs"})  # "lbfgs", "liblinear", "saga"
#     parameters.update({"embedding_nn_lr": 0.001})
#     parameters.update({"embedding_nn_epochs": 10})
#     parameters.update({"embedding_nn_optimizer": "adam"})
#     parameters.update({"embedding_nn_loss": "weighted_binary_cross_entropy"})
#     parameters.update({"embedding_nn_weight_decay": 0.001})
#     parameters.update({"embedding_nn_dropout_prob": 0.5})
#     processes = []
#     parameters_by_procesess = []
#     for data in parameters["dataset"]:
#         # for w_m_c in parameters["label_edges_weight"]:
#         param_by_parameters = parameters.copy()
#         param_by_parameters["dataset"] = [data]
#         # param_by_parameters["label_edges_weight"] = [w_m_c]
#         parameters_by_procesess.append(param_by_parameters)
#     for i in range(len(parameters_by_procesess)):
#         all_measures = run_grid(parameters_by_procesess[i], res_dir, now, ignore_params=["seen_advantage"],
#                                 add_to_exist_file=True)
#         # plots_2measures_vs_parameter(all_measures, parameters["seen_advantage"],
#         #                              relevant_keys=["seen_only_accuracy", "unseen_only_accuracy"],
#         #                              title=f"{parameters_by_procesess[i]['dataset'][0]} Dataset - Seen Advantage Influence",
#         #                              x_title="seen advantage", y_title="accuracy",
#         #                              path=Path(
#         #                                  f"{parameters_by_procesess[i]['dataset'][0]}/plots/Unseen Advantage Influence.png"))
#
#     #     proc = multiprocessing.Process(target=run_grid, args=(parameters_by_procesess[i], res_dir, now, ))
#     #     processes.append(proc)
#     #     proc.start()
#     # for p in processes:
#     #     p.join()


if __name__ == '__main__':
    res_dir = "C:\\Users\\kfirs\\lab\\Zero Shot Learning\\New-Graph-ZSL\\grid_results"
    now = "08_08_21"
    is_nni = False
    if is_nni:
        parameters = nni.get_next_parameter()
    else:
        parameters = {'dataset': 'lad', 'label_edges_weight': 30.65383,
                      'instance_edges_weight': 27.37175, 'kg_jacard_similarity_threshold': 0.3,
                      'seen_percentage': 0.8, 'seen_advantage': 'None', 'attributes_edges_weight': 92.95568,
                      'embedding_type': {'_name': 'OGRE', 'embedding_dimension': 32,
                                         "ogre_second_neighbor_advantage": 0,
                                         "ogre_initial_graph": "label_edges"},
                      'link_prediction_type': {'_name': 'norm_linear_regression', "norm_type": "L2 Norm",
                                               "false_per_true_edges": 13,
                                               "regression_solver": "liblinear"}}
    parameters = nested_nni_to_dict(parameters)
    parameters["seen_advantage"] = np.linspace(0.2, 0.9, 8)
    all_measures, max_harmonic_mean, best_advantage = run_grid(parameters, res_dir, now,
                                                               ignore_params=["seen_advantage"],
                                                               add_to_exist_file=True, is_nni=is_nni)
    nni.report_final_result({'default': max_harmonic_mean, 'best_advantage': best_advantage})
