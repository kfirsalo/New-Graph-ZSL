from itertools import chain

import nni
from node2vec import Node2Vec
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
import argparse
from pathlib import Path

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

# class GraphImporter(object):
#     def __init__(self, data_name):
#         self.data_name = data_name
#
#     def import_imdb_multi_graph(self):
#         path = os.path.join(self.data_name, 'IMDb_multi_graph.gpickle')
#         if os.path.exists(path):
#             multi_gnx = nx.read_gpickle(path)
#         else:
#             from IMDb_data_preparation import main
#             multi_gnx = main()
#             nx.write_gpickle(multi_gnx, path)
#         return multi_gnx
#
#     def import_graph(self):
#         graph = nx.MultiGraph()
#         data_path = self.data_name + '.txt'
#         path = os.path.join(self.data_name, data_path)
#         with open(path, 'r') as f:
#             for line in f:
#                 items = line.strip().split()
#                 att1 = str(items[0][0])
#                 att2 = str(items[1][0])
#                 graph.add_node(items[0], key=att1)
#                 graph.add_node(items[1], key=att2)
#                 sort_att = np.array([att1, att2])
#                 sort_att = sorted(sort_att)
#                 graph.add_edge(items[0], items[1], key=str(sort_att[0]) + str(sort_att[1]))
#         return graph
from IMDb_data_preparation_E2V import MoviesGraph
from utlis_graph_zsl import nested_nni_to_dict, plots_2measures_vs_parameter, grid


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
    def __init__(self, data_name=None, graph=None):
        self.data_name = data_name
        self.graph = graph

    def create_node2vec_embeddings(self):
        path1 = os.path.join(self.data_name, 'Node2Vec_embedding_old.pickle')
        path2 = os.path.join(self.data_name, 'Node2Vec_embedding_old.csv')
        if os.path.exists(path1):
            with open(path1, 'rb') as handle:
                dict_embeddings = pickle.load(handle)
        elif os.path.exists(path2):
            embedding_df = pd.read_csv(path2)
            dict_embeddings = embedding_df.to_dict(orient='list')
            with open(path2, 'wb') as handle:
                pickle.dump(dict_embeddings, handle, protocol=3)
        else:
            node2vec = Node2Vec(self.graph, dimensions=128, walk_length=80, num_walks=16, workers=2)
            model = node2vec.fit()
            nodes = list(self.graph.nodes())
            dict_embeddings = {}
            for i in range(len(nodes)):
                dict_embeddings.update({nodes[i]: np.asarray(model.wv.get_vector(nodes[i]))})
            with open(path1, 'wb') as handle:
                pickle.dump(dict_embeddings, handle, protocol=3)
        return dict_embeddings

    def create_event2vec_embeddings(self):
        data_path = self.data_name + '_e2v_embeddings.txt'
        path = os.path.join(self.data_name, data_path)
        cond = 0
        dict_embeddings = {}
        with open(path, 'r') as f:
            for line in f:
                if cond == 1:
                    items = line.strip().split()
                    dict_embeddings[items[0]] = items[1:]
                cond = 1
        return dict_embeddings


"""
Link prediction task for evaluation, as explained in the pdf file. Initialize the methods you want to compare in
static_embeddings.py file, full explanation in their. if initial_method == "hope" and method == "directed_node2vec",
link prediction will be applied on these 2 embedding methods and will compare between them. Each combination is legal.
"""

try:
    import cPickle as pickle
except:
    import pickle
from numpy import linalg as LA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import model_selection as sk_ms, metrics
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.linear_model import LogisticRegression as lr
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

# from state_of_the_art.state_of_the_art_embedding import *


# for plots that will come later
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 14


def read_file(X, G):
    """
    Read a txt file of embedding and get the embedding
    :param X: The file after np.loadtxt
    :param G: The graph
    :return: A dictionary where keys==nodes and values==emmbedings
    """
    nodes = list(G.nodes())
    my_dict = {}
    for i in range(len(nodes)):
        my_dict.update({nodes[i]: X[i]})
    return my_dict


def choose_true_edges(G, K, times):
    """
    Randomly choose a fixed number of existing edges
    :param edges: The graph's edges
    :param K: Fixed number of edges to choose
    :return: A list of K true edges
    """
    edges = list(G.edges)
    random_ind = random.sample(range(1, len(edges)), times * K)
    true_edges = []
    for j in range(times):
        true_edges_j = []
        indexes = np.arange(j * K, (j + 1) * K)
        for i in indexes:
            true_edges_j.append([edges[random_ind[i]][0], edges[random_ind[i]][1]])
        true_edges.append(true_edges_j)
    return true_edges


def choose_false_edges(G, K, data_name, times, in_prob):
    """
    Randomly choose a fixed number of non-existing edges
    :param G: Our graph
    :param K: Fixed number of edges to choose
    :return: A list of K false edges
    """
    data_path = data_name + '_non_edges.pickle'
    if os.path.exists(os.path.join(data_name, data_path)):
        with open(os.path.join(data_name, data_path), 'rb') as handle:
            non_edges = pickle.load(handle)
        # with open('pkl_e2v/dblp_non_edges_old.pickle', 'wb') as handle:
        #     pickle.dump(non_edges, handle, protocol=3)
    else:
        if in_prob is True:
            # times = int(1/(1-ratio)) * K
            relevant_edges = list(G.edges)
            false_edges = []
            for time in range(times * K):
                is_edge = True
                while is_edge:
                    indexes = random.sample(range(1, len(relevant_edges)), 2)
                    node_1 = relevant_edges[indexes[0]][0]
                    node_2 = relevant_edges[indexes[1]][1]
                    if node_2 != relevant_edges[indexes[0]][1]:
                        is_edge = False
                        false_edges.append([node_1, node_2])
            non_edges = false_edges
        else:
            non_edges = list(nx.non_edges(G))
            false_edges = []
            indexes = random.sample(range(1, len(non_edges)), times * K)
            for j in indexes:
                false_edges.append([non_edges[j][0], non_edges[j][1]])
        try:
            with open(os.path.join(data_name, data_path), 'wb') as handle:
                pickle.dump(false_edges, handle, protocol=3)
        except:
            pass
    # indexes = random.sample(range(1, len(non_edges)), K)
    # false_edges = []
    # for i in indexes:
    #     false_edges.append([non_edges[i][0], non_edges[i][1]])
    false_edges = []
    for j in range(times):
        false_edges_j = []
        indexes = np.arange(j * K, (j + 1) * K)
        for i in indexes:
            false_edges_j.append([non_edges[i][0], non_edges[i][1]])
        false_edges.append(false_edges_j)
    return false_edges


def calculate_classifier_value(dict_projections, true_edges, false_edges, K, norma):
    """
    Create X and Y for Logistic Regression Classifier.
    :param dict_projections: A dictionary of all nodes emnbeddings, where keys==nodes and values==embeddings
    :param true_edges: A list of K false edges
    :param false_edges: A list of K false edges
    :param K: Fixed number of edges to choose
    :return: X - The feature matrix for logistic regression classifier. Its size is 2K,1 and the the i'th row is the
                norm score calculated for each edge, as explained in the attached pdf file.
            Y - The edges labels, 0 for true, 1 for false
    """
    X = np.zeros(shape=(2 * K, 1))
    Y = np.zeros(shape=(2 * K, 2))
    my_dict = {}
    count = 0
    for edge in true_edges:
        embd1 = np.array(dict_projections[edge[0]]).astype(float)
        embd2 = np.array(dict_projections[edge[1]]).astype(float)
        if set(norma) == set('L1 Norm'):
            norm = LA.norm(np.subtract(embd1, embd2), 1)
        elif set(norma) == set('L2 Norm'):
            norm = LA.norm(np.subtract(embd1, embd2), 1)
        elif set(norma) == set('cosine'):
            norm = cosine_similarity(embd1.reshape(1, -1), embd2.reshape(1, -1))[0]
        X[count, 0] = norm
        Y[count, 0] = int(1)
        # my_dict.update({edge: [count, norm, int(0)]})
        count += 1
    for edge in false_edges:
        embd1 = np.array(dict_projections[edge[0]]).astype(float)
        embd2 = np.array(dict_projections[edge[1]]).astype(float)
        if set(norma) == set('L1 Norm'):
            norm = LA.norm(np.subtract(embd1, embd2), 1)
        elif set(norma) == set('L2 Norm'):
            norm = LA.norm(np.subtract(embd1, embd2), 1)
        elif set(norma) == set('cosine'):
            norm = cosine_similarity(embd1.reshape(1, -1), embd2.reshape(1, -1))[0]
        X[count, 0] = norm
        Y[count, 1] = int(1)
        # my_dict.update({edge: [count, norm, int(1)]})
        count += 1
    return my_dict, X, Y


class TopKRanker(oneVr):
    """
    Linear regression with one-vs-rest classifier
    """

    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = super(TopKRanker, self).predict_proba(X)
        # probs = np.asarray()
        prediction = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
            for label in labels:
                prediction[i, int(label)] = 1
        return prediction, probs


def evaluate_edge_classification(prediction, Y_test):
    """
    Predictions of nodes' labels.
    :param X: The features' graph- norm
    :param Y: The edges labels- 0 for true, 1 for false
    :param test_ratio: To determine how to split the data into train and test
    :return: Scores- F1-macro, F1-micro accuracy and auc
    """
    accuracy = accuracy_score(Y_test, prediction)
    micro = f1_score(Y_test, prediction, average='micro')
    macro = f1_score(Y_test, prediction, average='macro')
    auc = roc_auc_score(Y_test, prediction)
    precision = precision_score(Y_test, prediction, average='micro')
    return micro, macro, accuracy, auc, precision


def predict_edge_classification(X_train, X_test, Y_train, Y_test):
    """
    Predictions of nodes' labels.
    :param X: The features' graph- norm
    :param Y: The edges labels- 0 for true, 1 for false
    :param test_ratio: To determine how to split the data into train and test
    :return: Scores- F1-macro, F1-micro accuracy and auc
    """
    try:
        top_k_list = list(Y_test.toarray().sum(axis=1))
    except:
        top_k_list = list(Y_test.sum(axis=1))
    classif2 = TopKRanker(lr())
    classif2.fit(X_train, Y_train)
    prediction, probs = classif2.predict(X_test, top_k_list)
    return prediction, probs


def exp_lp(X, Y, test_ratio_arr, rounds):
    """
    The final node classification task as explained in our git.
    :param X: The features' graph- norm
    :param Y: The edges labels- 0 for true, 1 for false
    :param test_ratio_arr: To determine how to split the data into train and test. This an array
                with multiple options of how to split.
    :param rounds: How many times we're doing the mission. Scores will be the average.
    :return: Scores for all splits and all splits- F1-micro, F1-macro accuracy and auc
    """
    micro = [None] * rounds
    macro = [None] * rounds
    acc = [None] * rounds
    auc = [None] * rounds

    for round_id in range(rounds):
        micro_round = [None] * len(test_ratio_arr)
        macro_round = [None] * len(test_ratio_arr)
        acc_round = [None] * len(test_ratio_arr)
        auc_round = [None] * len(test_ratio_arr)
        for i, test_ratio in enumerate(test_ratio_arr):
            micro_round[i], macro_round[i], acc_round[i], auc_round[i] = predict_edge_classification(X, Y, test_ratio)

        micro[round_id] = micro_round
        macro[round_id] = macro_round
        acc[round_id] = acc_round
        auc[round_id] = auc_round

    micro = np.asarray(micro)
    macro = np.asarray(macro)
    acc = np.asarray(acc)
    auc = np.asarray(auc)

    return micro, macro, acc, auc


# def compute_precision_curve(Y, Y_test, true_digraph, k):
#     precision_scores = []
#     delta_factors = []
#     correct_edge = 0
#     for i in range(k):
#         if true_digraph.has_edge(sorted_edges[i][0], sorted_edges[i][1]):
#             correct_edge += 1
#             delta_factors.append(1.0)
#         else:
#             delta_factors.append(0.0)
#         precision_scores.append(1.0 * correct_edge / (i + 1))
#     return precision_scores, delta_factors


def calculate_avg_score(score, rounds):
    """
    Given the lists of scores for every round of every split, calculate the average score of every split.
    :param score: F1-micro / F1-macro / Accuracy / Auc
    :param rounds: How many times the experiment has been applied for each split.
    :return: Average score for every split
    """
    all_avg_scores = []
    for i in range(score.shape[1]):
        avg_score = (np.sum(score[:, i])) / rounds
        all_avg_scores.append(avg_score)
    return all_avg_scores


def calculate_all_avg_scores(micro, macro, acc, auc, rounds):
    """
    For all scores calculate the average score for every split. The function returns list for every
    score type- 1 for cheap node2vec and 2 for regular node2vec.
    """
    all_avg_micro = calculate_avg_score(micro, rounds)
    all_avg_macro = calculate_avg_score(macro, rounds)
    all_avg_acc = calculate_avg_score(acc, rounds)
    all_avg_auc = calculate_avg_score(auc, rounds)
    return all_avg_micro, all_avg_macro, all_avg_acc, all_avg_auc


def do_graph_split(avg_score1, avg_score2, test_ratio_arr, top, bottom, score, i):
    """
    Plot a graph of the score as a function of the test split value.
    :param avg_score1: list of average scores for every test ratio, 1 for cheap node2vec.
    :param avg_score2: list of average scores for every test ratio, 2 for regular node2vec.
    :param test_ratio_arr: list of the splits' values
    :param top: top limit of y axis
    :param bottom: bottom limit of y axis
    :param score: type of score (F1-micro / F1-macro / accuracy/ auc)
    :return: plot as explained above
    """
    fig = plt.figure(i)
    plt.plot(test_ratio_arr, avg_score1, '-ok', color='blue')
    plt.plot(test_ratio_arr, avg_score2, '-ok', color='red')
    plt.legend(['cheap node2vec', 'regular node2vec'], loc='upper left')
    plt.ylim(bottom=bottom, top=top)
    # plt.title("Pubmed2 dataset")
    plt.xlabel("test ratio")
    plt.ylabel(score)
    return fig


def split_vs_score(avg_micro1, avg_macro1, avg_micro2, avg_macro2, avg_acc1, avg_acc2, avg_auc1, avg_auc2,
                   test_ratio_arr):
    """
    For every type of score plot the graph as explained above.
    """
    # you can change borders as you like
    fig1 = do_graph_split(avg_micro1, avg_micro2, test_ratio_arr, 1, 0, "micro-F1 score", 1)
    fig2 = do_graph_split(avg_macro1, avg_macro2, test_ratio_arr, 1, 0, "macro-F1 score", 2)
    fig3 = do_graph_split(avg_acc1, avg_acc2, test_ratio_arr, 1, 0, "accuracy", 3)
    fig4 = do_graph_split(avg_auc1, avg_auc2, test_ratio_arr, 1, 0, "auc", 4)
    return fig1, fig2, fig3, fig4


def edges_to_predict(multi_graph):
    edges = list(multi_graph.edges.data(keys=True))
    start = False
    relevant_edges = None
    for edge in edges:
        if edge[3]['key'] == 'labels_edges':
            if edge[0][0] == 't':
                node_1 = edge[0]
                node_2 = edge[1]
            else:
                node_1 = edge[1]
                node_2 = edge[0]
            if start:
                relevant_edges = np.append(relevant_edges, np.array([[node_1, node_2]]), axis=0)
            else:
                relevant_edges = np.array([[node_1, node_2]])
            start = True
    return relevant_edges


#
# def choose_true_edges(relevant_edges, K):
#     """
#     Randomly choose a fixed number of existing edges
#     :param edges: The graph's edges
#     :param K: Fixed number of edges to choose
#     :return: A list of K true edges
#     """
#     indexes = random.sample(range(1, len(relevant_edges)), K)
#     true_edges = []
#     for i in indexes:
#         true_edges.append([relevant_edges[i][0], relevant_edges[i][1]])
#     return true_edges
#
#
# def choose_false_edges(G, relevant_edges, K):
#     """
#     Randomly choose a fixed number of non-existing edges
#     :param G: Our graph
#     :param K: Fixed number of edges to choose
#     :return: A list of K false edges
#     """
#     times = 5 * K
#     false_edges = []
#     for time in range(times):
#         is_edge = True
#         while is_edge:
#             indexes = random.sample(range(1, len(relevant_edges)), 2)
#             node_1 = relevant_edges[indexes[0]][0]
#             node_2 = relevant_edges[indexes[1]][1]
#             if node_2 != relevant_edges[indexes[0]][1]:
#                 is_edge = False
#                 false_edges.append([node_1, node_2])
#     return false_edges
def compute_final_measures(true_edges, false_edges, dict_embeddings, ratio, number, times, norm):
    all_micro, all_macro, all_acc, all_auc, all_precision = [], [], [], [], []
    mean_acc, mean_auc, mean_micro, mean_macro, mean_precision = [], [], [], [], []
    std_acc, std_auc, std_micro, std_macro, std_precision = [], [], [], [], []
    dict_measures = {}
    for j in range(len(ratio)):
        for i in range(times):
            my_dict, X, Y = calculate_classifier_value(dict_embeddings, true_edges[i], false_edges[i], number, norm)
            X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(X, Y, test_size=1 - ratio[j])
            prediction, probs = predict_edge_classification(X_train, X_test, Y_train, Y_test)

            micro, macro, acc, auc, precision = evaluate_edge_classification(prediction, Y_test)
            # auc, precision = random.uniform(0, 1), random.uniform(0, 1)
            # micro, macro, acc= evaluate_node_classification(prediction, Y_test)
            all_acc.append(acc)
            all_auc.append(auc)
            all_micro.append(micro)
            all_macro.append(macro)
            all_precision.append(precision)
        mean_acc.append(np.mean(np.array(all_acc)))
        mean_auc.append(np.mean(np.array(all_auc)))
        mean_micro.append(np.mean(np.array(all_micro)))
        mean_macro.append(np.mean(np.array(all_macro)))
        mean_precision.append(np.mean(np.array(all_precision)))
        std_acc.append(np.std(np.array(all_acc)))
        std_auc.append(np.std(np.array(all_auc)))
        std_micro.append(np.std(np.array(all_micro)))
        std_macro.append(np.std(np.array(all_macro)))
        std_precision.append(np.std(np.array(all_precision)))
    dict_measures['Accuracy'] = mean_acc
    dict_measures['AUC'] = mean_auc
    dict_measures['Micro-f1'] = mean_micro
    dict_measures['Macro-f1'] = mean_macro
    dict_measures['Precision'] = mean_precision
    dict_measures['std_acc'] = std_acc
    dict_measures['std_auc'] = std_auc
    dict_measures['std_micro'] = std_micro
    dict_measures['std_macro'] = std_macro
    dict_measures['std_precision'] = std_precision
    return dict_measures


def plot_roc_curve(true_edges, false_edges, dict_embeddings, number, norm):
    my_dict, X, Y = calculate_classifier_value(dict_embeddings, true_edges[0], false_edges[0], number, norm)
    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(X, Y, test_size=0.2)
    prediction, probs = predict_edge_classification(X_train, X_test, Y_train, Y_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(Y_test[:, 1], preds)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    plt.rcParams["font.family"] = "Times New Roman"
    plt.title('ROC Curve of Link Prediction Task')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("lad/plots/link_prediction_roc_curve.pdf")


import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams["font.family"] = "Times New Roman"


def plots_maker(dict_measures, ratio_arr, measure, data_name, number, norm):
    x_axis = np.array(ratio_arr)
    task = 'Link Prediction'
    bottom = 0.5
    top = 0.95
    keys = list(dict_measures.keys())
    plt.figure(figsize=(7, 6))
    for j in range(len(keys)):
        if 'event2vec' in keys[j]:
            color = 'red'
            marker = 'o'
            markersize = 8
            linestyle = 'solid'
            y_axis = dict_measures[keys[j]][measure]
        elif "node2vec" in keys[j]:
            color = 'green'
            marker = 's'
            markersize = 6
            linestyle = 'solid'
            y_axis = dict_measures[keys[j]][measure]
        plt.plot(x_axis, y_axis, marker=marker, linestyle=linestyle, markersize=markersize, color=color)
    plt.plot(x_axis, [0.58, 0.63, 0.74, 0.8, 0.83, 0.86, 0.88, 0.9, 0.93], marker='o', linestyle='dashed', markersize=8,
             color='red')
    plt.plot(x_axis, [0.53, 0.54, 0.58, 0.61, 0.64, 0.66, 0.67, 0.68, 0.81], marker='s', linestyle='dashed',
             markersize=6, color='green')
    keys = ['our_node2vec', 'our_event2vec', 'event2vec', 'node2vec']
    plt.ylim(bottom=bottom, top=top)
    plt.legend(keys, loc='best', ncol=3, fontsize='large')
    plt.title("{} Dataset \n {} Task - {} Score".format(data_name, task, measure))
    plt.xlabel("Percentage")
    plt.ylabel("{} ({})".format(measure, norm))
    plt.tight_layout()
    plt.savefig(os.path.join(data_name, "plots", "{} {} {} {} {}.png".format(data_name, task, measure, norm, number)))
    plt.show()


def lp_roc(params):
    """
    Main Function for link prediction task.
    :return:
    """
    all_micro = []
    all_macro = []
    all_acc = []
    all_auc = []
    number = 1000
    times = 1
    ratio_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    args, weights, params = define_args(params)
    graph_maker = GraphImporter(args)
    weights = np.array([params['instance_edges_weight'], params['label_edges_weight']]).astype(float)
    if args.dataset == 'our_imdb':
        graph = graph_maker.import_imdb_weighted_graph(weights)
    elif args.dataset == 'awa2' or args.dataset == 'cub' or args.dataset == 'lad':
        awa2_att_weight = params['attributes_edges_weight']
        specific_split = True
        graph, dict_val_edges, split = graph_maker.import_data_graph(weights, specific_split, awa2_att_weight,
                                                                     gephi_display=True)

    # nodes = graph.nodes()
    # indexes = np.linspace(0, len(nodes)-1, 5000)
    # indexes = indexes.astype(int)
    # relevant_nodes = np.array(nodes)[indexes]
    # graph = nx.subgraph(graph, relevant_nodes)

    # dict_event2vec_embeddings = embedding_model.create_event2vec_embeddings()
    # nodes = list(dict_event2vec_embeddings.keys())
    # relevant_edges = edges_to_predict(multi_graph)
    # true_edges = choose_true_edges(relevant_edges, number)
    # false_edges = choose_false_edges(multi_graph, relevant_edges, number)
    true_edges = choose_true_edges(graph, number, times)
    for edge in true_edges[0]:
        graph.remove_edge(edge[0], edge[1])
    false_edges = choose_false_edges(graph, number, args.dataset, times, in_prob=True)
    embeddings_maker = EmbeddingCreator(args.dataset, graph)
    # dict_embeddings_event2vec = embeddings_maker.create_event2vec_embeddings()
    dict_embeddings_node2vec = embeddings_maker.create_node2vec_embeddings()
    plot_roc_curve(true_edges, false_edges, dict_embeddings_node2vec, number, args.norm)
    # dict_measures_event2vec = compute_final_measures(true_edges, false_edges, dict_embeddings_event2vec, ratio_arr, number, times, args.norm)
    # dict_measures_node2vec = compute_final_measures(true_edges, false_edges, dict_embeddings_node2vec, ratio_arr,
    #                                                 number, times, args.norm)
    dict_measures = {}
    # dict_measures['node2vec'] = dict_measures_node2vec
    # dict_measures['event2vec'] = dict_measures_event2vec
    # plots_maker(dict_measures, ratio_arr, 'AUC', args.data_name.upper(), number, args.norm)
    # print('avg acc e2v: ', dict_measures_event2vec['Accuracy'])
    # print('avg auc e2v: ', dict_measures_event2vec['AUC'])
    # print('avg micro e2v: ', dict_measures_event2vec['Micro-f1'])
    # print('avg macro e2v: ', dict_measures_event2vec['Macro-f1'])
    # print('std acc e2v: ', dict_measures_event2vec['std_acc'])
    # print('std auc e2v: ', dict_measures_event2vec['std_auc'])
    # print('std micro e2v: ', dict_measures_event2vec['std_micro'])
    # print('std macro e2v: ', dict_measures_event2vec['std_macro'])
    # print('avg acc n2v: ', dict_measures_node2vec['Accuracy'])
    # print('avg auc n2v: ', dict_measures_node2vec['AUC'])
    # print('avg micro n2v: ', dict_measures_node2vec['Micro-f1'])
    # print('avg macro n2v: ', dict_measures_node2vec['Macro-f1'])
    # print('std acc n2v: ', dict_measures_node2vec['std_acc'])
    # print('std auc n2v: ', dict_measures_node2vec['std_auc'])
    # print('std micro n2v: ', dict_measures_node2vec['std_micro'])
    # print('std macro n2v: ', dict_measures_node2vec['std_macro'])
    # dict_embeddings = embeddings_maker.create_node2vec_embeddings()
    # micro, macro, acc, auc = exp_lp(X, Y, ratio_arr, 3)
    # avg_micro, avg_macro, avg_acc, avg_auc = calculate_all_avg_scores(micro, macro, acc, auc, 3)
    # all_micro.append(avg_micro)
    # all_macro.append(avg_macro)
    # all_acc.append(avg_acc)
    # all_auc.append(avg_auc)
    # fig1, fig2, fig3, fig4 = split_vs_score(all_micro[0], all_macro[0], all_micro[1], all_macro[1], all_acc[0],
    #                                         all_acc[1], all_auc[0], all_auc[1], ratio_arr)
    # plt.show()


def define_args(params):
    print(params)
    weights = np.array([params['instance_edges_weight'], params['label_edges_weight']]).astype(float)
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_edges_weight', default=params['instance_edges_weight'])
    parser.add_argument('--label_edges_weight', default=params['label_edges_weight'])
    parser.add_argument('--graph_percentage', default=0.05)
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

        parser.add_argument('--images_nodes_percentage', default=0.05)
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


def run_grid(grid_params, ignore_params: list = None):
    for config in grid(grid_params, ignore_params=ignore_params):
        config_keys = list(grid_params.keys())
        config_keys.remove(ignore_params[0])
        param = {p: config[i] for i, p in enumerate(config_keys)}
        if ignore_params is not None:
            for p in ignore_params:
                param[p] = grid_params[p]
        lp_roc(param)


def update_results(file, table_row, measures):
    table_row[HEADER.index('harmonic_mean')] = str(measures["harmonic_mean"])
    table_row[HEADER.index('seen_acc')] = str(measures["seen_accuracy"])
    table_row[HEADER.index('unseen_acc')] = str(measures["unseen_accuracy"])
    table_row[HEADER.index('seen_only_acc')] = str(measures["seen_only_accuracy"])
    table_row[HEADER.index('unseen_only_acc')] = str(measures["unseen_only_accuracy"])
    table_row[HEADER.index('seen_count')] = str(measures["seen_count"])
    table_row[HEADER.index('unseen_count')] = str(measures["unseen_count"])
    file.write(f"{','.join(table_row)}\n")


if __name__ == '__main__':
    res_dir = "C:\\Users\\kfirs\\lab\\Zero Shot Learning\\New-Graph-ZSL\\grid_results"
    now = "08_08_21"
    is_nni = False
    if is_nni:
        parameters = nni.get_next_parameter()
    else:
        parameters = {'dataset': 'lad', 'label_edges_weight': 49,
                      'instance_edges_weight': 97.413, 'kg_jacard_similarity_threshold': 0.3,
                      'seen_percentage': 0.8, 'seen_advantage': 'None', 'attributes_edges_weight': 19.36,
                      'embedding_type': {'_name': 'Node2Vec', 'embedding_dimension': 64},
                      'link_prediction_type': {'_name': 'norm_argmin', "norm_type": "L2 Norm"}}
        # parameters = {'dataset': 'lad', 'label_edges_weight': 86.83,
        #               'instance_edges_weight': 65.26, 'kg_jacard_similarity_threshold': 0.3,
        #               'seen_percentage': 0.8, 'seen_advantage': 'None', 'attributes_edges_weight': 31.49,
        #               'embedding_type': {'_name': 'OGRE', 'embedding_dimension': 32,
        #                                  "ogre_second_neighbor_advantage": 0.01, "ogre_initial_graph": "instance_nodes"},
        #               'link_prediction_type': {'_name': 'norm_argmin', "norm_type": "L2 Norm"}}
    parameters = nested_nni_to_dict(parameters)
    parameters["seen_advantage"] = [0.6]
    # parameters["seen_advantage"] = np.linspace(0.2, 0.9, 8)

    run_grid(parameters, ignore_params=["seen_advantage"])
