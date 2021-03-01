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
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.linear_model import LogisticRegression
import random
import matplotlib as mpl
import matplotlib.pyplot as plt


class GraphImporter(object):
    def __init__(self, args):
        self.data_name = args.data_name

    def import_imdb_multi_graph(self):
        path = os.path.join(self.data_name, 'IMDb_multi_graph.gpickle')
        if os.path.exists(path):
            multi_gnx = nx.read_gpickle(path)
        else:
            from IMDb_data_preparation import main
            multi_gnx = main()
            nx.write_gpickle(multi_gnx, path)
        return multi_gnx

    def import_graph(self):
        graph = nx.MultiGraph()
        data_path = self.data_name + '.txt'
        path = os.path.join(self.data_name, data_path)
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


class EmbeddingCreator(object):
    def __init__(self, graph=None, args=None):
        self.data_name = args.data_name
        self.graph = graph

    def create_node2vec_embeddings(self):
        path1 = os.path.join(self.data_name, 'Node2Vec_embedding.pickle')
        path2 = os.path.join(self.data_name, 'Node2Vec_embedding.csv')
        if os.path.exists(path1):
            with open(path1, 'rb') as handle:
                dict_embeddings = pickle.load(handle)
        elif os.path.exists(path2):
            embedding_df = pd.read_csv(path2)
            dict_embeddings = embedding_df.to_dict(orient='list')
            with open(path2, 'wb') as handle:
                pickle.dump(dict_embeddings, handle, protocol=3)
        else:
            node2vec = Node2Vec(self.graph, dimensions=16, walk_length=30, num_walks=200, workers=1)
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


class TopKRanker(oneVr):
    """
    Linear regression with one-vs-rest classifier
    """

    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = super(TopKRanker, self).predict_proba(X)
        prediction = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
            for label in labels:
                prediction[i, int(label)] = 1
        return prediction, probs


class EdgesPreparation(object):
    def __init__(self, graph, args):
        self.args = args
        self.graph = graph
        self.label_edges = self.make_label_edges()

    def make_label_edges(self):
        """
        Make a list with all the edge from type "labels_edges", i.e. edges between a movie and its class.
        :return: list with labels_edges
        """
        data_path = self.args.data_name + '_true_edges.pickle'
        if os.path.exists(os.path.join(self.args.data_name, data_path)):
            with open(os.path.join(self.args.data_name, data_path), 'rb') as handle:
                label_edges = pickle.load(handle)
        else:
            nodes = list(self.graph.nodes)
            label_edges = []
            for node in nodes:
                info = self.graph._adj[node]
                neighs = list(info.keys())
                for neigh in neighs:
                    if info[neigh][0]['key'] == 'labels_edges':
                        label_edges.append([node, neigh])
            try:
                with open(os.path.join(self.args.data_name, data_path), 'wb') as handle:
                    pickle.dump(label_edges, handle, protocol=3)
            except:
                pass
        return label_edges

    def label_edges_classes_ordered(self):
        """
        Make a dict of classes and their labels_edges they belong to. For every label_edge
        there is only one class it belongs to.
        :return: a dict of classes and their labels_edges
        """
        dict_class_label_edge = {}
        for edge in self.label_edges:
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

    def unseen_edges(self):
        unseen_edges = []
        dict_true_edges = self.label_edges_classes_ordered()
        classes = list(dict_true_edges.keys())
        for i, k in enumerate(sorted(dict_true_edges, key=lambda k: len(dict_true_edges[k]), reverse=True)):
            classes[i] = k
        unseen_classes = classes[int(0.8 * len(classes)):]
        for c in unseen_classes:
            unseen_edges.append(dict_true_edges[c])
        return unseen_edges

    def seen_graph(self):
        unseen_edges = self.unseen_edges()
        graph = self.graph
        for edge in unseen_edges:
            graph.remove_edge(edge[0][0], edge[0][1])
        return graph

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
        data_path = self.args.data_name + '_false_edges_balanced_{}.pickle'.format(self.args.false_per_true)
        if os.path.exists(os.path.join(self.args.data_name, data_path)):
            with open(os.path.join(self.args.data_name, data_path), 'rb') as handle:
                dict_class_false_edges = pickle.load(handle)
        else:
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
                        false_labels = list(set(labels) - set(label))
                    else:
                        false_labels = list(set(false_labels) - set(label))
                    indexes = random.sample(range(1, len(false_labels)), self.args.false_per_true)
                    for i, index in enumerate(indexes):
                        if dict_class_false_edges.get(label) is None:
                            dict_class_false_edges[label] = [[movie, false_labels[index]]]
                        else:
                            edges = dict_class_false_edges[label]
                            edges.append([movie, false_labels[index]])
                            dict_class_false_edges[label] = edges
                    false_labels = list(np.delete(np.array(false_labels), indexes))
            try:
                with open(os.path.join(self.args.data_name, data_path), 'wb') as handle:
                    pickle.dump(dict_class_false_edges, handle, protocol=3)
            except:
                pass
        return dict_class_false_edges


class Classifier(object):
    def __init__(self, dict_true_edges, dict_false_edges, dict_projections, args):
        self.args = args
        self.dict_true_edges = dict_true_edges
        self.dict_false_edges = dict_false_edges
        self.norm = set(args.norm)
        self.dict_projections = dict_projections

    def edge_distance(self, edge):
        """
        Calculate the distance of an edge. Take the vertices of the edge and calculate the distance between their
        embeddings.
        We use to calculate The distance with L1, l2, Cosine Similarity.
        :param edge: the edge we want to find its distance.
        :return: The distance
        """
        embd1 = np.array(self.dict_projections[edge[0]]).astype(float)
        embd2 = np.array(self.dict_projections[edge[1]]).astype(float)
        if self.norm == set('L1 Norm'):
            norm = la.norm(np.subtract(embd1, embd2), 1)
        elif self.norm == set('L2 Norm'):
            norm = la.norm(np.subtract(embd1, embd2), 1)
        elif self.norm == set('cosine'):
            norm = cosine_similarity(embd1.reshape(1, -1), embd2.reshape(1, -1))[0]
        return norm

    def calculate_classifier_value(self, true_edges, false_edges):
        """
        Create X and Y for Logistic Regression Classifier.
        self.dict_projections: A dictionary of all nodes embeddings, where keys==nodes and values==embeddings
        :param true_edges: A list of true edges.
        :param false_edges: A list of false edges.
        :return: X_true/X_false - The feature matrix for logistic regression classifier, of true/false edge.
        The i'th row is the norm score calculated for each edge.
                Y_true_edge/Y_false_edge - The edges labels, [1,0] for true/ [0,1] for false.
                Also the edge of the label is concatenate to the label.
        """
        X_true, X_false = np.zeros(shape=(len(true_edges), 1)), np.zeros(shape=(len(false_edges), 1))
        Y_true_edge, Y_false_edge = np.zeros(shape=(len(true_edges), 4)).astype(int).astype(str), \
                                    np.zeros(shape=(len(false_edges), 4)).astype(int).astype(str)
        for i, edge in enumerate(true_edges):
            norm = self.edge_distance(edge)
            X_true[i, 0] = norm
            Y_true_edge[i, 2] = edge[0]
            Y_true_edge[i, 3] = edge[1]
            Y_true_edge[i, 0] = str(1)
        for i, edge in enumerate(false_edges):
            norm = self.edge_distance(edge)
            X_false[i, 0] = norm
            Y_false_edge[i, 2] = edge[0]
            Y_false_edge[i, 3] = edge[1]
            Y_false_edge[i, 1] = str(1)
        return X_true, X_false, Y_true_edge, Y_false_edge

    @staticmethod
    def train_edge_classification(X_train, Y_train):
        """
        train  the classifier with the train set.
        :param X_train: The features' edge- norm (train set).
        :param Y_train: The edges labels- 0 for true, 1 for false (train set).
        :return: The classifier
        """
        classif2 = TopKRanker(LogisticRegression())
        classif2.fit(X_train, Y_train)
        return classif2

    @staticmethod
    def split_data(X_true, X_false, Y_true_edge, Y_false_edge, ratio):
        """
        split the data into rain and test for the true edges and the false one.
        :param ratio: determine the train size.
        :return: THe split data
        """
        X_train_true, X_test_true, Y_train_true_edge, \
        Y_test_true_edge = sk_ms.train_test_split(X_true, Y_true_edge, test_size=1 - ratio)
        X_train_false, X_test_false, Y_train_false_edge, \
        Y_test_false_edge = sk_ms.train_test_split(X_false, Y_false_edge, test_size=1 - ratio)
        true_edges_test_source = Y_test_true_edge.T[2].reshape(-1, 1)
        true_edges_test_target = Y_test_true_edge.T[3].reshape(-1, 1)
        X_train, X_test, Y_train_edge, Y_test_edge = np.concatenate((X_train_true, X_train_false), axis=0), \
                                                     np.concatenate((X_test_true, X_test_false), axis=0), \
                                                     np.concatenate((Y_train_true_edge, Y_train_false_edge), axis=0), \
                                                     np.concatenate((Y_test_true_edge, Y_test_false_edge), axis=0)
        Y_train = np.array([Y_train_edge.T[0].reshape(-1, 1), Y_train_edge.T[1].reshape(-1, 1)]).T.reshape(-1,
                                                                                                           2).astype(
            int)
        true_edges_test = np.array([true_edges_test_source, true_edges_test_target]).T[0]
        return X_train, Y_train, true_edges_test

    def train(self):
        """
        Prepare the data for train, also train the classifier and make the test data divide by classes.
        :return: The classifier and dict_class_movie_test
        """
        path1 = os.path.join(self.args.data_name, f'train/classifier23_{self.args.embedding}_{self.args.norm}.pkl')
        path2 = os.path.join(self.args.data_name, f'train/dict_{self.args.embedding}_{self.args.norm}.pkl')
        if os.path.exists(path1) and os.path.exists(path2):
            with open(path1, 'rb') as handle:
                classif2 = pickle.load(handle)
            with open(path2, 'rb') as handle:
                dict_class_movie_test = pickle.load(handle)
        else:
            classes = list(self.dict_true_edges.keys())
            for i, k in enumerate(sorted(self.dict_true_edges, key=lambda k: len(self.dict_true_edges[k]), reverse=True)):
                classes[i] = k
            dict_class_movie_test = {}
            X_train_all, Y_train_all = np.array([]), np.array([])
            seen_classes = classes[:int(0.8 * len(classes))]
            unseen_classes = classes[int(0.8 * len(classes)):]
            for j in range(len(self.args.ratio)):
                for c in seen_classes:
                    dict_movie_edge = {}
                    X_true, X_false, Y_true_edge, Y_false_edge = \
                        self.calculate_classifier_value(self.dict_true_edges[c], self.dict_false_edges[c])
                    X_train, Y_train, true_edges_test = self.split_data(X_true, X_false, Y_true_edge, Y_false_edge, self.args.ratio[j])
                    for edge in true_edges_test:
                        if edge[0][0] == 't':
                            movie = edge[0]
                        else:
                            movie = edge[1]
                        dict_movie_edge[movie] = edge
                    dict_class_movie_test[c] = dict_movie_edge.copy()
                    if len(X_train_all) > 0:
                        X_train_all = np.concatenate((X_train_all, X_train), axis=0)
                        Y_train_all = np.concatenate((Y_train_all, Y_train), axis=0)
                    else:
                        X_train_all = X_train
                        Y_train_all = Y_train
                for c in unseen_classes:
                    dict_movie_edge = {}
                    X_true, X_false, Y_true_edge, Y_false_edge = \
                        self.calculate_classifier_value(self.dict_true_edges[c], self.dict_false_edges[c])
                    _, _, true_edges_test = self.split_data(X_true, X_false, Y_true_edge, Y_false_edge, ratio=0)
                    for edge in true_edges_test:
                        if edge[0][0] == 't':
                            movie = edge[0]
                        else:
                            movie = edge[1]
                        dict_movie_edge[movie] = edge
                    dict_class_movie_test[c] = dict_movie_edge.copy()
                shuff = np.c_[X_train_all.reshape(len(X_train_all), -1), Y_train_all.reshape(len(Y_train_all), -1)]
                np.random.shuffle(shuff)
                X_train_all = shuff.T[0].reshape(-1, 1)
                Y_train_all = np.array([shuff.T[1].reshape(-1, 1), shuff.T[2].reshape(-1, 1)]).T.reshape(-1, 2).astype(
                    int)
                classif2 = self.train_edge_classification(np.array(X_train_all), np.array(Y_train_all))
                with open(path1, 'wb') as fid:
                    pickle.dump(classif2, fid)
                with open(path2, 'wb') as fid:
                    pickle.dump(dict_class_movie_test, fid)
        return classif2, dict_class_movie_test

    @staticmethod
    def predict_edge_classification(classif2, X_test):
        """
        With the test data make 
        :param classif2:
        :param X_test:
        :return:
        """
        top_k_list = list(np.ones(len(X_test)).astype(int))
        prediction, probs = classif2.predict(X_test, top_k_list)
        return prediction, probs

    def evaluate(self, classif2, dict_class_movie_test):
        # evaluate
        classes = list(self.dict_true_edges.keys())
        pred_true = []
        pred = []
        for i, k in enumerate(sorted(self.dict_true_edges, key=lambda k: len(self.dict_true_edges[k]), reverse=True)):
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
                for i, edge in enumerate(edges):
                    norm = self.edge_distance(edge)
                    class_test[i, 0] = norm
                _, probs = self.predict_edge_classification(classif2, class_test)
                pred_index = np.argmax(probs.T[0])
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
            acc = count / len(class_movies)
            dict_measures['acc'] = acc
            dict_class_measures[c] = dict_measures.copy()
        with open(os.path.join(self.args.data_name, f'dict_class_measures_{self.args.embedding}_{self.args.norm}.pkl'),
                  'wb') as handle:
            pickle.dump(dict_class_measures, handle, protocol=3)
        # TODO dict class measures for every ratio
        return dict_class_measures, pred, pred_true

    def confusion_matrix_maker(self, dict_class_measures, pred, pred_true):
        conf_matrix = confusion_matrix(pred_true, pred, labels=list(dict_class_measures.keys()))
        true_count = 0
        count = 0
        for i in range(len(conf_matrix)):
            true_count += conf_matrix[i][i]
            for j in range(len(conf_matrix)):
                count += conf_matrix[i][j]
        print(f'acc_all: {true_count / count}')
        plt.figure(1)
        mpl.rcParams['xtick.labelsize'] = 14
        mpl.rcParams['ytick.labelsize'] = 14
        mpl.rcParams['axes.titlesize'] = 20
        mpl.rcParams['axes.labelsize'] = 16
        plt.title('Confusion Matrix, ZSL OUR_IMDB')
        plt.xlabel("True Labels")
        plt.ylabel("Predicted Labels")
        plt.imshow(conf_matrix, cmap='gist_gray', vmin=0, vmax=2)
        plt.colorbar()
        plt.savefig(f'{self.args.data_name}/plots/confusion_matrix_{self.args.embedding}_{self.args.norm}')


def main():
    """
    Main Function for link prediction task.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='our_imdb')
    parser.add_argument('--norm', default='cosine')  # cosine / L2 Norm / L1 Norm
    parser.add_argument('--embedding', default='Node2Vec')  # Node2Vec / Event2Vec
    parser.add_argument('--false_per_true', default=10)
    parser.add_argument('--ratio', default=[0.8])
    args = parser.parse_args()
    # ratio_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    graph_maker = GraphImporter(args)
    graph = graph_maker.import_imdb_multi_graph()
    edges_preparation = EdgesPreparation(graph, args)
    dict_true_edges = edges_preparation.label_edges_classes_ordered()
    dict_false_edges = edges_preparation.make_false_label_edges(dict_true_edges)
    graph = edges_preparation.seen_graph()
    embeddings_maker = EmbeddingCreator(graph, args)
    if args.embedding == 'Node2Vec':
        dict_embeddings = embeddings_maker.create_node2vec_embeddings()
    elif args.embedding == 'Event2Vec':
        dict_embeddings = embeddings_maker.create_event2vec_embeddings()
    classifier = Classifier(dict_true_edges, dict_false_edges, dict_embeddings, args)
    classif, dict_class_movie_test = classifier.train()
    dict_class_measures_node2vec, pred, pred_true = classifier.evaluate(classif, dict_class_movie_test)
    classifier.confusion_matrix_maker(dict_class_measures_node2vec, pred, pred_true)
    with open(os.path.join(args.data_name, f'dict_class_measures_{args.embedding}_{args.norm}.pkl'), 'rb') as handle:
        dict_class_measures = pickle.load(handle)
    keys = list(dict_class_measures.keys())
    count = 0
    for key in keys:
        count += dict_class_measures[key]['acc']
    avg_acc = count / len(keys)
    print(f'avg. acc:{avg_acc}')


main()