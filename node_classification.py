"""
Node classification Task For Evaluation:
Full explanation for what is done can be found the survey file in our github page.
Code explanation:
For this task, one should have a labeled graph: 2 files are required: Graph edges in '.edgelist' or '.txt' format and
nodes' labels in '.txt' format. For labeled graphs examples you can enter the link in the github page. You should
insert them in the appropriate place in the main function in the file 'directed_cheap_node2vec' or
'undirected_cheap_node2vec', depends on your graph's type.
This task compares two things:
1. Compare performance of our method and regular node2vec, i.e. we do the same task with both methods, calculate
    needed scores and compare between them - This would be mission 1.
2. Only for our method, compare the success of the task (measuring by several scores) for different number of nodes
    in the initial projection - This would be mission 2.
For mission 1: Go to the file 'static_embedding.py' . Change 'initial' variable to a list that consists the percentage
    of nodes you want in the initial projection (for example, for pubmed2, 0.975 means 100 nodes in the initial
    projection). Go back to this file and run main(1).
For mission 2: Go to the file 'static_embedding.py'. Change 'initial' variable to a list that consists a number of
    percentages of nodes you want in the initial projection (for example, for pubmed2 the list is [0.975, 0.905, 0.715,
     0.447, 0.339], meaning run with 100 nodes in the initial projection, then with 1000, 3000, 7000 and 10000).
     Go back to this file to the function 'initial_proj_vs_scores' and replace x to be equal to 'initial' list you
     changed earlier. Then, you can run this file- main(2).
Notice methods to compare, graph and other parameters are all chosen in 'static_embedding.py' file.
"""

from node2vec import Node2Vec
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
import argparse
import random


class GraphImporter(object):
    def __init__(self, data_name):
        self.data_name = data_name

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



try: import cPickle as pickle
except: import pickle
from sklearn import model_selection as sk_ms
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score
import numpy as np


"""
Code for the node classification task as explained in GEM article. This part of the code belongs to GEM.
For more information, you can go to our github page.
"""


class TopKRanker(oneVr):
    """
    Linear regression with one-vs-rest classifier
    """
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        prediction = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
            for label in labels:
                prediction[i, label] = 1
        return prediction, probs


def evaluateNodeClassification(X_train, X_test, Y_train, Y_test):
    """
    Predictions of nodes' labels.
    :param X: The features' graph- the embeddings from node2vec
    :param Y: The nodes' labels
    :param test_ratio: To determine how to split the data into train and test
    :return: Scores- F1-macro, F1-micro and accuracy.
    """
    try:
        top_k_list = list(Y_test.toarray().sum(axis=1))
    except:
        top_k_list = list(Y_test.sum(axis=1))
    classif2 = TopKRanker(lr(solver='lbfgs', max_iter=1000))
    classif2.fit(X_train, Y_train)
    prediction, probs = classif2.predict(X_test, top_k_list)
    return prediction, probs


def evaluate_node_classification(prediction, Y_test):
    """
    Predictions of nodes' labels.
    :param X: The features' graph- norm
    :param Y: The edges labels- 0 for true, 1 for false
    :param test_ratio: To determine how to split the data into train and test
    :return: Scores- F1-macro, F1-micro accuracy and auc
    """
    accuracy = accuracy_score(Y_test, prediction)
    micro = f1_score(Y_test, prediction, average='micro', zero_division=0)
    macro = f1_score(Y_test, prediction, average='macro', zero_division=0)
    auc = roc_auc_score(Y_test, prediction)
    precision = precision_score(Y_test, prediction, average='micro')
    return micro, macro, accuracy, auc, precision
    # return micro, macro, accuracy


def expNC(X, Y, test_ratio_arr, rounds):
    """
    The final node classification task as explained in our git.
    :param X: The features' graph- the embeddings from node2vec
    :param Y: The nodes' labels
    :param test_ratio_arr: To determine how to split the data into train and test. This an array
                with multiple options of how to split.
    :param rounds: How many times we're doing the mission. Scores will be the average
    :return: Scores for all splits and all splits- F1-micro, F1-macro and accuracy.
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
            micro_round[i], macro_round[i], acc_round[i], auc_round[i] = evaluateNodeClassification(X, Y, test_ratio)

        micro[round_id] = micro_round
        macro[round_id] = macro_round
        acc[round_id] = acc_round
        auc[round_id] = auc_round

    micro = np.asarray(micro)
    macro = np.asarray(macro)
    acc = np.asarray(acc)
    auc = np.asarray(auc)

    return micro, macro, acc, auc


def input_for_classification(dict_proj, relevant_nodes):
    """
    Run cheap node2vec and make it a features matrix- matrix of size number of sample by number of embedding
    dimension, where the i_th row of X is its projection from cheap node2vec.
    :param dict_proj: A dictionary with keys==nodes in projection and values==projection
    :return: a matrix as explained above
    """
    X = []
    for i in range(len(relevant_nodes)):
        X_i = []
        for node in relevant_nodes[i]:
            X_i.append(dict_proj[node])
        X.append(np.array(X_i).astype(float))
    return np.array(X)


def read_labels(K, data_name, times):
    labels_path = data_name + '_labels.txt'
    path = os.path.join(data_name, labels_path)
    if os.path.exists(path):
        node_label = {}
        classes = []
        with open(path, 'r') as f:
            for line in f:
                items = line.strip().split()
                node_label[items[0]] = items[1]
                if items[1] not in classes:
                    classes.append(items[1])
        num_classes = np.max(np.array(classes).astype(int))
        one_hot_vec = np.zeros(num_classes)
        keys = list(node_label.keys())
        random_ind = random.sample(range(1, len(keys)), times * K)
        relevant_labels = []
        relevant_nodes = []
        for j in range(times):
            relevant_labels_i = []
            relevant_nodes_i = []
            indexes = np.arange(j * K, (j + 1) * K)
            for i in indexes:
                tmp = one_hot_vec.copy()
                tmp[int(node_label[keys[random_ind[i]]])-1] = 1
                tmp = tmp.astype(int)
                relevant_labels_i.append(tmp)
                relevant_nodes_i.append(keys[random_ind[i]])
            relevant_labels.append(relevant_labels_i)
            relevant_nodes.append(relevant_nodes_i)
    else:
        # TODO
        pass
    return np.array(relevant_labels), relevant_nodes


def compute_final_measures(input, labels, ratio, times):
    all_micro, all_macro, all_acc, all_auc, all_precision = [], [], [], [], []
    mean_acc, mean_auc, mean_micro, mean_macro, mean_precision = [], [], [], [], []
    std_acc, std_auc, std_micro, std_macro, std_precision = [], [], [], [], []
    dict_measures = {}
    for j in range(len(ratio)):
        for i in range(times):
            X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(input[i], labels[i], test_size=(1-ratio[j]))
            prediction, probs = evaluateNodeClassification(X_train, X_test, Y_train, Y_test)
            micro, macro, acc, auc, precision = evaluate_node_classification(prediction, Y_test)
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


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams["font.family"] = "Times New Roman"


def plots_maker(dict_measures, ratio_arr, measure, data_name):
    x_axis = np.array(ratio_arr)
    task = 'Node Classification'
    bottom = 0.7
    top = 0.99
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
    plt.plot(x_axis, [0.96, 0.957, 0.963, 0.964, 0.962, 0.964, 0.965, 0.97, 0.98], marker='o', linestyle='dashed', markersize=8, color='red')
    plt.plot(x_axis, [0.939, 0.945, 0.947, 0.95, 0.95, 0.952, 0.952, 0.955, 0.958], marker='s', linestyle='dashed', markersize=6, color='green')
    plt.ylim(bottom=bottom, top=top)
    keys = ['our_node2vec', 'our_event2vec', 'event2vec', 'node2vec']
    plt.legend(keys, loc='best', ncol=3, fontsize='large')
    plt.title("{} Dataset \n {} Task - {} Score".format(data_name, task, measure))
    plt.xlabel("Percentage")
    plt.ylabel("{}".format(measure))
    plt.tight_layout()
    plt.savefig(os.path.join(data_name, "plots", "{} {} {}.png".format(data_name, task, measure)))
    plt.show()


def main():
    """
    Main Function for link prediction task.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='dblp')
    args = parser.parse_args()
    number = 2500
    times = 1
    ratio_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    graph_maker = GraphImporter(args.data_name)
    graph = graph_maker.import_graph()
    # nodes = graph.nodes()
    # indexes = np.linspace(0, len(nodes)-1, 5000)
    # indexes = indexes.astype(int)
    # relevant_nodes = np.array(nodes)[indexes]
    # graph = nx.subgraph(graph, relevant_nodes)
    embeddings_maker = EmbeddingCreator(args.data_name, graph)
    dict_embeddings_event2vec = embeddings_maker.create_event2vec_embeddings()
    dict_embeddings_node2vec = embeddings_maker.create_event2vec_embeddings()
    # dict_event2vec_embeddings = embedding_model.create_event2vec_embeddings()
    # nodes = list(dict_event2vec_embeddings.keys())
    # relevant_edges = edges_to_predict(multi_graph)
    # true_edges = choose_true_edges(relevant_edges, number)
    # false_edges = choose_false_edges(multi_graph, relevant_edges, number)
    labels, relevant_nodes = read_labels(number, args.data_name, times)
    input_event2vec = input_for_classification(dict_embeddings_event2vec, relevant_nodes)
    input_node2vec = input_for_classification(dict_embeddings_node2vec, relevant_nodes)
    dict_measures_event2vec = compute_final_measures(input_event2vec, labels, ratio_arr, times)
    dict_measures_node2vec = compute_final_measures(input_node2vec, labels, ratio_arr, times)
    dict_measures = {}
    dict_measures['node2vec'] = dict_measures_node2vec
    dict_measures['event2vec'] = dict_measures_event2vec
    plots_maker(dict_measures, ratio_arr, 'AUC', 'DBLP')
    print('avg acc e2v: ', dict_measures_event2vec['Accuracy'])
    print('avg auc e2v: ', dict_measures_event2vec['AUC'])
    print('avg micro e2v: ', dict_measures_event2vec['Micro-f1'])
    print('avg macro e2v: ', dict_measures_event2vec['Macro-f1'])
    print('std acc e2v: ', dict_measures_event2vec['std_acc'])
    print('std auc e2v: ', dict_measures_event2vec['std_auc'])
    print('std micro e2v: ', dict_measures_event2vec['std_micro'])
    print('std macro e2v: ', dict_measures_event2vec['std_macro'])
    print('avg acc n2v: ', dict_measures_node2vec['Accuracy'])
    print('avg auc n2v: ', dict_measures_node2vec['AUC'])
    print('avg micro n2v: ', dict_measures_node2vec['Micro-f1'])
    print('avg macro n2v: ', dict_measures_node2vec['Macro-f1'])
    print('std acc n2v: ', dict_measures_node2vec['std_acc'])
    print('std auc n2v: ', dict_measures_node2vec['std_auc'])
    print('std micro n2v: ', dict_measures_node2vec['std_micro'])
    print('std macro n2v: ', dict_measures_node2vec['std_macro'])
    # micro, macro, acc, auc = exp_lp(X, Y, ratio_arr, 3)
    # avg_micro, avg_macro, avg_acc, avg_auc = calculate_all_avg_scores(micro, macro, acc, auc, 3)
    # all_micro.append(avg_micro)
    # all_macro.append(avg_macro)
    # all_acc.append(avg_acc)
    # all_auc.append(avg_auc)
    # fig1, fig2, fig3, fig4 = split_vs_score(all_micro[0], all_macro[0], all_micro[1], all_macro[1], all_acc[0],
    #                                         all_acc[1], all_auc[0], all_auc[1], ratio_arr)
    # plt.show()


main()