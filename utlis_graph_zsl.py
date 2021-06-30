import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import argparse
import pickle
import pandas as pd
import numpy as np
import random
import seaborn as sns
from itertools import chain
from itertools import product
from dataclasses import dataclass
import scipy.io as sio
import json

random.seed(0)
np.random.seed(0)


@dataclass
class C:
    a: float
    b: float
    c: float


def grid(dict_params, ignore_params=None):
    """ transforming continues space into discrete space by splitting every axis and then take the
        product of all the splits."""
    grid_dict_params = dict_params.copy()
    if ignore_params is not None:
        for key in ignore_params:
            grid_dict_params.pop(key)
    grid_combinations = product(*list(grid_dict_params.values()))
    return list(grid_combinations)


def plots_2measures_vs_parameter(dict_measures, x_axis, relevant_keys, title, x_title, y_title, path):
    keys = list(dict_measures.keys())
    # bottom = max([np.min(np.array([np.min(dict_measures[key]) for key in keys]))-10, 0])
    # top = min([np.max(np.array([np.max(dict_measures[key]) for key in keys]))+10, 100])
    plt.figure(figsize=(7, 6))
    for j in range(len(keys)):
        if relevant_keys[0] == keys[j]:
            color = 'red'
            marker = 'o'
            markersize = 8
            linestyle = 'solid'
            y_axis = dict_measures[keys[j]]
            plt.plot(x_axis, y_axis, marker=marker, linestyle=linestyle, markersize=markersize, color=color)
        elif relevant_keys[1] == keys[j]:
            color = 'green'
            marker = 's'
            markersize = 6
            linestyle = 'solid'
            y_axis = dict_measures[keys[j]]
            plt.plot(x_axis, y_axis, marker=marker, linestyle=linestyle, markersize=markersize, color=color)
    # plt.ylim(bottom=bottom, top=top)
    plt.legend(relevant_keys, bbox_to_anchor=(0., 0.7, 1., .102), loc=3,
               ncol=1)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.tight_layout()
    plt.savefig(path)


def old_plots_2measures_vs_parameter(dict_measures, parameter_arr, parameter_name, data_name, task, measure, norm,
                                     embedding):
    x_axis = np.array(parameter_arr)
    keys = list(dict_measures.keys())
    bottom = max([np.min(np.array([np.min(dict_measures[key]) for key in keys])) - 10, 0])
    top = min([np.max(np.array([np.max(dict_measures[key]) for key in keys])) + 10, 100])
    plt.figure(figsize=(7, 6))
    for j in range(len(keys)):
        if 'unseen_accuracy' in keys[j]:
            color = 'red'
            marker = 'o'
            markersize = 8
            linestyle = 'solid'
            y_axis = dict_measures[keys[j]]
        elif "seen_accuracy" in keys[j]:
            color = 'green'
            marker = 's'
            markersize = 6
            linestyle = 'solid'
            y_axis = dict_measures[keys[j]]
        plt.plot(x_axis, y_axis, marker=marker, linestyle=linestyle, markersize=markersize, color=color)
    plt.ylim(bottom=bottom, top=top)
    plt.legend(keys, loc='best', ncol=3, fontsize='large')
    plt.title("{} Dataset {} Task -\n{} Score Per {}".format(data_name, task, measure, parameter_name))
    plt.xlabel(parameter_name)
    plt.ylabel("{} ({})".format(measure, norm))
    plt.tight_layout()
    plt.savefig(os.path.join(data_name, "plots",
                             "{} {} {} {} {} with unseen advantage.png".format(data_name, task, measure, embedding,
                                                                               norm)))


def hist_plot(y_array, title, x_title, y_title):
    # num_bins = np.where(y_array != 0)[0][-1]
    # print(num_bins)
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    plt.figure(0)
    plt.bar(np.arange(len(y_array)), y_array, color='black')
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.tight_layout()


def plot_confusion_matrix(conf_matrix, title, x_title, y_title, save_path, fig=1, vmin=0, vmax=2, cmap='gist_gray'):
    plt.figure(fig)
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.imshow(conf_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(save_path)
    plt.close(fig)


def get_classes(images_dir):
    return np.array(os.listdir(images_dir))


def classes_split(dataset, data_dir, split_dir, return_translation=False):
    images_dir = os.path.join(data_dir, "images")
    if dataset == "awa2":
        awa2_split = json.load(open(split_dir, 'r'))
        seen_classes = awa2_split['train_names']
        unseen_classes = awa2_split['test_names']
    elif dataset == "cub":
        train_test_split = sio.loadmat(split_dir)
        classes = get_classes(images_dir)
        seen_classes = classes[train_test_split['train_cid'] - 1][0]
        unseen_classes = classes[train_test_split['test_cid'] - 1][0]
        classes_translation = {c[:3]: c for c in classes}
        if return_translation:
            return seen_classes, unseen_classes, classes_translation
    elif dataset == "lad":
        split_file = open(split_dir, "r")
        unseen_lists = {l.split(":")[0]: l.split(":")[1] for l in split_file.readlines()}
        unseen_list_1 = unseen_lists["Unseen_List_1"][:-1].replace(" ", "").split(",")
        classes_file = open(os.path.join(data_dir, "label_list.txt"), encoding="utf8")
        classes = classes_file.readlines()
        classes_translation = {**{c.split(", ")[0]: c.split(", ")[0].split("_")[1] + "_" + c.split(", ")[1]
                                  for c in classes}, **{c.split(", ")[0].split("_")[1] + "_"
                                                        + c.split(", ")[1]: c.split(", ")[0] for c in classes}}
        unseen_classes = np.array([classes_translation[c] for c in unseen_list_1])
        seen_classes = np.setdiff1d(get_classes(images_dir), unseen_classes)
        if return_translation:
            return seen_classes, unseen_classes, classes_translation
    else:
        raise ValueError("Wrong dataset name: replace with cub/lad/awa2")
    return seen_classes, unseen_classes


def calculate_weighted_jaccard_distance(vectors, vector, r=None, k=None):
    weighted_jd = 1 - np.sum(np.minimum(vectors, vector), axis=1) / np.sum(np.maximum(vectors, vector), axis=1)
    if r is not None:
        neighbors_indices = np.where(weighted_jd < r)[0]
    elif k is not None:
        neighbors_indices = np.argpartition(weighted_jd, k)[:k]
    else:
        raise ValueError("'r' and 'k' arguments can not be both None")
    distances = np.array([weighted_jd[i] for i in neighbors_indices])
    return neighbors_indices, distances



# def obj_func(weights):
#     """
#     Main Function for link prediction task.
#     :return:
#     """
#     np.random.seed(0)
#     print(weights)
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_name', default='our_imdb')
#     parser.add_argument('--norm', default='cosine')  # cosine / L2 Norm / L1 Norm
#     parser.add_argument('--embedding', default='Node2Vec')  # Node2Vec / Event2Vec / OGRE
#     parser.add_argument('--false_per_true', default=10)
#     parser.add_argument('--ratio', default=[0.8])
#     args = parser.parse_args()
#     # ratio_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     graph_maker = GraphImporter(args)
#     multi_graph = graph_maker.import_imdb_multi_graph(weights)
#     weighted_graph = graph_maker.import_imdb_weighted_graph(weights)
#     edges_preparation = EdgesPreparation(weighted_graph, multi_graph, args)
#     dict_true_edges = edges_preparation.label_edges_classes_ordered()
#     dict_false_edges = edges_preparation.make_false_label_edges(dict_true_edges)
#     graph = edges_preparation.seen_graph()
#     embeddings_maker = EmbeddingCreator(graph, args)
#     if args.embedding == 'Node2Vec':
#         dict_embeddings = embeddings_maker.create_node2vec_embeddings()
#     elif args.embedding == 'Event2Vec':
#         dict_embeddings = embeddings_maker.create_event2vec_embeddings()
#     elif args.embeddings == 'Oger':
#         dict_embeddings = embeddings_maker.create_oger_embeddings()
#     else:
#         raise ValueError(f"Wrong embedding name, {args.embedding}")
#     classifier = Classifier(dict_true_edges, dict_false_edges, dict_embeddings, args)
#     classif, dict_class_movie_test = classifier.train()
#     dict_class_measures_node2vec, pred, pred_true = classifier.evaluate(classif, dict_class_movie_test)
#     accuracy, seen_accuracy, unseen_accuracy = classifier.confusion_matrix_maker(
#         dict_class_measures_node2vec, pred, pred_true)
#     try:
#         values = pd.read_csv('our_imdb/train/optimaize_values_Node2Vec_l2.csv')
#         result = pd.read_csv('our_imdb/train/optimaize_result_Node2Vec_l2.csv')
#         df1 = pd.DataFrame(weights.reshape(1, 2), columns=['movie_weights', 'labels_weights'])
#         df2 = pd.DataFrame([accuracy], columns=['acc'])
#         frames1 = [values, df1]
#         frames2 = [result, df2]
#         values = pd.concat(frames1, axis=0, names=['movie_weights', 'labels_weights'])
#         result = pd.concat(frames2, axis=0, names=['acc'])
#     except:
#         values = pd.DataFrame(weights.reshape(1, 2), columns=['movie_weights', 'labels_weights'])
#         result = pd.DataFrame([accuracy], columns=['acc'])
#     values.to_csv('our_imdb/train/optimaize_values_Node2Vec_l2.csv', index=None)
#     result.to_csv('our_imdb/train/optimaize_result_Node2Vec_l2.csv', index=None)
#     print(accuracy)
#     return -accuracy

# x = np.array([0.5, 3.0])
# bnds = [(0, 100), (0, 100)]
# res = minimize(obj_func, x0=x, method='Nelder-Mead', bounds=bnds, options={'maxiter': 50})
# print(res)

# if __name__ == '__main__':
#     obj_func_nni()
