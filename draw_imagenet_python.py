import json
import numpy as np
from node2vec import Node2Vec
from train_resnet_fit import Awa2GraphCreator
import matplotlib.pyplot as plt
import networkx as nx
import os.path as osp
import matplotlib as mpl


def imagenet():
    awa2_graph_creator = Awa2GraphCreator([[1, 1], [2, 2]], {1: 1}, None, None, None, None)
    imagenet, dict_class_nodes_translation = awa2_graph_creator.imagenet_knowledge_graph()
    imagenet = awa2_graph_creator.attributed_graph(imagenet, 100)
    largest_cc = max(nx.connected_components(imagenet), key=len)
    imagenet = imagenet.subgraph(largest_cc).copy()
    return imagenet, dict_class_nodes_translation


def n2v(graph):
    node2vec = Node2Vec(graph, dimensions=2, walk_length=80, num_walks=16, workers=2)
    model = node2vec.fit()
    nodes = list(graph.nodes())
    dict_embeddings = {}
    for i in range(len(nodes)):
        dict_embeddings.update({nodes[i]: np.asarray(model.wv.get_vector(str(nodes[i])))})
    return dict_embeddings


def attributes(dict_embeddings, dict_class_nodes_translation):
    awa2_split = json.load(open('materials/awa2-split.json', 'r'))
    seen_classes = awa2_split['train']
    unseen_classes = awa2_split['test']
    dict_att = {0: "seen", 1: "unseen", 2: "other"}
    att = []
    for node in list(dict_embeddings.keys()):
        node = dict_class_nodes_translation[node]
        if len(set([node]).intersection(seen_classes)) > 0:
            att.append(0)  # seen
        elif len(set([node]).intersection(unseen_classes)) > 0:
            att.append(1)  # unseen
        else:
            att.append(2)  # other
    return att, dict_att


def plot_graph(dict_embed, att, dict_att, title):
    embeds = dict_embed.values()
    points = list(zip(*embeds))
    x_seen, y_seen, x_unseen, y_unseen, x_other, y_other, att_seen, att_unseen, att_other = [], [], [], [], [], [], [], [], []
    for i, (x, y) in enumerate(zip(points[0], points[1])):
        if att[i] == 0:
            x_seen.append(x)
            y_seen.append(y)
            att_seen.append(att[i])
        elif att[i] == 1:
            x_unseen.append(x)
            y_unseen.append(y)
            att_unseen.append(att[i])
        else:
            x_other.append(x)
            y_other.append(y)
            att_other.append(att[i])
    plt.scatter(x_other, y_other, c='yellow', label='other')
    plt.scatter(x_seen, y_seen, c='b', label='seen')
    plt.scatter(x_unseen, y_unseen, c='r', label='unseen')
    plt.legend(list(dict_att.values()), loc='best', ncol=3, fontsize='large')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(osp.join('awa2/plots', title))


if __name__=='__main__':
    title = 'ImageNet Graph With 100 times Attributes'
    imagenet_graph, dict_class_nodes_translation = imagenet()
    dict_embeds = n2v(imagenet_graph)
    att, dict_att = attributes(dict_embeds, dict_class_nodes_translation)
    plot_graph(dict_embeds, att, dict_att, title)