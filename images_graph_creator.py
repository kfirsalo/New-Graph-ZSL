import argparse
import json
import os.path as osp
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from itertools import chain
import networkx as nx
import random

import torch
from torch.utils.data import DataLoader

from models.resnet import make_resnet50_base

import sys

sys.path.insert(1, "./ZSL _DataSets")
from image_folder import ImageFolder
from utils import set_gpu, pick_vectors
from utlis_graph_zsl import get_classes, classes_split


class ImagesEmbeddings:
    def __init__(self, args, data_path, split_path, save_path):
        self.data_path = data_path
        self.split_path = split_path
        self.images_embed_path = osp.join(save_path, 'matrix_embeds.npy')
        self.dict_image_embed_path = osp.join(save_path, 'dict_image_embed.npy')
        self.dict_image_class_path = osp.join(save_path, 'dict_image_class.npy')
        self.classes_path = osp.join(save_path, 'classes_ordered.npy')
        self.args = args
        self.seen_classes, self.unseen_classes = self.classes()
        self.dict_name_class, self.dict_class_name = self.classes_names_translation()
        self.cnn = self.cnn_maker()

    def classes(self):
        seen_classes, unseen_classes = classes_split(self.args.dataset, self.data_path, self.split_path)
        return seen_classes, unseen_classes

    def classes_names_translation(self):
        awa2_split = json.load(open(self.split_path, 'r'))
        train_names = awa2_split['train_names']
        test_names = awa2_split['test_names']
        seen_classes, unseen_classes = self.classes()
        dict_name_class = {name: c for name, c in
                           zip(chain(train_names, test_names), chain(seen_classes, unseen_classes))}
        dict_class_name = {c: name for name, c in
                           zip(chain(train_names, test_names), chain(seen_classes, unseen_classes))}
        return dict_name_class, dict_class_name

    def cnn_maker(self):
        cnn = make_resnet50_base()
        cnn.load_state_dict(torch.load(self.args.cnn))
        if self.args.gpu == 0:
            cnn = cnn.cuda()
        cnn.eval()
        return cnn

    def one_class_images_embed(self, dataset, embed_matrix, count):
        loader = DataLoader(dataset=dataset, batch_size=32,
                            shuffle=False, num_workers=2)
        c = 0
        for batch_id, batch in enumerate(loader, 1):
            data, label = batch
            if cuda:
                data = data.cuda()
            with torch.no_grad():
                embeds = self.cnn(data)
            embed_matrix[c:c + loader.batch_size, :] = embeds  # (batch_size, d)
            count += loader.batch_size
            c += loader.batch_size
        return embed_matrix, count

    def images_embed_calculator(self):
        action = 'test'
        classes = np.array([])
        if osp.exists(self.images_embed_path) and True:
            embeds_matrix = np.load(self.images_embed_path, allow_pickle=True)
            dict_image_class = np.load(self.dict_image_class_path, allow_pickle=True).item()
            dict_image_embed = np.load(self.dict_image_embed_path, allow_pickle=True).item()
        else:
            count = 0
            for i, name in enumerate(
                    chain(self.dict_class_name[self.unseen_classes], self.dict_class_name[self.seen_classes])):
                dataset = ImageFolder(osp.join(self.data_path, 'images'), [name], f'{action}')
                embed_matrix = torch.tensor(np.zeros((len(dataset), 2048)))
                classes = np.concatenate((classes, np.repeat(name, len(dataset))))
                embed_matrix, count = self.one_class_images_embed(dataset, embed_matrix, count)
                b = np.array(dataset.data).T[0]
                im = np.array([item.split('/')[-1].split('.')[0] for item in b])
                if i == 0:
                    embeds_matrix = embed_matrix
                    ims = im
                else:
                    embeds_matrix = torch.cat((embeds_matrix, embed_matrix), 1)
                    ims = np.concatenate((ims, im))
            dict_image_class = dict(zip(ims, classes))
            dict_image_embed = dict(zip(ims, embeds_matrix.numpy()))
            np.save(self.images_embed_path, embeds_matrix)
            np.save(self.dict_image_class_path, dict_image_class)
            np.save(self.dict_image_embed_path, dict_image_embed)
        return embeds_matrix, dict_image_embed, dict_image_class


class Awa2GraphCreator:
    def __init__(self, embed_matrix, dict_image_embed, dict_name_class, dict_idx_image_class, images_nodes_percentage,
                 args):
        self.image_graph_path = 'save_awa2/image_graph.gpickle'
        self.pre_knowledge_graph_path = 'materials/imagenet-induced-graph.json'
        self.knowledge_graph_path = 'save_awa2/knowledge_graph.gpickle'
        self.dict_wnids_class_translation = dict_name_class
        self.embeddings = normalize(embed_matrix, norm='l2', axis=0)
        self.dict_image_embed = dict_image_embed
        self.images = list(dict_image_embed.keys())
        self.dict_idx_image_class = dict_idx_image_class
        self.images_nodes_percentage = images_nodes_percentage
        self.args = args

    def index_embed_transform(self):
        dict_index_embed = {i: item for i, item in enumerate(self.images)}
        dict_embed_index = {item: i for i, item in enumerate(self.images)}
        return dict_index_embed, dict_embed_index

    def create_image_graph(self):
        if osp.exists(self.image_graph_path) and True:
            image_gnx = nx.read_gpickle(self.image_graph_path)
        else:
            image_gnx = nx.Graph()
            kdt = KDTree(self.embeddings, leaf_size=40)
            # image_graph.add_nodes_from(np.arange(len(self.embeddings)))
            count = 0
            for i in range(len(self.embeddings)):
                neighbors, distances = kdt.query_radius(self.embeddings[i:i + 1], r=self.args.images_threshold,
                                                        return_distance=True)
                if len(neighbors[0]) == 1:
                    distances, neighbors = kdt.query(self.embeddings[i:i + 1], k=2,
                                                     return_distance=True)
                neighbors, distances = neighbors[0], distances[0]
                loop_ind = np.where(distances == 0)
                if len(loop_ind[0]) > 1:
                    loop_ind = np.where(neighbors == i)
                neighbors = np.delete(neighbors, loop_ind)
                distances = np.delete(distances, loop_ind)
                # make distance into weights and fix zero distances
                edges_weights = [1 / dist if dist > 0 else 1000 for dist in distances]
                len_neigh = len(neighbors)
                count += len_neigh
                mean = count / (i + 1)
                if i % 1000 == 0:
                    print('Progress:', i, '/', len(self.embeddings), ';  Current Mean:', mean)  # 37273
                weight_edges = list(zip(np.repeat(i, len(neighbors)).astype(str), neighbors.astype(str), edges_weights))
                image_gnx.add_weighted_edges_from(weight_edges)
            nx.write_gpickle(image_gnx, self.image_graph_path)
        return image_gnx

    def imagenet_knowledge_graph(self):
        graph = json.load(open(self.pre_knowledge_graph_path, 'r'))
        edges = graph['edges']
        nodes = graph['wnids']
        # dict_nodes_translation = {i: node for i, node in enumerate(nodes)}
        dict_class_nodes_translation = {node: 'c' + str(i) for i, node in enumerate(nodes)}
        dict_nodes_class_translation = {'c' + str(i): node for i, node in enumerate(nodes)}
        dict_class_nodes_translation = {**dict_class_nodes_translation, **dict_nodes_class_translation}
        # edges = [(dict_nodes_translation[x[0]],
        #           dict_nodes_translation[x[1]]) for x in edges]
        edges = [('c' + str(x[0]), 'c' + str(x[1])) for x in edges]
        kg_imagenet = nx.Graph()
        kg_imagenet.add_edges_from(edges)
        return kg_imagenet, dict_class_nodes_translation

    def attributed_graph(self, kg_imagenet, att_weight):
        graph = json.load(open(self.pre_knowledge_graph_path, 'r'))
        nodes = graph['wnids']
        all_attributes = graph['vectors']
        dict_class_nodes = {node: i for i, node in enumerate(nodes)}
        dict_nodes_class = {i: node for i, node in enumerate(nodes)}
        dict_class_nodes_translation = {**dict_class_nodes, **dict_nodes_class}
        awa2_split = json.load(open('materials/awa2-split.json', 'r'))
        seen_classes = awa2_split['train']
        unseen_classes = awa2_split['test']
        s_u_classes = seen_classes + unseen_classes
        s_u_idx = [dict_class_nodes_translation[c] for c in s_u_classes]
        kd_idx_to_class_idx = {i: dict_class_nodes_translation[c] for i, c in enumerate(s_u_idx)}
        attributes = np.array([all_attributes[idx] for idx in s_u_idx])
        attributes = normalize(attributes, norm='l2', axis=1)
        kdt = KDTree(attributes, leaf_size=10)
        # image_graph.add_nodes_from(np.arange(len(self.embeddings)))
        count = 0
        for i in range(len(attributes)):
            neighbors, distances = kdt.query_radius(attributes[i:i + 1], r=1.15,
                                                    return_distance=True)
            if len(neighbors[0]) == 1:
                distances, neighbors = kdt.query(attributes[i:i + 1], k=2,
                                                 return_distance=True)
            neighbors, distances = neighbors[0], distances[0]
            loop_ind = np.where(distances == 0)
            if len(loop_ind[0]) > 1:
                loop_ind = np.where(neighbors == i)
            neighbors = np.delete(neighbors, loop_ind)
            distances = np.delete(distances, loop_ind)
            # make distance into weights and fix zero distances
            edges_weights = [float(att_weight) / dist if dist > 0 else 1000 for dist in distances]
            len_neigh = len(neighbors)
            if len_neigh == 0:
                print('hi Im number ' + str(i))
            count += len_neigh
            mean = count / (i + 1)
            if i % 10 == 0:
                print('Progress:', i, '/', len(attributes), ';  Current Mean:', mean)  # 37273
            neighbors_translation = [dict_class_nodes_translation[kd_idx_to_class_idx[neighbor]] for neighbor in
                                     neighbors]
            weight_edges = list(zip(np.repeat(dict_class_nodes_translation[kd_idx_to_class_idx[i]], len(neighbors)),
                                    neighbors_translation, edges_weights))
            kg_imagenet.add_weighted_edges_from(weight_edges)
            # TODO: add the weight from the attributes to the pre graph and not replace them
            #  (minor problem because it is sparse graph)
            largest_cc = max(nx.connected_components(kg_imagenet), key=len)
            kg_imagenet = kg_imagenet.subgraph(largest_cc).copy()
        return kg_imagenet

    def awa2_knowledge_graph(self):
        kg_imagenet = self.imagenet_knowledge_graph()
        awa2_split = json.load(open('materials/awa2-split.json', 'r'))
        train_wnids = awa2_split['train']
        test_wnids = awa2_split['test']
        relevant_nodes = list(chain(train_wnids, test_wnids))
        kg_awa2 = kg_imagenet.subgraph(relevant_nodes)
        return kg_awa2

    def create_labels_graph(self, dict_class_nodes_translation):
        labels_graph = nx.Graph()
        edges = np.array([(key, dict_class_nodes_translation[self.dict_idx_image_class[key]])
                          for key in list(self.dict_idx_image_class.keys())]).astype(str)
        labels_graph.add_edges_from(edges)
        return labels_graph

    def weighted_graph(self, images_gnx, knowledge_graph, labels_graph, weights_dict):
        weighted_graph = nx.Graph()
        classes_nodes = knowledge_graph.nodes
        images_nodes = images_gnx.nodes
        labels_edges = labels_graph.edges
        images_edges = images_gnx.edges
        classes_edges = knowledge_graph.edges
        weighted_graph.add_nodes_from(images_nodes, key='movies')
        weighted_graph.add_nodes_from(classes_nodes, key='classes')
        # weighted_graph.add_edges_from(images_edges, key='images_edges')
        # weighted_graph.add_edges_from(classes_edges, key='classes_edges')
        # weighted_graph.add_edges_from(labels_edges, key='labels_edges')
        for edge in images_edges:
            dict_weight = images_gnx.get_edge_data(edge[0], edge[1])
            weight = dict_weight.get('weight')
            if weight is not None:
                weighted_graph.add_edge(edge[0], edge[1], weight=weight, key='images_edges')
        classes_weight = weights_dict['classes_edges']
        labels_weight = weights_dict['labels_edges']
        for edge in classes_edges:
            weighted_graph.add_edge(edge[0], edge[1], weight=classes_weight, key='images_edges')
        for edge in labels_edges:
            weighted_graph.add_edge(edge[0], edge[1], weight=labels_weight, key='labels_edges')
        images_nodes = np.array(images_nodes)
        random.Random(4).shuffle(images_nodes)
        classes_nodes = np.array(classes_nodes)
        random.Random(4).shuffle(classes_nodes)
        if self.images_nodes_percentage < 1:
            for image_node in images_nodes[0:int(len(images_nodes) * (1 - self.images_nodes_percentage))]:
                weighted_graph.remove_node(image_node)
            # for c in classes_nodes[0:int(len(classes_nodes) * 0.5)]:
            #     weighted_graph.remove_node(c)
        return weighted_graph


def define_path(dataset_name):
    if dataset_name == "awa2":
        _data_path = 'ZSL _DataSets/awa2/Animals_with_Attributes2'
        _split_path = 'materials/awa2-split.json'
        _chkpt_path = "save_data_graph/awa2"
    elif dataset_name == "cub":
        _data_path = "ZSL _DataSets/cub/CUB_200_2011"
        _split_path = "ZSL _DataSets/cub/CUB_200_2011/train_test_split_easy.mat"
        _chkpt_path = 'save_data_graph/cub'
    elif dataset_name == "lad":
        _data_path = "ZSL _DataSets/lad"
        _split_path = "ZSL _DataSets/lad/split_zsl.txt"
        _chkpt_path = 'save_data_graph/lad'
    else:
        raise ValueError("Wrong dataset name: replace with awa2/cub/lad")
    return _data_path, _split_path, _chkpt_path

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    # cuda = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest="dataset", help=' Name of the dataset', type=str, default="awa2")
    parser.add_argument('--cnn', default='materials/resnet50-base.pth')
    # parser.add_argument('--cnn', default='save_awa2/resnet-fit/epoch-1.pth')
    # parser.add_argument('--pred', default='save_awa2/gcn-dense-att/epoch-30.pred')
    # parser.add_argument('--pred', default='save_awa2/gcn-basic/epoch-34.pred')
    if cuda:
        parser.add_argument('--gpu', default='0')
    else:
        parser.add_argument('--gpu', default='-1')
    # parser.add_argument('--consider-trains', action='store_true')
    parser.add_argument('--consider-trains', action='store_false')

    parser.add_argument('--output', default=None)
    parser.add_argument('--images_threshold', default=0.10)
    parser.add_argument('--images_nodes_percentage', default=1)
    args = parser.parse_args()

    set_gpu(args.gpu)
    data_path, split_path, save_path = define_path(args.dataset)
    graph_preparation = ImagesEmbeddings(args, data_path, split_path, save_path)
    dict_name_class, dict_class_name = graph_preparation.dict_name_class, graph_preparation.dict_class_name
    seen_classes, unseen_classes = graph_preparation.seen_classes, graph_preparation.unseen_classes
    embeds_matrix, dict_image_embed, dict_image_class = graph_preparation.images_embed_calculator()
    # try:
    #     dict_image_class_new = {image: dict_name_class[dict_image_class[image]]
    #                         for image in list(dict_image_class.keys())}
    # except:
    #     dict_image_class_new = dict_image_class.copy()
    dict_idx_image_class = {i: dict_name_class[dict_image_class[image]]
                            for i, image in enumerate(list(dict_image_class.keys()))}
    awa2_graph_creator = Awa2GraphCreator(embeds_matrix, dict_image_embed, dict_name_class, dict_idx_image_class,
                                          args.images_nodes_percentage, args)
    image_graph = awa2_graph_creator.create_image_graph()
    kg, dict_class_nodes_translation = awa2_graph_creator.imagenet_knowledge_graph()
    att_weight = 10
    kg = awa2_graph_creator.attributed_graph(kg, att_weight)
    seen_classes = [dict_class_nodes_translation[c] for c in seen_classes]
    unseen_classes = [dict_class_nodes_translation[c] for c in unseen_classes]
    split = {'seen': seen_classes, 'unseen': unseen_classes}
    labels_graph = awa2_graph_creator.create_labels_graph(dict_class_nodes_translation)
    weights = [1, 1]
    weights_dict = {'classes_edges': weights[0], 'labels_edges': weights[1]}
    awa2_graph = awa2_graph_creator.weighted_graph(image_graph, kg, labels_graph, weights_dict)
    print(len(image_graph.edges))
    print((len(image_graph.nodes)))
    print(len(kg.edges))
    print((len(kg.nodes)))
    print(len(labels_graph.edges))
    print((len(awa2_graph.edges)))
