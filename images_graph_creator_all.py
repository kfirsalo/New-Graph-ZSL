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
from utils import set_gpu, pick_vectors, get_device
from utlis_graph_zsl import get_classes, classes_split
from general_resnet50 import ResNet50


class ImagesEmbeddings:
    def __init__(self, args, data_path, split_path, save_path, model_path):
        self.data_path = data_path
        self.split_path = split_path
        self.model_path = model_path
        self.images_path = osp.join(data_path, "images")
        self.images_embed_path = osp.join(save_path, 'matrix_embeds.npy')
        self.dict_image_embed_path = osp.join(save_path, 'dict_image_embed.npy')
        self.dict_image_class_path = osp.join(save_path, 'dict_image_class.npy')
        self.classes_path = osp.join(save_path, 'classes_ordered.npy')
        self.args = args
        self.device = get_device()
        self.embeddings_dimension = 2048 if args.dataset == "awa2" else 512
        self.seen_classes, self.unseen_classes = self.classes()
        self.cnn = self.cnn_maker()

    def classes(self):
        seen_classes, unseen_classes = classes_split(self.args.dataset, self.data_path, self.split_path)
        return seen_classes, unseen_classes

    def cnn_maker(self):
        if self.args.dataset == "awa2":
            cnn = make_resnet50_base()
            cnn.load_state_dict(torch.load(self.model_path))
        elif self.args.dataset == ("cub" or "lad"):
            cnn = ResNet50(out_dimension=len(self.seen_classes), chkpt_dir=self.model_path, device=self.device)
            cnn.load_checkpoint()
        else:
            raise ValueError("Wrong dataset name: replace with awa2/cub/lad")
        cnn.to(self.device)
        cnn.eval()
        return cnn

    def one_class_images_embed(self, dataset, embed_matrix, count):
        loader = DataLoader(dataset=dataset, batch_size=64,
                            shuffle=False, num_workers=4, pin_memory=True)
        c = 0
        for batch_id, (data, _) in enumerate(loader, 1):
            data.to(self.device)
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
            for i, name in enumerate(chain(self.unseen_classes, self.seen_classes)):
                dataset = ImageFolder(self.images_path, [name], stage=f'{action}')
                embed_matrix = torch.tensor(np.zeros((len(dataset), self.embeddings_dimension)))
                classes = np.concatenate((classes, np.repeat(name, len(dataset))))
                embed_matrix, count = self.one_class_images_embed(dataset, embed_matrix, count)
                im_paths = np.array(dataset.data).T[0]
                im_names = np.array([item.replace("\\", "/").split('/')[-1].split('.')[0] for item in im_paths])
                if i == 0:
                    embeds_matrix = embed_matrix
                    ims = im_names
                else:
                    embeds_matrix = torch.cat((embeds_matrix, embed_matrix), 1)
                    ims = np.concatenate((ims, im_names))
            dict_image_class = dict(zip(ims, classes))
            dict_image_embed = dict(zip(ims, embeds_matrix.numpy()))
            np.save(self.images_embed_path, embeds_matrix)
            np.save(self.dict_image_class_path, dict_image_class)
            np.save(self.dict_image_embed_path, dict_image_embed)
        return embeds_matrix, dict_image_embed, dict_image_class


class Awa2GraphCreator:
    def __init__(self, _data_path, _split_path, _attributes_path, embed_matrix, dict_image_embed,
                 dict_idx_image_class, images_nodes_percentage, args):
        self.data_path = _data_path
        self.split_path = _split_path
        self.attributes_path = _attributes_path
        self.image_graph_path = 'save_data_graph/awa2/image_graph.gpickle'
        self.pre_knowledge_graph_path = 'materials/imagenet-induced-graph.json'
        self.knowledge_graph_path = 'save_data_graph/awa2/knowledge_graph.gpickle'
        self.seen_classes, self.unseen_classes = classes_split(self.args.dataset, self.data_path, self.split_path)
        self.nodes = None
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

    @staticmethod
    def idx_nodes_translation(class_nodes):
        _dict_class_nodes_translation = {node: 'c' + str(i) for i, node in enumerate(class_nodes)}
        _dict_nodes_class_translation = {'c' + str(i): node for i, node in enumerate(class_nodes)}
        _dict_class_nodes_translation = {**_dict_class_nodes_translation, **_dict_nodes_class_translation}
        return _dict_class_nodes_translation

    def imagenet_knowledge_graph(self):
        if self.args.dataset == "awa2":
            graph = json.load(open(self.pre_knowledge_graph_path, 'r'))
            edges = graph['edges']
            self.nodes = graph['wnids']
            # dict_nodes_translation = {i: node for i, node in enumerate(nodes)}
            # edges = [(dict_nodes_translation[x[0]],
            #           dict_nodes_translation[x[1]]) for x in edges]
            edges = [('c' + str(x[0]), 'c' + str(x[1])) for x in edges]
            kg_imagenet = nx.Graph()
            kg_imagenet.add_edges_from(edges)
        elif self.args.dataset == ("cub" or "lad"):
            self.nodes = [*self.seen_classes, *self.unseen_classes]
            kg_imagenet = nx.Graph()
        else:
            raise ValueError("Wrong dataset name: replace with awa2/cub/lad")
        _dict_class_nodes_translation = self.idx_nodes_translation(self.nodes)
        return kg_imagenet, _dict_class_nodes_translation

    def _attributes(self):
        if self.args.dataset == "awa2":
            graph = json.load(open(self.pre_knowledge_graph_path, 'r'))
            all_attributes = graph['vectors']
        elif self.args.dataset == ("cub" or "lad"):
            all_attributes = open(self.attributes_path, "r")
            all_attributes = all_attributes.readlines()
        else:
            raise ValueError("Wrong dataset name: replace with awa2/cub/lad")
        dict_class_nodes = {node: i for i, node in enumerate(self.nodes)}
        dict_nodes_class = {i: node for i, node in enumerate(self.nodes)}
        dict_class_nodes_translation = {**dict_class_nodes, **dict_nodes_class}
        s_u_classes = [*self.seen_classes, *self.unseen_classes]
        s_u_idx = [dict_class_nodes_translation[c] for c in s_u_classes]
        kd_idx_to_class_idx = {i: dict_class_nodes_translation[c] for i, c in enumerate(s_u_idx)}
        attributes = np.array([all_attributes[idx] for idx in s_u_idx])
        attributes = normalize(attributes, norm='l2', axis=1)
        return kd_idx_to_class_idx, attributes

    def attributed_graph(self, final_kg, att_weight):
        kd_idx_to_class_idx, attributes = self._attributes()
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
            final_kg.add_weighted_edges_from(weight_edges)
            # TODO: add the weight from the attributes to the pre graph and not replace them
            #  (minor problem because it is sparse graph)
            largest_cc = max(nx.connected_components(final_kg), key=len)
            final_kg = final_kg.subgraph(largest_cc).copy()
        return final_kg

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
        _model_path = "materials/resnet50-base.pth"
        _attributes_path = ""
    elif dataset_name == "cub":
        _data_path = "ZSL _DataSets/cub/CUB_200_2011"
        _split_path = "ZSL _DataSets/cub/CUB_200_2011/train_test_split_easy.mat"
        _chkpt_path = 'save_data_graph/cub'
        _model_path = "save_models/cub/ResNet50_best"
        _attributes_path = "ZSL _DataSets/cub/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"
    elif dataset_name == "lad":
        _data_path = "ZSL _DataSets/lad"
        _split_path = "ZSL _DataSets/lad/split_zsl.txt"
        _chkpt_path = 'save_data_graph/lad'
        _model_path = "save_models/lad/ResNet50_best"
        _attributes_path = "ZSL _DataSets/lad/attributes_per_class.txt"
    else:
        raise ValueError("Wrong dataset name: replace with awa2/cub/lad")
    return _data_path, _split_path, _chkpt_path, _model_path, _attributes_path

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
    data_path, split_path, save_path, model_path, attributes_path = define_path(args.dataset)
    graph_preparation = ImagesEmbeddings(args, data_path, split_path, save_path, model_path)
    seen_classes, unseen_classes = graph_preparation.seen_classes, graph_preparation.unseen_classes
    embeds_matrix, dict_image_embed, dict_image_class = graph_preparation.images_embed_calculator()
    dict_idx_image_class = {i: dict_image_class[image]
                            for i, image in enumerate(list(dict_image_class.keys()))}
    awa2_graph_creator = Awa2GraphCreator(data_path, split_path, attributes_path, embeds_matrix, dict_image_embed,
                                          dict_idx_image_class, args.images_nodes_percentage, args)
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
