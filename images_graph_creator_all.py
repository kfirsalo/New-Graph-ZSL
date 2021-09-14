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
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from utlis_graph_zsl import calculate_weighted_jaccard_distance

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
    def __init__(self, paths, _args):
        self.data_path = paths["data_path"]
        self.split_path = paths["split_path"]
        self.model_path = paths["model_path"]
        self.save_path = paths["save_path"]
        self.images_path = osp.join(self.data_path, "images")
        self.images_embed_path = osp.join(self.save_path, 'matrix_embeds.npy')
        self.dict_image_embed_path = osp.join(self.save_path, 'dict_image_embed.npy')
        self.dict_image_class_path = osp.join(self.save_path, 'dict_image_class.npy')
        self.val_images_names_path = osp.join(self.save_path, "val_images_names.npy")
        self.classes_path = osp.join(self.save_path, 'classes_ordered.npy')
        self.args = _args
        self.device = get_device()
        # self.embeddings_dimension = 2048 if self.args.dataset == "awa2" else 512
        self.embeddings_dimension = 512
        self.seen_classes, self.unseen_classes = self.get_classes()
        self.classes = [*self.seen_classes, *self.unseen_classes]
        self.cnn = self.cnn_maker()

    def get_classes(self):
        seen_classes, unseen_classes = classes_split(self.args.dataset, self.data_path, self.split_path)
        return seen_classes, unseen_classes

    def cnn_maker(self):
        # if self.args.dataset == "awa2":
        #     cnn = make_resnet50_base()
        #     cnn.load_state_dict(torch.load(self.model_path))
        if self.args.dataset == "cub" or self.args.dataset == "lad" or self.args.dataset == "awa2":
            cnn = ResNet50(out_dimension=len(self.seen_classes), chkpt_dir=self.model_path, device=self.device)
            cnn.load_best()
        else:
            raise ValueError("Wrong dataset name: replace with awa2/cub/lad")
        cnn.to(self.device)
        cnn.eval()
        return cnn

    def one_class_images_embed(self, dataset, embed_matrix, stage="Train"):
        loader = DataLoader(dataset=dataset, batch_size=64,
                            shuffle=False, num_workers=4, pin_memory=True)
        c = 0
        classes = []
        for data, cls in tqdm(loader, desc=f"{stage} Images Embedding Extract"):
            classes.extend(cls)
            data.to(self.device)
            with torch.no_grad():
                # if self.args.dataset == "awa2":
                #     embeds = self.cnn(data)
                if self.args.dataset == "cub" or self.args.dataset == "lad" or self.args.dataset == "awa2":
                    embeds = self.cnn.resnet50(data)
                else:
                    raise ValueError("Wrong dataset name: replace with awa2/cub/lad")
            embed_matrix[c:c + loader.batch_size, :] = embeds  # (batch_size, d)
            c += loader.batch_size
        return embed_matrix, classes

    # def images_embed_calculator(self):
    #     action = 'test'
    #     classes = np.array([])
    #     if osp.exists(self.images_embed_path) and True:
    #         embeds_matrix = np.load(self.images_embed_path, allow_pickle=True)
    #         dict_image_class = np.load(self.dict_image_class_path, allow_pickle=True).item()
    #         dict_image_embed = np.load(self.dict_image_embed_path, allow_pickle=True).item()
    #     else:
    #         print("... calculate images embeddings ...")
    #         count = 0
    #         for i, name in enumerate(chain(self.unseen_classes, self.seen_classes)):
    #             dataset = ImageFolder(self.images_path, [name], stage=f'{action}')
    #             embed_matrix = torch.tensor(np.zeros((len(dataset), self.embeddings_dimension)))
    #             classes = np.concatenate((classes, np.repeat(name, len(dataset))))
    #             embed_matrix, count = self.one_class_images_embed(dataset, embed_matrix, count)
    #             im_paths = np.array(dataset.data).T[0]
    #             im_names = np.array([item.replace("\\", "/").split('/')[-1].split('.')[0] for item in im_paths])
    #             if i == 0:
    #                 embeds_matrix = embed_matrix
    #                 ims = im_names
    #             else:
    #                 embeds_matrix = torch.cat((embeds_matrix, embed_matrix), 0)
    #                 ims = np.concatenate((ims, im_names))
    #         dict_image_class = dict(zip(ims, classes))
    #         dict_image_embed = dict(zip(ims, embeds_matrix.numpy()))
    #         np.save(self.images_embed_path, embeds_matrix)
    #         np.save(self.dict_image_class_path, dict_image_class)
    #         np.save(self.dict_image_embed_path, dict_image_embed)
    #     return embeds_matrix, dict_image_embed, dict_image_class

    def images_embed_calculator(self):
        if osp.exists(self.images_embed_path) and osp.exists(self.val_images_names_path) and True:
            embeds_matrix = np.load(self.images_embed_path, allow_pickle=True)
            dict_image_class = np.load(self.dict_image_class_path, allow_pickle=True).item()
            dict_image_embed = np.load(self.dict_image_embed_path, allow_pickle=True).item()
            val_im_names = np.load(self.val_images_names_path, allow_pickle=True)
        else:
            print("... calculate images embeddings ...")
            train_set = ImageFolder(self.images_path, self.seen_classes, train_percentage=self.args.train_percentage,
                                    stage='train', specific_class_names=True)
            val_set = ImageFolder(self.images_path, self.seen_classes, train_percentage=self.args.train_percentage,
                                  stage='val', specific_class_names=True)
            test_set = ImageFolder(self.images_path, self.unseen_classes, stage='test', specific_class_names=True)
            train_embed_matrix, train_im_names, train_classes = self.embedding_extractor(train_set, stage="Train")
            val_embed_matrix, val_im_names, val_classes = self.embedding_extractor(val_set, stage="Validation")
            test_embed_matrix, test_im_names, test_classes = self.embedding_extractor(test_set, stage="Test")
            # test_classes = np.array(test_classes).astype(int) + np.max(np.array(val_classes).astype(int)+1).astype(str)
            embeds_matrix = torch.cat((train_embed_matrix, val_embed_matrix, test_embed_matrix), 0)
            ims = np.concatenate((train_im_names, val_im_names, test_im_names))
            classes = np.concatenate((train_classes, val_classes, test_classes))
            dict_image_class = dict(zip(ims, classes))
            dict_image_embed = dict(zip(ims, embeds_matrix.numpy()))
            np.save(self.images_embed_path, embeds_matrix)
            np.save(self.dict_image_class_path, dict_image_class)
            np.save(self.dict_image_embed_path, dict_image_embed)
            np.save(self.val_images_names_path, val_im_names)
        return embeds_matrix, dict_image_embed, dict_image_class, val_im_names

    def embedding_extractor(self, dataset, stage):
        embed_matrix = torch.tensor(np.zeros((len(dataset), self.embeddings_dimension)))
        embed_matrix, classes = self.one_class_images_embed(dataset, embed_matrix, stage)
        im_paths = np.array(dataset.data).T[0]
        im_names = np.array([item.replace("\\", "/").split('/')[-1].split('.')[0] for item in im_paths])
        return embed_matrix, im_names, classes


class FinalGraphCreator:
    def __init__(self, paths, _embed_matrix, _dict_image_embed,
                 _dict_idx_image_class, images_nodes_percentage, _args):
        self.args = _args
        self.data_path = paths["data_path"]
        self.split_path = paths["split_path"]
        self.attributes_path = paths["attributes_path"]
        self.image_graph_path = osp.join("save_data_graph", self.args.dataset, "image_graph.gpickle")
        self.pre_knowledge_graph_path = paths["pre_knowledge_graph_path"]
        self.knowledge_graph_path = osp.join("save_data_graph", self.args.dataset, "knowledge_graph.gpickle")
        if self.args.dataset == "lad" or self.args.dataset == "cub":
            self.seen_classes, self.unseen_classes, self.classes_translate = classes_split(self.args.dataset,
                                                                                           self.data_path,
                                                                                           self.split_path,
                                                                                           return_translation=True)
        else:
            self.seen_classes, self.unseen_classes = classes_split(self.args.dataset, self.data_path, self.split_path)
        if self.args.dataset == "awa2_w_imagenet":
            self.dict_name_class, self.dict_class_name = self.classes_names_translation()
            self.seen_classes = [self.dict_class_name[item] for item in self.seen_classes]
            self.unseen_classes = [self.dict_class_name[item] for item in self.unseen_classes]
        self.nodes = None
        self.embeddings = normalize(_embed_matrix, norm='l2', axis=0)
        self.dict_image_embed = _dict_image_embed
        self.images = list(_dict_image_embed.keys())
        self.dict_idx_image_class = _dict_idx_image_class
        self.images_nodes_percentage = images_nodes_percentage

    def index_embed_transform(self):
        dict_index_embed = {i: item for i, item in enumerate(self.images)}
        dict_embed_index = {item: i for i, item in enumerate(self.images)}
        return dict_index_embed, dict_embed_index

    def classes_names_translation(self):
        awa2_split = json.load(open(self.split_path, 'r'))
        train_names = awa2_split['train']
        test_names = awa2_split['test']
        dict_name_class = {name: c for name, c in
                           zip(chain(train_names, test_names), chain(self.seen_classes, self.unseen_classes))}
        dict_class_name = {c: name for name, c in
                           zip(chain(train_names, test_names), chain(self.seen_classes, self.unseen_classes))}
        return dict_name_class, dict_class_name

    def create_image_graph(self, _radius):
        if osp.exists(self.image_graph_path) and True:
            image_gnx = nx.read_gpickle(self.image_graph_path)
        else:
            image_gnx = nx.Graph()
            kdt = KDTree(self.embeddings, leaf_size=40)
            # image_graph.add_nodes_from(np.arange(len(self.embeddings)))
            count = 0
            for i in tqdm(range(len(self.embeddings)), desc="Create Image Graph"):
                neighbors, distances = kdt.query_radius(self.embeddings[i:i + 1], r=_radius["images_radius"],
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
                if i % 100 == 0:
                    # try to make avg. mean between 10 to 20
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
        if self.args.dataset == "awa2_w_imagenet":
            graph = json.load(open(self.pre_knowledge_graph_path, 'r'))
            edges = graph['edges']
            self.nodes = graph['wnids']
            # dict_nodes_translation = {i: node for i, node in enumerate(nodes)}
            # edges = [(dict_nodes_translation[x[0]],
            #           dict_nodes_translation[x[1]]) for x in edges]
            edges = [('c' + str(x[0]), 'c' + str(x[1])) for x in edges]
            kg_imagenet = nx.Graph()
            kg_imagenet.add_edges_from(edges)
        elif self.args.dataset == "cub" or self.args.dataset == "lad" or self.args.dataset == "awa2":
            self.nodes = [*self.seen_classes, *self.unseen_classes]
            kg_imagenet = nx.Graph()
        else:
            raise ValueError("Wrong dataset name: replace with awa2/cub/lad")
        _dict_class_nodes_translation = self.idx_nodes_translation(self.nodes)
        return kg_imagenet, _dict_class_nodes_translation

    def _attributes(self):
        if self.args.dataset == "awa2_w_imagenet":
            graph = json.load(open(self.pre_knowledge_graph_path, 'r'))
            all_attributes = graph['vectors']
        elif self.args.dataset == "awa2":
            all_attributes = open("ZSL _DataSets/awa2/Animals_with_Attributes2/predicate-matrix-binary.txt", 'r')
            all_attributes = all_attributes.readlines()
            all_attributes = np.array([attribute.strip('\n').split(' ') for attribute in all_attributes]).astype(float)
            classes = open("ZSL _DataSets/awa2/Animals_with_Attributes2/classes.txt", 'r')
            classes = classes.readlines()
            classes_attributes = np.array([c.strip().split("\t")[1] for c in classes])
        elif self.args.dataset == "cub":
            all_attributes = open(self.attributes_path, "r")
            all_attributes = all_attributes.readlines()
            all_attributes = np.array([attribute.strip().split(" ") for attribute in all_attributes]).astype(float)
            classes_attributes = list(self.classes_translate.values())
        elif self.args.dataset == "lad":
            raw_attributes = open(self.attributes_path, "r")
            raw_attributes = raw_attributes.readlines()
            all_attributes = np.array([attribute.strip().split(", ")[1].split("  ")[1:-1] for
                                       attribute in raw_attributes]).astype(float)
            attribute_classes_order = np.array([attribute.strip().split(", ")[0].split("  ") for
                                                attribute in raw_attributes])
            classes_attributes = [self.classes_translate[c[0]] for c in attribute_classes_order]
        else:
            raise ValueError("Wrong dataset name: replace with awa2/cub/lad")
        dict_class_nodes = {node: i for i, node in enumerate(self.nodes)}
        dict_nodes_class = {i: node for i, node in enumerate(self.nodes)}
        # if self.args.dataset == "awa2":
        #     dict_name_class, dict_class_name = self.classes_names_translation()
        #     self.seen_classes = [dict_class_name[item] for item in self.seen_classes]
        #     self.unseen_classes = [dict_class_name[item] for item in self.unseen_classes]
        _dict_class_nodes_translation = {**dict_class_nodes, **dict_nodes_class}
        s_u_classes = [*self.seen_classes, *self.unseen_classes]
        s_u_idx = [_dict_class_nodes_translation[c] for c in s_u_classes]
        seen_idx = set([_dict_class_nodes_translation[c] for c in self.seen_classes])
        unseen_idx = set([_dict_class_nodes_translation[c] for c in self.unseen_classes])
        kd_idx_to_class = {i: _dict_class_nodes_translation[c] for i, c in enumerate(s_u_idx)}
        if self.args.dataset == "lad" or self.args.dataset == "cub" or self.args.dataset == "awa2":
            kd_idx_to_class_idx = {i: _dict_class_nodes_translation[c] for i, c in enumerate(classes_attributes)}
            kd_idx_to_class = {i: _dict_class_nodes_translation[kd_idx_to_class_idx[i]] for i in
                               list(kd_idx_to_class.keys())}
            unseen_idx.update(seen_idx)
            seen_idx = set(
                [idx for idx in list(seen_idx) + list(unseen_idx) if kd_idx_to_class_idx[idx] < len(seen_idx)])
            unseen_idx -= seen_idx
        else:
            kd_idx_to_class_idx = {i: _dict_class_nodes_translation[self.dict_class_name[c]] for i, c in enumerate(classes_attributes)}
            kd_idx_to_class = {i: self.dict_name_class[_dict_class_nodes_translation[kd_idx_to_class_idx[i]]] for i in
                               list(kd_idx_to_class.keys())}
            unseen_idx.update(seen_idx)
            seen_idx = set([idx for idx in range(len(s_u_classes)) if kd_idx_to_class_idx[idx] in seen_idx])
            unseen_idx = set(range(len(s_u_classes))) - seen_idx
        attributes = np.array([all_attributes[idx] for idx in range(len(s_u_idx))])
        attributes = normalize(attributes, norm='l2', axis=1)
        return kd_idx_to_class_idx, kd_idx_to_class, seen_idx, unseen_idx, attributes

    @staticmethod
    def _display_kg_nodes(current_classes_nodes_df, index, class_name, class_attribute, class_kind):
        current_classes_nodes_df.loc[index] = [class_name, class_kind, class_attribute]
        index += 1
        return current_classes_nodes_df, index

    def attributed_graph(self, final_kg, _dict_class_nodes_translation, att_weight, _radius, jaccard=True):
        kd_idx_to_class_idx, kd_idx_to_class, seen_idx, unseen_idx, attributes = self._attributes()
        kdt = KDTree(attributes, leaf_size=10) if not jaccard else None
        # image_graph.add_nodes_from(np.arange(len(self.embeddings)))
        count = 0
        index = 0
        classes_nodes_df = pd.DataFrame(columns=["class", "class kind", "attribute"])
        for i in range(len(attributes)):
            if jaccard:
                neighbors, distances = calculate_weighted_jaccard_distance(attributes, attributes[i:i + 1],
                                                                           r=0.5)
            else:
                neighbors, distances = kdt.query_radius(attributes[i:i + 1], r=_radius["classes_radius"],
                                                        return_distance=True)
                neighbors, distances = neighbors[0], distances[0]
            if i in unseen_idx:
                classes_nodes_df, index = self._display_kg_nodes(classes_nodes_df, index, kd_idx_to_class[i],
                                                                 attributes[i:i + 1][0], "unseen class")
                k = max(len(neighbors) + 2, len(unseen_idx) + 2)

                if jaccard:
                    _neighbors, _distances = calculate_weighted_jaccard_distance(attributes, attributes[i:i + 1],
                                                                                 k=k)
                else:
                    _distances, _neighbors = kdt.query(attributes[i:i + 1], k=k, return_distance=True)
                    _distances, _neighbors = _distances[0], _neighbors[0]
                dict_neigh_dist = dict(zip(_neighbors, _distances))
                sorted_neigh_dist = dict(sorted(dict_neigh_dist.items(), key=lambda item: item[1]))
                sorted_relevant_seen_neighs = list(sorted_neigh_dist.keys())
                neighbors = set(neighbors).intersection(seen_idx)
                num_seen_neighbors = len(neighbors)
                if num_seen_neighbors < 3:
                    seen_neighs = set(sorted_relevant_seen_neighs).intersection(seen_idx)
                    dict_seen_neigh_dist = {seen_neigh: sorted_neigh_dist[seen_neigh] for seen_neigh in seen_neighs}
                    dict_seen_neigh_dist = dict(sorted(dict_seen_neigh_dist.items(), key=lambda item: item[1]))
                    seen_neighs = list(dict_seen_neigh_dist.keys())
                    neighbors.update(seen_neighs[:2])
                neighbors = list(neighbors)
                distances = [sorted_neigh_dist[neigh] for neigh in neighbors]
            elif len(neighbors) == 1:
                if jaccard:
                    neighbors, distances = calculate_weighted_jaccard_distance(attributes, attributes[i:i + 1], k=2)
                else:
                    distances, neighbors = kdt.query(attributes[i:i + 1], k=2,
                                                     return_distance=True)
                classes_nodes_df, index = self._display_kg_nodes(classes_nodes_df, index, kd_idx_to_class[i],
                                                                 attributes[i:i + 1][0], "seen class")
            else:
                classes_nodes_df, index = self._display_kg_nodes(classes_nodes_df, index, kd_idx_to_class[i],
                                                                 attributes[i:i + 1][0], "seen class")

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
            if self.args.dataset == "awa2_w_imagenet":
                neighbors_translation = [_dict_class_nodes_translation[self.dict_class_name[kd_idx_to_class[neighbor]]] for neighbor in
                                         neighbors]
                weight_edges = list(zip(np.repeat(_dict_class_nodes_translation[self.dict_class_name[kd_idx_to_class[i]]], len(neighbors)),
                                        neighbors_translation, edges_weights))
            else:
                neighbors_translation = [_dict_class_nodes_translation[kd_idx_to_class[neighbor]] for neighbor in
                                         neighbors]
                weight_edges = list(zip(np.repeat(_dict_class_nodes_translation[kd_idx_to_class[i]], len(neighbors)),
                                        neighbors_translation, edges_weights))
            final_kg.add_weighted_edges_from(weight_edges)
            # TODO: add the weight from the attributes to the pre graph and not replace them
            #  (minor problem because it is sparse graph)z
        classes_nodes_df_path = Path(f"{self.args.dataset}/plots/gephi/kg_classes_kind_and_attributes.csv", index=False)
        classes_nodes_df_path.parent.mkdir(parents=True, exist_ok=True)
        classes_nodes_df.to_csv(classes_nodes_df_path, index=False)
        if self.args.dataset == "awa2_w_imagenet":
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

    def create_labels_graph(self, _dict_class_nodes_translation):
        labels_graph = nx.Graph()
        if self.args.dataset == "awa2_w_imagenet":
            edges = np.array([(key, _dict_class_nodes_translation[self.dict_class_name[self.dict_idx_image_class[key]]])
                              # dict_class_nodes_translation ?
                              for key in list(self.dict_idx_image_class.keys())]).astype(str)
        elif self.args.dataset == "cub" or self.args.dataset == "lad" or self.args.dataset == "awa2":
            edges = np.array([(key, _dict_class_nodes_translation[self.dict_idx_image_class[key]])
                              for key in list(self.dict_idx_image_class.keys())]).astype(str)
        else:
            raise ValueError("Wrong dataset name: replace with awa2/cub/lad")
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


def define_graph_args(dataset_name):
    if dataset_name == "awa2_w_imagenet":
        _data_path = 'ZSL _DataSets/awa2/Animals_with_Attributes2'
        _split_path = 'materials/awa2-split.json'
        _chkpt_path = "save_data_graph/awa2"
        _model_path = "materials/resnet50-base.pth"
        _attributes_path = ""
        _pre_knowledge_graph_path = "materials/imagenet-induced-graph.json"
        _radius = {"images_radius": 0.10, "classes_radius": 1.15}
    elif dataset_name == "awa2":
        _data_path = 'ZSL _DataSets/awa2/Animals_with_Attributes2'
        _split_path = {"unseen": 'ZSL _DataSets/awa2/Animals_with_Attributes2/testclasses.txt',
                       "seen": "ZSL _DataSets/awa2/Animals_with_Attributes2/trainclasses.txt"}
        _chkpt_path = "save_data_graph/awa2"
        # _model_path = "materials/resnet50-base.pth"
        _model_path = "save_models/awa2"
        _attributes_path = "ZSL _DataSets/awa2/Animals_with_Attributes2/predicate-matrix-binary.txt"
        _pre_knowledge_graph_path = "materials/imagenet-induced-graph.json"
        _radius = {"images_radius": 0.015, "classes_radius": 1.15}
    elif dataset_name == "cub":
        _data_path = "ZSL _DataSets/cub/CUB_200_2011"
        _split_path = "ZSL _DataSets/cub/CUB_200_2011/train_test_split_easy.mat"
        _chkpt_path = 'save_data_graph/cub'
        _model_path = "save_models/cub"
        _attributes_path = "ZSL _DataSets/cub/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"
        _pre_knowledge_graph_path = ""
        _radius = {"images_radius": 0.04, "classes_radius": 0.6}
    elif dataset_name == "lad":
        _data_path = "ZSL _DataSets/lad"
        _split_path = "ZSL _DataSets/lad/split_zsl.txt"
        _chkpt_path = 'save_data_graph/lad'
        _model_path = "save_models/lad"
        _attributes_path = "ZSL _DataSets/lad/attributes_per_class.txt"
        _pre_knowledge_graph_path = ""
        _radius = {"images_radius": 0.018, "classes_radius": 0.9}
    else:
        raise ValueError("Wrong dataset name: replace with awa2/cub/lad")
    _dict_paths = {"data_path": _data_path, "split_path": _split_path, "save_path": _chkpt_path,
                   "model_path": _model_path, "attributes_path": _attributes_path,
                   "pre_knowledge_graph_path": _pre_knowledge_graph_path}
    return _dict_paths, _radius


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest="dataset", help=' Name of the dataset', type=str, default="awa2")
    parser.add_argument('--cnn', default='materials/resnet50-base.pth')
    # parser.add_argument('--cnn', default='save_awa2/resnet-fit/epoch-1.pth')
    # parser.add_argument('--pred', default='save_awa2/gcn-dense-att/epoch-30.pred')
    # parser.add_argument('--pred', default='save_awa2/gcn-basic/epoch-34.pred')
    # parser.add_argument('--consider-trains', action='store_true')
    parser.add_argument('--consider-trains', action='store_false')
    parser.add_argument("train_percentage", help="train percentage from the seen images", default=90)
    parser.add_argument('--output', default=None)
    parser.add_argument('--images_nodes_percentage', default=1)
    args = parser.parse_args()

    dict_paths, radius = define_graph_args(args.dataset)
    graph_preparation = ImagesEmbeddings(dict_paths, args)
    embeds_matrix, dict_image_embed, dict_image_class, val_images = graph_preparation.images_embed_calculator()
    dict_idx_image_class = {i: "c" + dict_image_class[image]
                            for i, image in enumerate(list(dict_image_class.keys()))}
    final_graph_creator = FinalGraphCreator(dict_paths, embeds_matrix, dict_image_embed,
                                            dict_idx_image_class, args.images_nodes_percentage, args)
    image_graph = final_graph_creator.create_image_graph(radius)
    kg, dict_class_nodes_translation = final_graph_creator.imagenet_knowledge_graph()
    att_weight = 10
    kg = final_graph_creator.attributed_graph(kg, dict_class_nodes_translation, att_weight, radius)
    seen_classes, unseen_classes = final_graph_creator.seen_classes, final_graph_creator.unseen_classes
    seen_classes = [dict_class_nodes_translation[c] for c in seen_classes]
    unseen_classes = [dict_class_nodes_translation[c] for c in unseen_classes]
    split = {'seen': seen_classes, 'unseen': unseen_classes}
    labels_graph = final_graph_creator.create_labels_graph(dict_class_nodes_translation)
    weights = [1, 1]
    weights_dict = {'classes_edges': weights[0], 'labels_edges': weights[1]}
    final_graph = final_graph_creator.weighted_graph(image_graph, kg, labels_graph, weights_dict)
    print("image graph edges & nodes: ", len(image_graph.edges), "&", len(image_graph.nodes))
    print("knowledge graph edges & nodes: ", len(kg.edges), "&", len(kg.nodes))
    print("labels graph edges & nodes: ", len(labels_graph.edges), "&", len(labels_graph.nodes))
    print("final graph edges & nodes: ", len(final_graph.edges), "&", len(final_graph.nodes))
