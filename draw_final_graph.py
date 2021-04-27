from images_graph_creator_all import FinalGraphCreator, ImagesEmbeddings, define_graph_args
import argparse
import numpy as np
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import os.path as osp
from utlis_graph_zsl import grid
from IMDb_data_preparation_E2V import MoviesGraph


class FinalGraphDrawing:
    def __init__(self, _args, _class_edges_weights, _labels_edges_weights, _attributes_weight):
        self.args = _args
        self.class_edges_weights = _class_edges_weights
        self.labels_edges_weights = _labels_edges_weights
        self.att_weight = attributes_weight
        self.final_graph, self.labels_graph, self.dict_class_identification = self.create_graph()
        self.dict_nodes_characterization = self.nodes_characterization()
        self.dict_embed = self.node2vec_embedding()

    def create_graph(self):
        weights_dict = {'classes_edges': self.class_edges_weights, 'labels_edges': self.labels_edges_weights}
        if self.args.dataset == "our_imdb":
            weights_dict["movies_edges"] = weights_dict["classes_edges"]
            dict_paths = {'cast': 'data_set/IMDb title_principals.csv', 'genre': 'data_set/IMDb movies.csv'}
            imdb = MoviesGraph(dict_paths, self.args.images_nodes_percentage)
            image_graph = imdb.create_graph()
            labels = imdb.labels2int(image_graph)
            labels_graph = imdb.create_labels_graph(labels)
            kg, knowledge_data = imdb.create_knowledge_graph(labels, float(0.3))
            final_graph = imdb.weighted_graph(image_graph, kg, labels, weights_dict)
            dict_true_edges = self.label_edges_classes_ordered(list(labels_graph.edges))
            classes = list(dict_true_edges.keys())
            for i, k in enumerate(sorted(dict_true_edges, key=lambda x: len(dict_true_edges[x]), reverse=True)):
                classes[i] = k
            seen_classes = classes[:int(0.8 * len(classes))]
            unseen_classes = classes[int(0.8 * len(classes)):]

        elif self.args.dataset == "cub" or self.args.dataset == "lad":
            dict_paths, radius = define_graph_args(self.args.dataset)
            graph_preparation = ImagesEmbeddings(dict_paths, self.args)
            embeds_matrix, dict_image_embed, dict_image_class = graph_preparation.images_embed_calculator()
            dict_idx_image_class = {i: dict_image_class[image]
                                    for i, image in enumerate(list(dict_image_class.keys()))}
            final_graph_creator = FinalGraphCreator(dict_paths, embeds_matrix, dict_image_embed,
                                                    dict_idx_image_class, self.args.images_nodes_percentage, self.args)
            image_graph = final_graph_creator.create_image_graph(radius)
            kg, dict_class_nodes_translation = final_graph_creator.imagenet_knowledge_graph()
            kg = final_graph_creator.attributed_graph(kg, dict_class_nodes_translation, self.att_weight, radius)
            seen_classes, unseen_classes = final_graph_creator.seen_classes, final_graph_creator.unseen_classes
            seen_classes = [dict_class_nodes_translation[c] for c in seen_classes]
            unseen_classes = [dict_class_nodes_translation[c] for c in unseen_classes]
            labels_graph = final_graph_creator.create_labels_graph(dict_class_nodes_translation)
            final_graph = final_graph_creator.weighted_graph(image_graph, kg, labels_graph, weights_dict)
        else:
            raise ValueError("Wrong dataset name: replace with awa2/cub/lad")
        dict_class_identification = {**{c: 0 for c in seen_classes}, **{c: 1 for c in unseen_classes}}
        print("image graph edges & nodes: ", len(image_graph.edges), "&", len(image_graph.nodes))
        print("knowledge graph edges & nodes: ", len(kg.edges), "&", len(kg.nodes))
        print("labels graph edges & nodes: ", len(labels_graph.edges), "&", len(labels_graph.nodes))
        print("final graph edges & nodes: ", len(final_graph.edges), "&", len(final_graph.nodes))
        return final_graph, labels_graph, dict_class_identification

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

    def nodes_characterization(self):
        dict_nodes_characterization = {}
        for edge in list(self.labels_graph.edges):
            if edge[0][0] == "c":
                dict_nodes_characterization[edge[0]] = self.dict_class_identification[edge[0]]
                dict_nodes_characterization[edge[1]] = self.dict_class_identification[edge[0]] + 2
            else:
                dict_nodes_characterization[edge[0]] = self.dict_class_identification[edge[1]] + 2
                dict_nodes_characterization[edge[1]] = self.dict_class_identification[edge[1]]
        return dict_nodes_characterization

    def node2vec_embedding(self):
        node2vec = Node2Vec(self.final_graph, dimensions=2, walk_length=80, num_walks=16, workers=2)
        model = node2vec.fit()
        nodes = list(self.final_graph.nodes())
        dict_embeddings = {}
        for i in range(len(nodes)):
            dict_embeddings.update({nodes[i]: np.asarray(model.wv.get_vector(str(nodes[i])))})
        return dict_embeddings

    def plot_graph(self, title):
        plt.figure()
        class_seen_points, class_unseen_points, image_seen_points, image_unseen_points, point_id = [], [], [], [], []
        for node in list(self.dict_embed.keys()):
            if self.dict_nodes_characterization[node] == 0:
                class_seen_points.append([*self.dict_embed[node]])
                point_id.append(0)
            elif self.dict_nodes_characterization[node] == 1:
                class_unseen_points.append([*self.dict_embed[node]])
                point_id.append(1)
            elif self.dict_nodes_characterization[node] == 2:
                image_seen_points.append([*self.dict_embed[node]])
                point_id.append(2)
            elif self.dict_nodes_characterization[node] == 3:
                image_unseen_points.append([*self.dict_embed[node]])
                point_id.append(3)
        plt.scatter(list(zip(*image_seen_points))[0], list(zip(*image_seen_points))[1], c='yellow', label='seen_images')
        plt.scatter(list(zip(*image_unseen_points))[0], list(zip(*image_unseen_points))[1], c='mediumspringgreen', label='unseen_images')
        plt.scatter(list(zip(*class_seen_points))[0], list(zip(*class_seen_points))[1], c='b', label='seen_classes')
        plt.scatter(list(zip(*class_unseen_points))[0], list(zip(*class_unseen_points))[1], c='r', label='unseen_classes')
        labels = ["seen_images", "unseen_images", "seen_classes", "unseen_classes"]
        plt.legend(labels, loc='best', ncol=2, fontsize='large')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(osp.join(f'{self.args.dataset}/plots', title))
        plt.close()

if __name__ == "__main__":
    class_edges_weights = 1
    parameters = {"dataset": ["our_imdb"],  "attributes_weight": [1, 10, 100],
                  "labels_edges_weights": [1, 10, 100]}
    for param in grid(parameters):
        dict_param = {p: param[i] for i, p in enumerate(list(parameters.keys()))}
        attributes_weight = dict_param["attributes_weight"]
        labels_edges_weights = dict_param["labels_edges_weights"]
        data = dict_param["dataset"]
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', dest="dataset", help=' Name of the dataset', type=str, default=data)

        parser.add_argument('--images_nodes_percentage', default=0.10)
        args = parser.parse_args()
        title = f'{args.dataset} Graph With {attributes_weight} times Attributes and ' \
                f'{labels_edges_weights} times label edges weight'
        final_graph_drawing = FinalGraphDrawing(args, class_edges_weights, labels_edges_weights, attributes_weight)
        final_graph_drawing.plot_graph(title)