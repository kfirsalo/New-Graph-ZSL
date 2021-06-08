import os
import pickle
import random
import numpy as np

class EdgesPreparation:
    def __init__(self, graph, dict_val_edges, args, split=None):
        self.dataset = args.dataset
        self.ratio = args.ratio
        self.false_per_true = args.false_per_true
        self.seen_percentage = args.seen_percentage
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
        data_path = self.dataset + '_true_edges.pickle'
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
            with open(os.path.join(self.dataset, data_path), 'wb') as handle:
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
        ratio = self.ratio[0]
        dict_true_edges = self.label_edges_classes_ordered(self.label_edges)
        classes = list(dict_true_edges.keys())
        for i, k in enumerate(sorted(dict_true_edges, key=lambda x: len(dict_true_edges[x]), reverse=True)):
            classes[i] = k
        seen_classes = classes[:int(self.seen_percentage * len(classes))]
        unseen_classes = classes[int(self.seen_percentage * len(classes)):]
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
        data_path = self.dataset + '_false_edges_balanced_{}.pickle'.format(self.false_per_true)
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
                if len(false_labels) < self.false_per_true + 1:
                    false_labels = list(set(labels) - set([label]))  # The set makes every run different edges.
                else:
                    false_labels = list(set(false_labels) - set([label]))
                indexes = random.sample(range(1, len(false_labels)), self.false_per_true)
                # random.Random(4).shuffle(false_labels)
                # false_labels = false_labels[:self.args.false_per_true + 1]
                for i, index in enumerate(indexes):
                    if dict_class_false_edges.get(label) is None:
                        dict_class_false_edges[label] = [[movie, false_labels[index]]]
                    else:
                        edges = dict_class_false_edges[label]
                        edges.append([movie, false_labels[index]])
                        dict_class_false_edges[label] = edges
                false_labels = list(np.delete(np.array(false_labels), indexes))
        try:
            with open(os.path.join(self.dataset, data_path), 'wb') as handle:
                pickle.dump(dict_class_false_edges, handle, protocol=3)
        except:
            pass
        return dict_class_false_edges
