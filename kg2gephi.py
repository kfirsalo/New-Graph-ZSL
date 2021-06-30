import networkx as nx
import pandas as pd


class KG2Gephi:
    def __init__(self, kg, seen_classes, unseen_classes):
        self.kg = kg
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.nodes = list(kg.nodes)
        self.edges_attributes = self.get_edge_attributes()
        self.nodes_attributes = self.get_nodes_attributes()

    def extract_adjacency_matrix(self):
        adj_mat = nx.linalg.graphmatrix.adjacency_matrix(self.kg).todense()
        return adj_mat

    def get_edge_attributes(self, name="weight"):
        edges = self.kg.edges(data=True)
        return [(x[:-1][0], x[:-1][1], x[-1][name]) for x in edges if name in x[-1]]

    def get_nodes_attributes(self):
        attributes = []
        for node in self.nodes:
            if len(set([node]).intersection(self.seen_classes)) > 0:
                attributes.append((node, 'seen class'))  # seen
            if len(set([node]).intersection(self.unseen_classes)) > 0:
                attributes.append((node, 'unseen class'))  # unseen
        return attributes

    def extract_kg_csv(self, edges_path, nodes_path, nodes_translate=None):
        edges = pd.DataFrame(data=self.edges_attributes, columns=["source", "target", "weight"])
        nodes = pd.DataFrame(data=self.nodes_attributes, columns=["Id", "attribute"])
        if nodes_translate is not None:
            edges['source'] = edges['source'].apply(lambda node: nodes_translate[node])
            edges['target'] = edges['target'].apply(lambda node: nodes_translate[node])
            nodes['Id'] = nodes['Id'].apply(lambda node: nodes_translate[node])
        edges.to_csv(edges_path, index=False)
        nodes.to_csv(nodes_path, index=False)

