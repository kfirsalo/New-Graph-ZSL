import json
import csv
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize

def imagenet2csv():
    graph = json.load(open('materials/imagenet-induced-graph.json', 'r'))
    edges = graph['edges']
    nodes = graph['wnids']
    dict_class_nodes_translation = {node: i for i, node in enumerate(nodes)}
    nodes_idx = [dict_class_nodes_translation[node] for node in nodes]
    weight_edges = []
    for edge in edges:
        weight_edges.append([edge[0], edge[1], 1])
    awa2_split = json.load(open('materials/awa2-split.json', 'r'))
    seen_classes = awa2_split['train']
    unseen_classes = awa2_split['test']
    all_attributes = graph['vectors']
    dict_class_nodes = {node: i for i, node in enumerate(nodes)}
    dict_nodes_class = {i: node for i, node in enumerate(nodes)}
    dict_class_nodes_translation = {**dict_class_nodes, **dict_nodes_class}
    s_u_classes = seen_classes + unseen_classes
    s_u_idx = [dict_class_nodes_translation[c] for c in s_u_classes]
    kd_idx_to_class_idx = {i: dict_class_nodes_translation[c] for i, c in enumerate(s_u_classes)}
    attributes = np.array([all_attributes[idx] for idx in s_u_idx])
    attributes = normalize(attributes, norm='l2', axis=1)
    kdt = KDTree(attributes, leaf_size=10)
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
        edges_weights = [10 / dist if dist > 0 else 1000 for dist in distances]
        len_neigh = len(neighbors)
        if len_neigh == 0:
            print('hi Im number ' + str(i))
        count += len_neigh
        mean = count / (i + 1)
        if i % 10 == 0:
            print('Progress:', i, '/', len(attributes), ';  Current Mean:', mean)  # 37273
        neighbors_translation = [kd_idx_to_class_idx[neighbor] for neighbor in neighbors]
        weight_edges_to_add = list(
            zip(np.repeat(kd_idx_to_class_idx[i], len(neighbors)), neighbors_translation, edges_weights))
        weight_edges = weight_edges + weight_edges_to_add
    attributes = []
    for node in nodes:
        if len(set([node]).intersection(seen_classes)) > 0:
            attributes.append('s')  # seen
        if len(set([node]).intersection(unseen_classes)) > 0:
            attributes.append('u')  # unseen
        else:
            attributes.append('o')  # other

    np.savetxt('materials/imagenet_nodes.csv', [p for p in zip(nodes_idx, attributes)], delimiter=',', fmt='%s')
    np.savetxt('materials/weight_dense_imagenet_edges.csv', weight_edges, delimiter=',', fmt='%s')
    # print("bb")
    # with open('materials/imagenet_nodes.csv', mode='w') as employee_file:
    #     print("aa")
    #     # employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     for row in zip(nodes_idx, attributes):
    #         employee_file.write(row)
    # with open('materials/imagenet_edges.csv', mode='w') as employee_file:
    #     employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     for row in edges:
    #         employee_writer.writerow(row)


imagenet2csv()