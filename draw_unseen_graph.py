from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


class DrawUnseenGraph:
    def __init__(self, dict_embeds, dict_train, dict_test, dict_unseen, kind="tsne", dataset="our_imdb", args=None):
        self.dict_embeds = dict_embeds
        self.dict_train = dict_train
        self.dict_test = dict_test
        self.dict_unseen = dict_unseen
        self.seen_classes, self.unseen_classes, self.unseen_samples, self.train_samples, self.test_samples = \
            self._prepare_nodes_embeds()
        self.kind = kind
        self.dataset = dataset
        self.args = args
        self.dict_tsne_embeds = self._dimension_reduction(kind=self.kind)
        # self.train_class_points, self.test_class_points, self.unseen_class_points, self.unseen_samples_points = \
        #     self._separate_embeds()

    def _prepare_nodes_embeds(self):
        unseen_samples = set([])
        train_samples = set([])
        test_samples = set([])
        seen_classes = set(self.dict_train.keys())
        unseen_classes = set(self.dict_unseen.keys())
        const = 0

        for seen_class in list(seen_classes):
            train_edge = self.dict_train[seen_class]
            sample = train_edge[const][1] if train_edge[const][0][0] == 'c' else train_edge[const][0]
            train_samples.update([sample])
        for seen_class in list(set(self.dict_test.keys())):
            test_edge = self.dict_test[seen_class]
            sample = test_edge[const][1] if test_edge[const][0][0] == 'c' else test_edge[const][0]
            test_samples.update([sample])
        for unseen_class in list(self.dict_unseen.keys()):
            edge = self.dict_unseen[unseen_class]
            sample = edge[const][1] if edge[const][0][0] == 'c' else edge[const][0]
            unseen_samples.update([sample])
        return seen_classes, unseen_classes, unseen_samples, train_samples, test_samples

    def _dimension_reduction(self, kind="tsne"):
        if kind == "tsne":
            model = TSNE(n_components=2)
        elif kind == "pca":
            model = PCA(n_components=2)
        else:
            raise ValueError("options: tsne/pca")
        tsne_embeds = model.fit_transform(np.array(list(self.dict_embeds.values())))
        nodes = list(self.dict_embeds.keys())
        dict_tsne_embeds = {nodes[i]: tsne_embeds[i] for i in range(len(nodes))}
        return dict_tsne_embeds

    def _fit_embeds(self, nodes):
        dict_nodes_tsne_embeds = {node: self.dict_tsne_embeds[node] for node in nodes}
        return np.array(list(dict_nodes_tsne_embeds.values()))

    @staticmethod
    def _fit_label(nodes, label: int):
        return np.zeros(len(nodes)) + label

    def _prepare_data(self):
        seen_class_points, seen_class_labels = self._fit_embeds(self.seen_classes), self._fit_label(
            self.seen_classes, label=0)
        unseen_class_points, unseen_class_labels = self._fit_embeds(self.unseen_classes), self._fit_label(
            self.unseen_classes, label=1)
        unseen_samples_points, unseen_samples_labels = self._fit_embeds(self.unseen_samples), self._fit_label(
            self.unseen_samples, label=2)
        train_samples_points, train_samples_labels = self._fit_embeds(self.train_samples), self._fit_label(
            self.train_samples, label=3)
        test_samples_points, test_samples_labels = self._fit_embeds(self.test_samples), self._fit_label(
            self.test_samples, label=4)
        feat_cols = ["Principal Component 1", "Principal Component 2"]
        data = np.concatenate((seen_class_points, unseen_class_points, unseen_samples_points, train_samples_points,
                               test_samples_points))
        labels = np.concatenate((seen_class_labels, unseen_class_labels, unseen_samples_labels, train_samples_labels,
                                 test_samples_labels))
        labels_translate = {0: "seen classes", 1: "unseen classes", 3: "train samples",
                            4: "test samples", 2: "unseen samples"}
        style_translate = {0: "classes", 1: "classes", 2: "samples", 3: "samples", 4: "samples"}
        df = pd.DataFrame(data, columns=feat_cols)
        df['int_labels'] = labels
        df['labels'] = df['int_labels'].apply(lambda i: labels_translate[i])
        df['node kind'] = df['int_labels'].apply(lambda i: style_translate[i])
        return df, labels_translate, style_translate

    def draw_graph(self):
        mpl.rcParams['xtick.labelsize'] = 14
        mpl.rcParams['ytick.labelsize'] = 18
        mpl.rcParams['axes.titlesize'] = 20
        mpl.rcParams['axes.labelsize'] = 18
        plt.rcParams["font.family"] = "Times New Roman"
        data, label_translate, style_translate = self._prepare_data()
        style_order = list(style_translate.values())[0], list(style_translate.values())[-1]
        data = data.sort_values("Principal Component 1")
        plt.figure(figsize=(16, 10))
        plt.title(f"{self.kind.upper()} Visualization of {self.args.embedding} Embedding Applied on {self.dataset.upper()} Dataset")
        sns.scatterplot(
            x="Principal Component 1", y="Principal Component 2",
            hue="labels",
            style="node kind",
            palette=['green', 'orange', 'blue', 'dodgerblue', 'red'],
            data=data,
            hue_order=label_translate.values(),
            style_order=style_order)
        base_path = Path(f"{self.dataset}/unseen_plots/{self.args.embedding}")
        if self.args.embedding == "OGRE":
            visualization_path = base_path.joinpath(f"unseen_graph_{self.kind}_visualization_type={self.args.embedding}"
                                                    f"_dim={self.args.embedding_dimension}_label_weight="
                                                    f"{self.args.label_edges_weight}_instance_weight="
                                                    f"{self.args.instance_edges_weight}_ogre_second_neighbor_advantage="
                                                    f"{self.args.ogre_second_neighbor_advantage}.pdf")
        else:
            visualization_path = base_path.joinpath(f"unseen_graph_{self.kind}_visualization_type={self.args.embedding}"
                                                    f"_dim={self.args.embedding_dimension}_label_weight="
                                                    f"{self.args.label_edges_weight}_instance_weight="
                                                    f"{self.args.instance_edges_weight}.pdf")
        # visualization_path = Path(
        #     f"{self.dataset}/unseen_plots/{self.args.embedding}/unseen_graph_{self.kind}_visualization_"
        #     f"type={self.args.embedding}_dim={self.args.embedding_dimension}_label_weight={self.args.label_edges_weight}_"
        #     f"instance_weight={self.args.instance_edges_weight}.png")
        visualization_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(visualization_path)
        plt.close()
        # plt.scatter(self.unseen_samples_points[0], y_other[1], c='yellow', label='other')
        # plt.scatter(x_seen, y_seen, c='b', label='seen')
        # plt.scatter(x_unseen, y_unseen, c='r', label='unseen')
        # plt.legend(list(dict_att.values()), loc='best', ncol=3, fontsize='large')
        # plt.title(title)
        # plt.tight_layout()
        # plt.savefig(osp.join('awa2/plots', title))
