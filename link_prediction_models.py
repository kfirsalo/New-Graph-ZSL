import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import random
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation
import torch
from torch.backends import cudnn
import keras
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import nni
from copy import deepcopy

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.deterministic = True
random.seed(seed)


class TopKRanker(OneVsRestClassifier):
    """
    Linear regression with one-vs-rest classifier
    """

    def predict_kfir(self, x, top_k_list):
        assert x.shape[0] == len(top_k_list)
        probs = super(TopKRanker, self).predict_proba(x)
        prediction = np.zeros((x.shape[0], self.classes_.shape[0]))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
            for label in labels:
                prediction[i, int(label)] = 1
        return prediction, probs


def train_edge_classification(x_train, y_train):
    """
    train  the classifier with the train set.
    :param x_train: The features' edge- norm (train set).
    :param y_train: The edges labels- 0 for true, 1 for false (train set).
    :return: The classifier
    """
    model = LogisticRegression()
    parameters = {"penalty": ["l2"], "C": [0.01, 0.1, 1]}
    model = TopKRanker(
        GridSearchCV(model, param_grid=parameters, cv=2, scoring='balanced_accuracy', n_jobs=1, verbose=0,
                     pre_dispatch='n_jobs'))
    model.fit(x_train, y_train)
    return model


def predict_edge_classification(classif2, x_test):
    """
    With the test data make
    :param classif2:
    :param x_test:
    :return:
    """
    top_k_list = list(np.ones(len(x_test)).astype(int))
    prediction, probs = classif2.predict_kfir(x_test, top_k_list)
    if np.argmax(x_test) != np.argmax(probs.T[0]):
        print('stop')
    return prediction, probs


def create_keras_model(input_shape, hidden_layer_size):
    """
    :param input_shape: tuple - shape of a single sample (2d, 1)
    """

    model = tf.keras.Sequential()

    model.add(Dense(hidden_layer_size, activation="relu", input_shape=(input_shape,)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # opti = keras.optimizers.Adam(lr=0.01)
    opti = "Adam"
    model.compile(optimizer=opti, loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])
    model.summary()

    # parameters = {"learning_rate": [0.005, 0.01, 0.1]}
    # model = GridSearchCV(model, param_grid=parameters, cv=2, scoring='balanced_accuracy', n_jobs=1, verbose=0,
    #                      pre_dispatch='n_jobs')
    return model


def keras_model_fit(model, x_train, y):
    y_train = y[:, 0]
    tf.config.run_functions_eagerly(True)
    model.fit(x_train, y_train, epochs=5)
    print("done fitting")
    return model


def keras_model_predict(model, x_test):
    y_pred = model.predict(x_test)
    return y_pred


class EmbeddingLinkPredictionDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        # labels = labels[:, 0]
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class EmbeddingLinkPredictionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layer_dim: int, lr: float = 0.01, weight_decay=0.0, device="cpu"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_dim, 2)
        self.sigmoid = nn.Sigmoid()

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1, 10]))
        self.device = device
        self.to(self.device)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


class TrainLinkPrediction:
    def __init__(self, model, epochs, train_loader=None, val_loader=None, test_loader=None, to_nni=False):
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = model.device
        self.to_nni = to_nni

    def train(self):
        running_loss = 0.0
        accuracy = []
        best_val_accuracy = -1.0
        best_epoch = 0
        self.model.train()
        for epoch in range(self.epochs):

            for i, (embeddings, labels) in enumerate(self.train_loader):
                labels = torch.tensor(np.array(labels).astype(int), dtype=torch.float, device=self.device)
                embeddings = torch.tensor(np.array(embeddings), dtype=torch.float, device=self.device)
                self.model.optimizer.zero_grad()
                self.model.train()
                predictions = self.model(embeddings)
                # one_hot_labels = torch.zeros(predictions.shape)
                # one_hot_labels[torch.arange(predictions.shape[0]), torch.tensor(np.array(labels).astype(int), dtype=torch.long)] = 1
                loss = self.model.loss(predictions, labels).to(self.device)
                loss.backward()
                self.model.optimizer.step()
                running_loss += loss.item()
                final_preds = 1 - torch.argmax(predictions, dim=1)
                row_labels = labels[:, 0].cpu()
                samples_weight = row_labels*10 + 1 - row_labels
                accuracy.append(accuracy_score(row_labels, final_preds.cpu(), sample_weight=samples_weight))
            _, _, val_accuracy = self.eval()
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                best_classif = deepcopy(self.model)
            if self.to_nni:
                nni.report_intermediate_result(val_accuracy)
            else:
                print('num_epochs:{} || loss: {} || train accuracy: {} || val accuracy: {} '
                      .format(epoch, running_loss / len(self.train_loader), np.mean(accuracy[-9:]), val_accuracy))
                running_loss = 0.0
        if self.to_nni:
            nni.report_final_result({'default': best_val_accuracy, 'best_num_epochs': best_epoch})
        return best_classif

    def eval(self):
        self.model.eval()
        concat = False
        with torch.no_grad():
            for i, (embeddings, labels) in enumerate(self.val_loader):
                labels = torch.tensor(np.array(labels).astype(int), dtype=torch.float, device=self.device)
                embeddings = torch.tensor(np.array(embeddings), dtype=torch.float, device=self.device)
                self.model.eval()
                predictions = self.model(embeddings)
                predictions = 1 - torch.argmax(predictions, dim=1)
                if concat:
                    final_predictions = torch.cat((final_predictions, predictions))
                    all_labels = torch.cat((all_labels, labels))
                else:
                    final_predictions = predictions
                    all_labels = labels
                    concat = True
            all_row_labels = all_labels[:, 0].cpu()
            samples_weight = all_row_labels*10 + 1 - all_row_labels
        return all_labels.cpu(), final_predictions.cpu(), accuracy_score(all_row_labels, final_predictions.cpu(), sample_weight=samples_weight)

    def test(self):
        self.model.eval()
        concat = False
        with torch.no_grad():
            for i, (embeddings, labels) in enumerate(self.test_loader):
                embeddings = torch.tensor(np.array(embeddings), dtype=torch.float, device=self.device)
                self.model.eval()
                probs = self.model(embeddings)[:, 0]
                if concat:
                    final_probs = torch.cat((final_probs, probs))
                else:
                    final_probs = probs
                    concat = True
        return final_probs
