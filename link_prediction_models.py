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


