from general_resnet50 import ResNet50
from image_folder import ImageFolder
from torch.utils.data import DataLoader
import os
import scipy.io as sio
from sklearn.metrics import accuracy_score
import torch
import numpy as np
from torch.backends import cudnn
import random
import argparse
from utils import get_device
import nni
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from utlis_graph_zsl import plot_confusion_matrix, classes_split

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.deterministic = True
random.seed(seed)


class ImagesEmbedding:
    def __init__(self, args, data_dir, split_dir, chkpt_dir, lr=0.01, train_percentage=90, epochs=50,
                 batch_size: int = 32, weight_decay=0.0, state_dict_path=None):
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.chkpt_dir = chkpt_dir
        self.args = args
        self.images_dir = os.path.join(data_dir, "images")
        self.epochs = epochs
        self.train_percentage = train_percentage
        self.lr = lr
        self.device = get_device()
        # self.classes = self.get_classes()
        self.seen_classes, self.unseen_classes = classes_split(self.args.dataset, self.data_dir, self.split_dir)
        self.batch_size = batch_size
        self.learning_model = ResNet50(out_dimension=len(self.seen_classes), chkpt_dir=self.chkpt_dir, lr=self.lr,
                                       weight_decay=weight_decay, device=self.device)
        if state_dict_path is not None:
            self.learning_model.load_state_dict(torch.load(state_dict_path))
        self.train_loader, self.val_loader, self.test_loader = self._prepare_dataloader()

    def _prepare_dataloader(self):
        seen_dataset = ImageFolder(self.images_dir, self.seen_classes, train_percentage=self.train_percentage,
                                   stage='train')
        train_dataloader = DataLoader(seen_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                      pin_memory=True)
        val_dataset = ImageFolder(self.images_dir, self.seen_classes, train_percentage=self.train_percentage,
                                  stage='val')
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True)
        unseen_dataset = ImageFolder(self.images_dir, self.unseen_classes, stage='test')
        unseen_dataloader = DataLoader(unseen_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4,
                                       pin_memory=True)
        return train_dataloader, val_dataloader, unseen_dataloader

    def get_classes(self):
        return np.array(os.listdir(self.images_dir))

    # def classes_split(self):
    #     if self.args.dataset == "cub":
    #         train_test_split = sio.loadmat(self.split_dir)
    #         classes = self.get_classes()
    #         seen_classes = classes[train_test_split['train_cid'] - 1][0]
    #         unseen_classes = classes[train_test_split['test_cid'] - 1][0]
    #     elif self.args.dataset == "lad":
    #         split_file = open(self.split_dir, "r")
    #         unseen_lists = {l.split(":")[0]: l.split(":")[1] for l in split_file.readlines()}
    #         unseen_list_1 = unseen_lists["Unseen_List_1"][:-1].replace(" ", "").split(",")
    #         classes_file = open(os.path.join(self.data_dir, "label_list.txt"), encoding="utf8")
    #         classes = classes_file.readlines()
    #         classes_translation = {**{c.split(", ")[0]: c.split(", ")[0].split("_")[1]+"_"+c.split(", ")[1]
    #                                   for c in classes}, **{c.split(", ")[0].split("_")[1]+"_"
    #                                                         +c.split(", ")[1]: c.split(", ")[0] for c in classes}}
    #         unseen_classes = np.array([classes_translation[c] for c in unseen_list_1])
    #         seen_classes = np.setdiff1d(self.get_classes(), unseen_classes)
    #     else:
    #         raise ValueError("Wrong dataset name: replace with cub/lad")
    #     return seen_classes, unseen_classes

    def train(self):
        running_loss = 0.0
        accuracy = []
        best_val_accuracy = 0.0
        best_epoch = 0
        self.learning_model.train()
        for epoch in range(self.epochs):

            for i, (images, labels) in enumerate(self.train_loader):
                labels = torch.tensor(np.array(labels).astype(int), dtype=torch.long, device=self.device)
                images = images.to(self.device)
                self.learning_model.optimizer.zero_grad()
                self.learning_model.train()
                predictions = self.learning_model(images)
                # one_hot_labels = torch.zeros(predictions.shape)
                # one_hot_labels[torch.arange(predictions.shape[0]), torch.tensor(np.array(labels).astype(int), dtype=torch.long)] = 1
                loss = self.learning_model.loss(predictions, labels).to(self.device)
                loss.backward()
                self.learning_model.optimizer.step()
                running_loss += loss.item()
                final_preds = torch.argmax(predictions, dim=1)
                accuracy.append(accuracy_score(labels.cpu(), final_preds.cpu()))
            _, _, val_accuracy = self.eval()
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                self.save_best_model()
            if args.nni:
                nni.report_intermediate_result(val_accuracy)
            else:
                print('num_epochs:{} || loss: {} || train accuracy: {} || val accuracy: {} '
                      .format(epoch, running_loss / len(self.train_loader), np.mean(accuracy[-9:]), val_accuracy))
                running_loss = 0.0
                self.save_model()
        if args.nni:
            nni.report_final_result({'default': best_val_accuracy, 'best_num_epochs': best_epoch})

    def eval(self):
        self.learning_model.eval()
        concat = False
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                labels = torch.tensor(np.array(labels).astype(int), dtype=torch.long, device=self.device)
                images = images.to(self.device)
                self.learning_model.eval()
                predictions = self.learning_model(images)
                predictions = torch.argmax(predictions, dim=1)
                if concat:
                    final_predictions = torch.cat((final_predictions, predictions))
                    all_labels = torch.cat((all_labels, labels))
                else:
                    final_predictions = predictions
                    all_labels = labels
                    concat = True
        return all_labels.cpu(), final_predictions.cpu(), accuracy_score(all_labels.cpu(), final_predictions.cpu())

    def save_model(self):
        self.learning_model.save_checkpoint()

    def save_best_model(self):
        self.learning_model.save_best()

    def load_models(self):
        self.learning_model.load_checkpoint()


def confusion_matrix_maker(dataset, gt, predictions):
    conf_matrix = confusion_matrix(gt, predictions)
    plt.figure(0)
    title = f'Confusion Matrix, Resnet50 Classification {dataset}'
    x_title = f"True Labels"
    y_title = f"Predicted Labels"
    plot_confusion_matrix(conf_matrix, title, x_title, y_title)
    plt.savefig(f'{dataset}/plots/confusion_matrix_ResNet50_{dataset}')


def define_path(dataset_name):
    if dataset_name == "cub":
        data_path = "ZSL _DataSets/cub/CUB_200_2011"
        split_path = "ZSL _DataSets/cub/CUB_200_2011/train_test_split_easy.mat"
        chkpt_path = 'save_models/cub'
    elif dataset_name == "lad":
        data_path = "ZSL _DataSets/lad"
        split_path = "ZSL _DataSets/lad/split_zsl.txt"
        chkpt_path = 'save_models/lad'
    else:
        raise ValueError("Wrong dataset name: replace with cub/lad")
    return data_path, split_path, chkpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet50 for Images Embedding")
    parser.add_argument('--nni', dest="nni", action='store_true',
                        help=' Whether to use nni')
    parser.add_argument('--dataset', dest="dataset", help=' Name of the dataset', type=str, default="cub")
    args = parser.parse_args()

    data_dir, split_dir, chkpt_dir = define_path(args.dataset)
    if args.nni:
        params = nni.get_next_parameter()
    else:
        params = {"lr": 0.0012694, "batch_size": 64, "weight_decay": 0.0000848412}
        print(f'--start training on {args.dataset}--')
    images_embedding = ImagesEmbedding(args, data_dir, split_dir, chkpt_dir, lr=params["lr"],
                                       batch_size=params["batch_size"], weight_decay=params["weight_decay"])
    images_embedding.train()
    gt, predictions, _ = images_embedding.eval()
    confusion_matrix_maker(args.dataset, gt, predictions)
