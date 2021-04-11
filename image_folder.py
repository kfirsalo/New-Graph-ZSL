import os
import os.path as osp

from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np


class ImageFolder(Dataset):
    def __init__(self, path, classes, train_percentage=100, stage='train'):
        self.data = []
        for i, c in enumerate(classes):
            cls_path = osp.join(path, c)
            images = os.listdir(cls_path)
            for image in images:
                self.data.append((osp.join(cls_path, image), i))
        self.data = np.array(self.data)
        if stage == "train":
            self.data = self.data[np.linspace(0, len(self.data)-1, int(0.01*train_percentage*len(self.data))).astype(int)]
        if stage == "val":
            train_indices = np.linspace(0, len(self.data)-1, int(0.01*train_percentage*len(self.data))).astype(int)
            self.data = self.data[np.setdiff1d(np.arange(len(self.data)), train_indices)]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        if stage == 'train':
            self.transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  normalize])
        if stage == 'test' or stage == 'val':
            self.transforms = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  normalize])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i]
        image = Image.open(path).convert('RGB')
        image = self.transforms(image)
        if image.shape[0] != 3 or image.shape[1] != 224 or image.shape[2] != 224:
            print('you should delete this guy:', path)
        return image, label

