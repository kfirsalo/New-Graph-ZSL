from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
from utils import get_device
import torch
import os

class ResNet50(nn.Module):
    def __init__(self, out_dimension: int, chkpt_dir: str, lr: float = 0.01, weight_decay=0.0):
        super().__init__()

        self.out_dimension = out_dimension
        self.lr = lr
        self.name = "ResNet50"
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)
        self.resnet50 = resnet50(pretrained=True)
        self.freeze()
        self.in_fc_features = self.resnet50.fc.in_features

        self.resnet50.fc = nn.Sequential(
            nn.Linear(in_features=self.in_fc_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.out_dimension),
            nn.LogSoftmax())

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.loss = nn.NLLLoss()
        self.device = get_device()
        self.to(self.device)

    def freeze(self):
        c = 0
        l = 0
        num_layer = 0
        for _ in self.resnet50.layer4.parameters():
            num_layer += 1
        for _ in self.resnet50.parameters():
            l += 1
        for params in self.resnet50.parameters():
            if c < l - num_layer - 2:
                params.requires_grad = False
            c += 1

    def forward(self, x):
        x = self.resnet50(x)
        # x = self.resnet50.avgpool(x).squeeze()
        # x = self.resnet50.fc(x)
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_best')
        torch.save(self.state_dict(), checkpoint_file)