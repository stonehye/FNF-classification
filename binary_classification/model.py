import torch
import torch.nn as nn

from torchvision import models
from collections import OrderedDict


class Resnet50(nn.Module):
    def __init__(self, n_class = 3, freeze_backbone = True):
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, n_class)

        self.init_fc()
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


    def init_fc(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


class MobileNet(nn.Module):
    def __init__(self, n_class = 2, freeze_backbone = True):
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280 , n_class)

        self.init_fc()
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


    def init_fc(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)