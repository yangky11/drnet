import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
import random


class V(nn.Module):

  def __init__(self, backbone='vgg16'):
    super(V, self).__init__()

    self.backbone = models.__dict__[backbone](pretrained=True)
    self.backbone.classifier = nn.Sequential(*(list(self.backbone.classifier)[:-1] + [nn.Linear(4096, 70)]))

  def forward(self, im):
    return self.backbone(im)
