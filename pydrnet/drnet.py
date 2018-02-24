import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict


class DRNet(nn.Module):

  def __init__(self, backbone='vgg16'):
    super(DRNet, self).__init__()

    self.appr_module = models.__dict__[backbone](pretrained=True)
    self.appr_module.classifier = nn.Sequential(*(list(self.appr_module.classifier)[:-1] + [nn.Linear(4096, 256), nn.ReLU()]))
    self.pos_module = nn.Sequential(OrderedDict([
      ('conv1_p', nn.Conv2d(2, 96, 5, 2, 2)),
      ('relu1_p', nn.ReLU()),
      ('conv2_p', nn.Conv2d(96, 128, 5, 2, 2)),
      ('conv3_p', nn.Conv2d(128, 64, 8)),
      ('relu3_p', nn.ReLU()), 
    ]))

    self.fc2_c = nn.Linear(256 + 64, 128)
    self.PhiR_0 = nn.Linear(128, 70)

    self.PhiA = nn.Linear(100, 70)
    self.PhiB = nn.Linear(100, 70)
    self.PhiR = nn.Linear(70, 70)

  def forward(self, qa, qb, im, posdata):
    appr_feature = self.appr_module(im)
    pos_feature = self.pos_module(posdata)
    if pos_feature.size(0) == 1:
      pos_feature = torch.unsqueeze(torch.squeeze(pos_feature), 0)
    else:
      pos_feature = torch.squeeze(pos_feature)
    pos_appr = torch.cat([appr_feature, pos_feature], 1)
    pos_appr = F.relu(self.fc2_c(pos_appr))
    qr = F.relu(self.PhiR_0(pos_appr))    

    
    for i in range(8):
      qr = self.PhiA(qa) + self.PhiB(qb) + self.PhiR(qr)
      if i < 7:
        qr = F.relu(qr)
    return qr
