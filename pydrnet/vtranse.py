import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
import random
from crf import CRF


class VtransE(nn.Module):

  def __init__(self, backbone='vgg16'):
    super(VtransE, self).__init__()

    self.backbone = models.__dict__[backbone](pretrained=True)
    self.backbone = nn.Sequential(*(list(self.backbone.features)[:-2]))
 
    feature_dim = 100 + 4  + 7 * 7 * 512  
    self.scale_factor = nn.Parameter(torch.Tensor(feature_dim))
    nn.init.uniform(self.scale_factor)  
    self.W_o = nn.Linear(feature_dim, 70, bias=False)
    self.W_s = nn.Linear(feature_dim, 70, bias=False)
    self.crf = CRF()


  def forward(self, subj, obj, full_im, t_s, t_o, bbox_s, bbox_o):
    img_feature_map = self.backbone(full_im)
    # (y_min, x_min, y_max, x_max)
    subj_img_feature = []
    obj_img_feature = []
    for i in range(bbox_s.size(0)):
      bbox_subj = self.fix_bbox(torch.round(14 * bbox_s[i]).int().data.cpu().numpy())
      bbox_obj = self.fix_bbox(torch.round(14 * bbox_o[i]).int().data.cpu().numpy())
      subj_img_feature.append(F.upsample_bilinear(
       img_feature_map[i, :, bbox_subj[1] : bbox_subj[3], bbox_subj[0] : bbox_subj[2]].unsqueeze(0), 7))
      obj_img_feature.append(F.upsample_bilinear(
       img_feature_map[i, :, bbox_obj[1] : bbox_obj[3], bbox_obj[0] : bbox_obj[2]].unsqueeze(0), 7))
    subj_img_feature = torch.cat(subj_img_feature)
    obj_img_feature = torch.cat(obj_img_feature)
    subj_img_feature = subj_img_feature.view(subj_img_feature.size(0), -1)
    obj_img_feature = obj_img_feature.view(obj_img_feature.size(0), -1)
    x_s = torch.cat([subj, t_s, subj_img_feature], 1) * self.scale_factor
    x_o = torch.cat([obj, t_o, obj_img_feature], 1) * self.scale_factor
    return self.W_o(x_o) - self.W_s(x_s)
    #return self.crf(self.W_o(x_o) - self.W_s(x_s))

  
  def fix_bbox(self, bbox):
    if bbox[0] == bbox[2]:
      if bbox[0] == 0:
        bbox[2] += 1
      elif bbox[2] == 14:
        bbox[0] -= 1
      else:
        if random.random() < 0.5:
          bbox[0] -= 1
        else:
          bbox[2] += 1
    if bbox[1] == bbox[3]:
      if bbox[1] == 0:
        bbox[3] += 1
      elif bbox[3] == 14:
        bbox[1] -= 1
      else:
        if random.random() < 0.5:
          bbox[1] -= 1
        else:
          bbox[3] += 1     
    return bbox
