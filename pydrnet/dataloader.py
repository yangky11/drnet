import numpy as np
import math
import os
import cv2
cv2.setNumThreads(0)
from torch.utils.data import Dataset, DataLoader
import json
import random
from progressbar import ProgressBar
import sys


class VRD(Dataset):
  
  def __init__(self, datapath, num_classes=100, img_size=224, heatmap_size=32):
    super(VRD, self).__init__()
    self.datapath = datapath
    self.num_classes = num_classes
    self.img_size = img_size
    self.heatmap_size = heatmap_size

    self.samples = json.load(open(datapath, 'rt'))
    random.shuffle(self.samples)


  def __len__(self):
    return len(self.samples)


  def __getitem__(self, idx):
    im = cv2.imread(self.samples[idx]['imPath']).astype(np.float32, copy=False)
    ih = im.shape[0]
    iw = im.shape[1]
    qa = np.zeros(self.num_classes, dtype=np.float32)
    qa[self.samples[idx]['aLabel'] - 1] = 1
    qb = np.zeros(self.num_classes, dtype=np.float32)
    qb[self.samples[idx]['bLabel'] - 1] = 1
    
    aBBox = self.samples[idx]['aBBox']
    bBBox = self.samples[idx]['bBBox']
    posdata = np.stack([self._getDualMask(ih, iw, aBBox),
                        self._getDualMask(ih, iw, bBBox)])
    # (y_min, x_min, y_max, x_max)
    t_s = [(aBBox[0] - bBBox[0]) / float(bBBox[2] - bBBox[0]), 
           (aBBox[1] - bBBox[1]) / float(bBBox[3] - bBBox[1]),
           math.log((aBBox[2] - aBBox[0]) / float(bBBox[2] - bBBox[0])),
           math.log((aBBox[3] - aBBox[1]) / float(bBBox[3] - bBBox[1]))]
    t_o = [(bBBox[0] - aBBox[0]) / float(aBBox[2] - aBBox[0]), 
           (bBBox[1] - aBBox[1]) / float(aBBox[3] - aBBox[1]),
           math.log((bBBox[2] - bBBox[0]) / float(aBBox[2] - aBBox[0])),
           math.log((bBBox[3] - bBBox[1]) / float(aBBox[3] - aBBox[1]))]
    sample = {'full_im': self._getAppr(im, [0, 0, iw, ih]),
              't_s': np.asarray(t_s, dtype=np.float32),
              't_o': np.asarray(t_o, dtype=np.float32),
              'bbox_s': np.asarray([aBBox[0] / float(iw), aBBox[1] / float(ih), aBBox[2] / float(iw), aBBox[3] / float(ih)], 
                                   dtype=np.float32),
              'bbox_o': np.asarray([bBBox[0] / float(iw), bBBox[1] / float(ih), bBBox[2] / float(iw), bBBox[3] / float(ih)], 
                                   dtype=np.float32),
              'im': self._getAppr(im, self.samples[idx]['rBBox']), 
              'qa': qa, 
              'qb': qb,
              'posdata': posdata,
              'label': self.samples[idx]['rLabel']}
    return sample

  def _getAppr(self, im, bb):
    subim = im[bb[1] : bb[3], bb[0] : bb[2], :]
    subim = cv2.resize(subim, None, None, 224.0 / subim.shape[1], 224.0 / subim.shape[0], interpolation=cv2.INTER_LINEAR)
    pixel_means = np.array([[[103.939, 116.779, 123.68]]])
    subim -= pixel_means
    subim = subim.transpose((2, 0, 1))
    return subim


  def _getDualMask(self, ih, iw, bb):
    rh = float(self.heatmap_size) / ih
    rw = float(self.heatmap_size) / iw
    x1 = max(0, int(math.floor(bb[0] * rw)))
    x2 = min(self.heatmap_size, int(math.ceil(bb[2] * rw)))
    y1 = max(0, int(math.floor(bb[1] * rh)))
    y2 = min(self.heatmap_size, int(math.ceil(bb[3] * rh)))
    mask = np.zeros((self.heatmap_size, self.heatmap_size), dtype=np.float32)
    mask[y1 : y2, x1 : x2] = 1
    assert(mask.sum() == (y2 - y1) * (x2 - x1))
    return mask		


def dataloader(split, batchsize):
  ds = VRD('rel%s.json' % split)
  return DataLoader(ds, batchsize, shuffle=split.startswith('train'), pin_memory=True, num_workers=4)


if __name__ == '__main__':
  loader = dataloader('train', 25)
  while True:
    bar = ProgressBar(max_value=len(loader))
    for idx, batch in enumerate(loader):
      if idx == 0:
        print(batch)
      bar.update(idx)
