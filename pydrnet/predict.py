import torch
import math
import numpy as np
import pickle
from progressbar import ProgressBar
import cv2
import json
import torch.autograd as autograd


def getUnionBBox(aBB, bBB, ih, iw):
  margin = 10
  return [max(0, min(aBB[0], bBB[0]) - margin), \
          max(0, min(aBB[1], bBB[1]) - margin), \
          min(iw, max(aBB[2], bBB[2]) + margin), \
          min(ih, max(aBB[3], bBB[3]) + margin)]


def getAppr(im, bb):
  subim = im[bb[1] : bb[3], bb[0] : bb[2], :]
  subim = cv2.resize(subim, None, None, 224.0 / subim.shape[1], 224.0 / subim.shape[0], interpolation=cv2.INTER_LINEAR)
  pixel_means = np.array([[[103.939, 116.779, 123.68]]])
  subim -= pixel_means
  subim = subim.transpose((2, 0, 1))
  return subim


def getDualMask(ih, iw, bb):
  rh = 32.0 / ih
  rw = 32.0 / iw
  x1 = max(0, int(math.floor(bb[0] * rw)))
  x2 = min(32, int(math.ceil(bb[2] * rw)))
  y1 = max(0, int(math.floor(bb[1] * rh)))
  y2 = min(32, int(math.ceil(bb[3] * rh)))
  mask = np.zeros((32, 32))
  mask[y1 : y2, x1 : x2] = 1
  assert(mask.sum() == (y2 - y1) * (x2 - x1))
  return mask


def predict(model, output_file, args, image_paths_file='image_paths_vrd.json', gt_file='gt_vrd.pickle'):
  model.eval()
  image_paths = json.load(open(image_paths_file, 'rt'))
  all_gts, all_gt_bboxes = pickle.load(open(gt_file, 'rb'))
  num_img = len(image_paths)
  num_classes = 100
  thresh = 0.05
  batch_size = 20
  pred = []
  pred_bboxes = []
  bar = ProgressBar(max_value=num_img)
  for i in range(num_img):
    bar.update(i)
    im = cv2.imread(image_paths[i]).astype(np.float32, copy=False)
    ih = im.shape[0]
    iw = im.shape[1]
    gts = all_gts[i]
    gt_bboxes = all_gt_bboxes[i]
    num_gts = gts.shape[0]
    pred.append([])
    pred_bboxes.append([])
    ims = []
    poses = []
    qas = []
    qbs = []
    t_s = []
    t_o = []
    bbox_s = []
    bbox_o = []
    for j in range(num_gts):
      sub = gt_bboxes[j, 0, :]
      obj = gt_bboxes[j, 1, :]
      rBB = getUnionBBox(sub, obj, ih, iw)
      if args.model == 'drnet':
        rAppr = getAppr(im, rBB)
      else:
        rAppr = getAppr(im, [0, 0, iw, ih])
      rMask = np.array([getDualMask(ih, iw, sub), getDualMask(ih, iw, obj)])
      ims.append(rAppr)
      poses.append(rMask)
      qa = np.zeros(num_classes)
      qa[gts[j, 0] - 1] = 1
      qb = np.zeros(num_classes)
      qb[gts[j, 2] - 1] = 1
      qas.append(qa)
      qbs.append(qb)
      if args.model == 'vtranse':
        t_s.append(np.asarray([(sub[0] - obj[0]) / float(obj[2] - obj[0]),
                    (sub[1] - obj[1]) / float(obj[3] - obj[1]),
                    math.log((sub[2] - sub[0]) / float(obj[2] - obj[0])),
                    math.log((sub[3] - sub[1]) / float(obj[3] - obj[1]))], dtype=np.float32))
        t_o.append(np.asarray([(obj[0] - sub[0]) / float(sub[2] - sub[0]),
                    (obj[1] - sub[1]) / float(sub[3] - sub[1]),
                    math.log((obj[2] - obj[0]) / float(sub[2] - sub[0])),
                    math.log((obj[3] - obj[1]) / float(sub[3] - sub[1]))], dtype=np.float32))
        bbox_s.append(np.asarray([sub[0] / float(iw), sub[1] / float(ih), sub[2] / float(iw), sub[3] / float(ih)],
                                   dtype=np.float32))
        bbox_o.append(np.asarray([obj[0] / float(iw), obj[1] / float(ih), obj[2] / float(iw), obj[3] / float(ih)],
                                   dtype=np.float32))

    if len(ims) == 0:
      continue
    ims = np.array(ims)
    poses = np.array(poses)
    qas = np.array(qas)
    qbs = np.array(qbs)
    if args.model == 'vtranse':
      t_s = np.stack(t_s)
      t_o = np.stack(t_o)
      bbox_s = np.stack(bbox_s)
      bbox_o = np.stack(bbox_o)
    _cursor = 0
    itr_pred = None
    num_ins = ims.shape[0]
    while _cursor < num_ins:
      _end_batch = min(_cursor + batch_size, num_ins)
      im_batch = autograd.Variable(torch.Tensor(ims[_cursor : _end_batch]).cuda(async=True), volatile=True)
      qa_batch = autograd.Variable(torch.Tensor(qas[_cursor : _end_batch]).cuda(async=True), volatile=True)
      qb_batch = autograd.Variable(torch.Tensor(qbs[_cursor : _end_batch]).cuda(async=True), volatile=True)
      if args.model == 'drnet':
        posdata_batch = autograd.Variable(torch.Tensor(poses[_cursor : _end_batch]).cuda(async=True), volatile=True)
        itr_pred_batch = model(qa_batch, qb_batch, im_batch, posdata_batch).data.cpu()
      else:
        ts_batch = autograd.Variable(torch.Tensor(t_s[_cursor : _end_batch]).cuda(async=True), volatile=True)
        to_batch = autograd.Variable(torch.Tensor(t_o[_cursor : _end_batch]).cuda(async=True), volatile=True)
        bboxs_batch = autograd.Variable(torch.Tensor(bbox_s[_cursor : _end_batch]).cuda(async=True), volatile=True)
        bboxo_batch = autograd.Variable(torch.Tensor(bbox_o[_cursor : _end_batch]).cuda(async=True), volatile=True)
        itr_pred_batch = model(qa_batch, qb_batch, im_batch, ts_batch, to_batch, bboxs_batch, bboxo_batch).data.cpu()

      if itr_pred is None:
        itr_pred = itr_pred_batch
      else:
        itr_pred = np.vstack((itr_pred, itr_pred_batch))
      _cursor = _end_batch

    for j in range(num_gts):
      sub = gt_bboxes[j, 0, :]
      obj = gt_bboxes[j, 1, :]
      for k in range(itr_pred.shape[1]):
        if itr_pred[j, k] < thresh:
          continue
        pred[i].append([itr_pred[j, k], 1, 1, gts[j, 0], k, gts[j, 2]])
        pred_bboxes[i].append([sub, obj])
    pred[i] = np.array(pred[i])
    pred_bboxes[i] = np.array(pred_bboxes[i])
  print('\nwriting file..')
  pickle.dump([pred, pred_bboxes], open(output_file, 'wb'))


