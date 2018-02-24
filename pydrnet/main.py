import torch
import torch.nn as nn
import torch.autograd as autograd
from dataloader import dataloader
from drnet import DRNet
from vtranse import VtransE
from baseline import V
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
import argparse
from progressbar import ProgressBar
import pickle
import os
import os.path
import shutil
from datetime import datetime
from predict import predict


def num_true_positives(logits, labels):
  return (torch.max(logits, 1)[1].view(labels.size()).data == labels.data).sum()


def train(model, criterion, loader, epoch, summary_writer, args):
  model.train()
  loss = 0
  acc = 0
  num_samples = 0
  bar = ProgressBar(max_value=len(loader))
  for idx, data_batch in enumerate(loader):
    qa_batch_var = autograd.Variable(data_batch['qa'].cuda(async=True), requires_grad=False)
    qb_batch_var = autograd.Variable(data_batch['qb'].cuda(async=True), requires_grad=False)
    label_batch_var = autograd.Variable(data_batch['label'].cuda(async=True), requires_grad=False)
    if args.model == 'drnet':
      im_batch_var = autograd.Variable(data_batch['im'].cuda(async=True), requires_grad=False)
      posdata_batch_var = autograd.Variable(data_batch['posdata'].cuda(async=True), requires_grad=False)
      output_batch_var = model(qa_batch_var, qb_batch_var, im_batch_var, posdata_batch_var)
    elif args.model == 'vtranse':
      im_batch_var = autograd.Variable(data_batch['full_im'].cuda(async=True), requires_grad=False)
      ts_batch_var = autograd.Variable(data_batch['t_s'].cuda(async=True), requires_grad=False)
      to_batch_var = autograd.Variable(data_batch['t_o'].cuda(async=True), requires_grad=False)
      bboxs_batch_var = autograd.Variable(data_batch['bbox_s'].cuda(async=True), requires_grad=False)
      bboxo_batch_var = autograd.Variable(data_batch['bbox_o'].cuda(async=True), requires_grad=False)
      output_batch_var = model(qa_batch_var, qb_batch_var, im_batch_var, ts_batch_var, to_batch_var, bboxs_batch_var, bboxo_batch_var)
    else:
      im_batch_var = autograd.Variable(data_batch['im'].cuda(async=True), requires_grad=False)
      output_batch_var = model(im_batch_var)

    loss_batch_var = criterion(output_batch_var, label_batch_var)

    loss_batch = loss_batch_var.data[0]
    loss += (torch.numel(data_batch['label']) * loss_batch)
    acc += num_true_positives(output_batch_var, label_batch_var)
    num_samples += torch.numel(data_batch['label'])

    optimizer.zero_grad()
    loss_batch_var.backward()
    optimizer.step()

    summary_writer.add_scalar('train/loss', loss_batch, epoch * len(loader) + idx)
    bar.update(idx)

  loss /= num_samples
  acc /= (num_samples / 100.)

  summary_writer.add_scalar('train/accuracy', loss, (epoch + 1) * len(loader))

  return loss, acc


def test(model, criterion, loader, epoch, summary_writer, args):
  model.eval()
  loss = 0
  acc = 0
  num_samples = 0
  bar = ProgressBar(max_value=len(loader))
  for idx, data_batch in enumerate(loader):
    qa_batch_var = autograd.Variable(data_batch['qa'].cuda(async=True), volatile=True)
    qb_batch_var = autograd.Variable(data_batch['qb'].cuda(async=True), volatile=True)
    label_batch_var = autograd.Variable(data_batch['label'].cuda(async=True), volatile=True)
    if args.model == 'drnet':
      im_batch_var = autograd.Variable(data_batch['im'].cuda(async=True), volatile=True)
      posdata_batch_var = autograd.Variable(data_batch['posdata'].cuda(async=True), volatile=True)
      output_batch_var = model(qa_batch_var, qb_batch_var, im_batch_var, posdata_batch_var)
    elif args.model == 'vtranse':
      im_batch_var = autograd.Variable(data_batch['full_im'].cuda(async=True), volatile=True)
      ts_batch_var = autograd.Variable(data_batch['t_s'].cuda(async=True), volatile=True)
      to_batch_var = autograd.Variable(data_batch['t_o'].cuda(async=True), volatile=True)
      bboxs_batch_var = autograd.Variable(data_batch['bbox_s'].cuda(async=True), volatile=True)
      bboxo_batch_var = autograd.Variable(data_batch['bbox_o'].cuda(async=True), volatile=True)
      output_batch_var = model(qa_batch_var, qb_batch_var, im_batch_var, ts_batch_var, to_batch_var, bboxs_batch_var, bboxo_batch_var)
    else:
      im_batch_var = autograd.Variable(data_batch['im'].cuda(async=True), volatile=True)
      output_batch_var = model(im_batch_var)

    loss_batch_var = criterion(output_batch_var, label_batch_var)

    loss_batch = loss_batch_var.data[0]
    loss += (torch.numel(data_batch['label']) * loss_batch)
    acc += num_true_positives(output_batch_var, label_batch_var)
    num_samples += torch.numel(data_batch['label'])

    bar.update(idx)

  loss /= num_samples
  acc /= (num_samples / 100.)

  num_iters = (epoch + 1) * len(loader)
  summary_writer.add_scalar('test/loss', loss, num_iters)
  summary_writer.add_scalar('test/accuracy', acc, num_iters)

  return loss, acc


parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', type=str)
parser.add_argument('--log_dir', type=str, default=os.path.join('./runs', str(datetime.now())[:-7]))
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--model', type=str, default='drnet', choices=['drnet', 'vtranse', 'baseline'])
args = parser.parse_args()
if args.exp_id != None:
  args.log_dir = os.path.join('./runs', args.exp_id)
if not os.path.exists(args.log_dir):
  os.makedirs(args.log_dir)
  os.makedirs(os.path.join(args.log_dir, 'predictions'))
  os.makedirs(os.path.join(args.log_dir, 'checkpoints'))
print(args)

loader_train = dataloader('train', args.batchsize)
loader_test = dataloader('test', args.batchsize)
print('%d batches of training examples' % len(loader_train))
print('%d batches of testing examples' % len(loader_test))

if args.model == 'drnet':
  model = DRNet()
elif args.model == 'vtranse':
  model = VtransE()
else:
  model = V()
model.cuda()
criterion = nn.CrossEntropyLoss()
criterion.cuda()

optimizer = torch.optim.RMSprop(model.parameters(), 
                                lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
scheduler = ReduceLROnPlateau(optimizer, patience=3, verbose=True)

wt = SummaryWriter(args.log_dir)
best_acc = -1.

for epoch in range(args.num_epochs):
  print('EPOCH #%d' % epoch)

  print('training..')
  loss, acc = train(model, criterion, loader_train, epoch, wt, args)
  print('\nloss = %f, accuracy = %.02f' % (loss, acc)) 

  print('testing..')
  loss, acc = test(model, criterion, loader_test, epoch, wt, args)
  print('\nloss = %f, accuracy = %.02f' % (loss, acc))  

  pred_file = os.path.join(args.log_dir, 'predictions/pred_%02d.pickle') % epoch
  predict(model, pred_file, args)
  os.system("python ../tools/eval_triplet_recall.py --det_file '%s'" % pred_file)

  checkpoint_filename = os.path.join(args.log_dir, 'checkpoints/model_%02d.pth' % epoch)
  model.cpu()
  torch.save({'epoch': epoch + 1,
              'args': args,
              'state_dict': model.state_dict(),
              'accuracy': acc,
              'optimizer' : optimizer.state_dict(),
             }, checkpoint_filename)
  model.cuda()
  if best_acc < acc:
    best_acc = acc
    shutil.copyfile(checkpoint_filename, os.path.join(args.log_dir, 'checkpoints/model_best.pth'))
    shutil.copyfile(os.path.join(args.log_dir, 'predictions/pred_%02d.pickle' % epoch),
                    os.path.join(args.log_dir, 'predictions/pred_best.pickle'))

  scheduler.step(loss)

wt.close()
