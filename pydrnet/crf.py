import torch
import torch.nn as nn
import torch.nn.functional as F


class CRF(nn.Module):

  def __init__(self):
    super().__init__()
    self.similarities = nn.Parameter(torch.Tensor(70, 70))
    nn.init.xavier_uniform(self.similarities)


  def forward(self, logits):
    L = (self.similarities + self.similarities.t()) / 2.0
    Q = F.log_softmax(logits, 1)
    for n_iter in range(10):
      tmp = logits + torch.matmul(L, (2 * Q.exp().t() - 1)).t() - L.diag() * (2 * Q.exp() - 1) 
      Q = F.log_softmax(torch.stack([tmp, -tmp], 2), 2)[:, :, 0]
    return Q

