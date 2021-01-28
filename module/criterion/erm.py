import torch
from torch import nn
from collections import namedtuple

class ERM(nn.Module):
  """
  Class implementing the classic Expected Risk Minimization loss
  """
  def __init__(self, error_cost=None):
    super().__init__()
    self.error_cost = error_cost if error_cost else nn.MSELoss()

  def forward(self, model, phase, batch, batch_idx):
    inp, target, env = batch
    if inp.size(0)==1:
      inp, target, env = inp.squeeze(0), target.squeeze(0), env.squeeze(0)

    pred = model(inp)

    error = self.error_cost(pred, target)

    return pred, error

  def reduce(self, losses):
    return losses.mean()
  
  def combine_to_scalar(self, losses):
    return losses.mean()