from collections import defaultdict
from torch import nn
import torch

from .adareg import ExpAverageSquare


class SoftAdaptiveRegularizer(nn.Module):
    def __init__(self, model, optimizer, backward, beta=0.99, eps=1e-15):
        super().__init__()
        self.exp_avg_sq = ExpAverageSquare(model, beta)

        self.model = model
        self.backward = backward
        self.optimizer = optimizer
        self.beta = beta
        self.eps = eps
        
    def forward(self, losses):
        error, reg = losses
        # We start by backwarding the regularization term
        self.backward(reg, self.optimizer, retain_graph=True)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                name = self.exp_avg_sq.convert_name(name)
                denom = getattr(self.exp_avg_sq, name).sqrt() + self.eps 
                p.grad.div_(denom)
        self.backward(error, self.optimizer)

        # We update the reg strength if gradient is small enough
        self.exp_avg_sq.update(self.model)

