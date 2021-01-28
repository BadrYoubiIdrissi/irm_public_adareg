from collections import defaultdict
from torch import nn
import torch

class SimpleAdaptiveRegularizer(nn.Module):
    def __init__(self, model, optimizer, backward, reg_multiplier=10, threshold=0.01, beta=0.99, init_reg_strength=1):
        super().__init__()
        self.reg_threshold = threshold
        self.reg_multiplier = reg_multiplier
        self.reg_strength = init_reg_strength

        self.model = model
        self.backward = backward
        self.optimizer = optimizer
        self.beta = beta
        
    def forward(self, losses):
        error, reg = losses
        # We start by backwarding the regularization term
        self.backward(reg, self.optimizer, retain_graph=True)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                # This is equivalent to optimizing for error + reg_weight*penalty 
                p.grad *= self.reg_strength
        self.backward(error, self.optimizer)

        # We update the reg strength if gradient is small enough
        total_grad = 0
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                total_grad += p.grad.norm()**2
        if total_grad < self.reg_threshold:
            self.reg_strength *= self.reg_multiplier

