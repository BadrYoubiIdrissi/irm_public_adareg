from collections import defaultdict
from torch import nn
import torch
from .adareg import AdaptiveRegularizer

class InverseAdaptiveRegularizer(AdaptiveRegularizer):
        
    def forward(self, losses):
        error, reg = losses
        # We start by backwarding the regularization term
        
        self.backward(error, self.optimizer, retain_graph=True)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                name = self.reg_weights.convert_name(name)
                # This is equivalent to optimizing for error/reg_weight + penalty 
                p.grad *= getattr(self.reg_weights, name)
        self.backward(reg, self.optimizer)
        # We update the reg strength if gradient is small enough
        self.exp_avg_sq.update(self.model)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                name = self.reg_weights.convert_name(name)
                exp_avg_sq = getattr(self.exp_avg_sq, name)
                mask = (exp_avg_sq < self.reg_threshold)
                getattr(self.reg_weights, name)[mask] /= self.reg_multiplier

