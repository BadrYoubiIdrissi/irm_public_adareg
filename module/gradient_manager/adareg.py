from collections import defaultdict
from torch import nn
import torch

def reset_optimizer_state(optimizer):
    optimizer.state = defaultdict(dict)
    
class RegularizationWeights(nn.Module):
    def __init__(self, model, init_reg_strength):
        super().__init__()
        for name, p in model.named_parameters():
            self.register_buffer(self.convert_name(name), init_reg_strength*torch.ones_like(p))
    
    def convert_name(self, name):
        return name.replace('.','_')

class ExpAverageSquare(nn.Module):
    def __init__(self, model, beta):
        super().__init__()
        self.beta = beta
        for name, p in model.named_parameters():
            self.register_buffer(self.convert_name(name), torch.zeros_like(p))
            
    def update(self, model):
        for name, p in model.named_parameters():
            name = self.convert_name(name)
            getattr(self, name).mul_(self.beta).addcmul_(p.grad, p.grad, value=1 - self.beta)

    def convert_name(self, name):
        return name.replace('.','_')

class AdaptiveRegularizer(nn.Module):
    def __init__(self, model, optimizer, backward, reg_multiplier=10, threshold=0.01, beta=0.99, init_reg_strength=1):
        super().__init__()
        self.reg_threshold = threshold
        self.reg_multiplier = reg_multiplier
        self.reg_weights = RegularizationWeights(model, init_reg_strength)
        self.exp_avg_sq = ExpAverageSquare(model, beta)

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
                name = self.reg_weights.convert_name(name)
                # This is equivalent to optimizing for error + reg_weight*penalty 
                p.grad *= getattr(self.reg_weights, name)
        self.backward(error, self.optimizer)

        # We update the reg strength if gradient is small enough
        self.exp_avg_sq.update(self.model)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                name = self.reg_weights.convert_name(name)
                exp_avg_sq = getattr(self.exp_avg_sq, name)
                mask = (exp_avg_sq < self.reg_threshold)
                getattr(self.reg_weights, name)[mask] *= self.reg_multiplier

