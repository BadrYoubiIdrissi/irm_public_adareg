import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from hydra.utils import instantiate

class BaseModule(pl.LightningModule):
    """
    This is the base module on which we will build upon and configure. 
    It encapsulates the following : 
        - Model : nn.Module : contains the model architecture and inner logic
        - Criterion : nn.Module : responsible for calculating the loss 
        - Optimizer : torch.optim.Optimizer 
        - Metrics : nn.Module : calculates the different metrics and logs them

    This object is initialized with preconfigured partial functions that return the corresponding object.
    So for example if we want to instantiate an Adam optimizer we would need the model parameters and some hyperparameters.
        torch.optim.Adam(model.parameters(), lr=0.01, momentum=0.99)
    What the function in here will achieve is to have already passed values that are in the configuration.
    For example optimizer = lambda p : torch.optim.Adam(p, lr=0.01, momentum=0.99). 
    So that if it is called with the model's parameters (that we can only have access to here) it is already preconfigured.
    This is done to alleviate this class and remove the burden of configuration from it.
    
    """

    def __init__(self, hparams, model=None, criterion=None, optimizer=None, metrics=None):
        super().__init__()
        
        # Initilization should be done in this order since the optimizer depends on the model
        self.hparams = hparams
        self.build_model(model)
        self.build_criterion(criterion)
        self.build_optimizer(optimizer)
        self.build_metrics(metrics)

    # The build functions here are simple but can be overriden to 
    # allow for more interaction between the submodules.

    def build_model(self, model):
        self.model = model()

    def build_criterion(self, criterion):
        self.criterion = criterion()
    
    def build_optimizer(self, optimizer):
        self.optimizer = optimizer(params=self.model.parameters())
    
    def build_metrics(self, metrics):
        self.metrics = metrics()

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, inputs):
        return self.model(x)

    def step(self, phase, batch, batch_idx):
        out, losses = self.criterion(model=self.model, 
                                     phase=phase, 
                                     batch=batch, 
                                     batch_idx=batch_idx)
        self.metrics(module=self, 
                     phase=phase, 
                     batch=batch, 
                     batch_idx=batch_idx, 
                     out=out, 
                     losses=losses)
        return losses

    def on_after_backward(self):
        self.metrics.on_after_backward(self)

    def training_step(self, batch, batch_idx):
        losses = self.step("train", batch, batch_idx)
        return self.criterion.combine_to_scalar(losses)
    
    def validation_step(self, batch, batch_idx):
        self.step("valid", batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        self.step("test", batch, batch_idx)