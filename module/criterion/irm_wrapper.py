import torch
from torch import nn
from collections import OrderedDict

class TrivialClassifier(nn.Module):
    def __init__(self):
      super().__init__()
      self.dummy = nn.Parameter(torch.ones(1))
    def forward(self, x):
      return self.dummy*x

class IRMModelWrapper(nn.Sequential):
  """
  This wrapper takes in a model and can dynamically split it into a representation $phi$ and classifier $w$.
  The model should be a nn.Sequential model with each submodule being the building blocks between which it is acceptable to split.
  To control the behavior of this wrapper one should create building blocks (with nn.Sequential for example) in the following way
  model = nn.Sequential(
    torch.nn.Sequential(
      torch.nn.Linear(dataset.dim, 64),
      torch.nn.ReLU(),
      torch.nn.Linear(64, 64),
    ),
    torch.nn.Sequential(
      torch.nn.ReLU(),
      torch.nn.Linear(64, 64),
    ),
    torch.nn.Sequential(
      torch.nn.ReLU(),
      torch.nn.Linear(64, num_features),
    )
  ) 
  This model therefore has a first 2 layer block that we can't split then a succession of 3 single layers. 
  Therefore the IRMWrapper can take any the first n blocks in the root sequential model for the representation  and N-n  remaining blocks for the classifier.
  For multi head models one could pass each branch 
       ____             ____
      |                | 
  ----         =   ----        +   ----
      \____                            \____
  
  If the model passed isn't sequential it will be converted to a single block sequential model and thus can't be split. 
  A trivial non trainable classifier nn.Linear(1,1) will be added on top.
  """
  def __init__(self, model, add_trivial=False):
    if not isinstance(model, nn.Sequential) or len(model) == 1 or add_trivial:
      super().__init__(model, TrivialClassifier().to(next(model.parameters()).device))
    else:
      super().__init__(*model)
  
  def __getitem__(self, idx):
    if isinstance(idx, slice):
        return nn.Sequential(OrderedDict(list(self._modules.items())[idx]))
    else:
        return self._get_item_by_idx(self._modules.values(), idx)

  def split_and_get_classifier(self, block_idx):
    assert block_idx != 0, "The representation can't be empty"
    assert block_idx != -len(self), "The representation can't be empty"
    assert block_idx != len(self), "The classifier can't be empty"
    self[block_idx].register_forward_pre_hook(self.on_classifier_forward)
    return self[:block_idx], self[block_idx:]

  def on_classifier_forward(self, module, classifier_inp):
    self.current_features = classifier_inp[0]