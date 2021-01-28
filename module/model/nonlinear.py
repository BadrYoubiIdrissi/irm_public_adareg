from torch import nn

"""
Simple nonlinear configurable model
"""

class SimpleNonLinearModel(nn.Sequential):
  def __init__(self, input_dim, nb_layers, nb_hidden_neur, output_dim):
    super().__init__()
    self.append_block(nn.Linear(input_dim, nb_hidden_neur))
    
    for i in range(nb_layers-1):
      self.append_block(nn.Sequential(nn.ReLU(),nn.Linear(nb_hidden_neur, nb_hidden_neur)))

    self.append_block(nn.Sequential(nn.ReLU(),nn.Linear(nb_hidden_neur, output_dim)))

  def append_block(self, module):
    self.add_module(str(len(self)), module)
    