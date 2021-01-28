from torch import nn

"""
Simple linear model
"""
class LinearModel(nn.Sequential):
  def __init__(self, input_dim, output_dim, bias):
    super().__init__(
      nn.Linear(input_dim, output_dim, bias)
    )
