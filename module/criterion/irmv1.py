import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
from .irm_wrapper import IRMModelWrapper

from collections import namedtuple
RegularizedLoss = namedtuple('RegularizedLoss', 'error penalty')
# This class fetches the correct dummy weight with an embedding matrix
class IRMv1(nn.Module):
  """
  This module implements IRMv1 as presented in the original paper. Some modifications are made to accept arbitrary assignements to environments e.
  In the original paper's code, they loop over the environment at each batch. Here we store the environement in a seperate vector that comes in through the batch.
  """
  def __init__(self, nb_envs, split_position, reg_strength, err_strength=1.0, add_trivial=False, error_cost=None, penalty_scaling=None):
    super().__init__()
    self.error_cost = error_cost if error_cost else lambda x, y: (x-y)**2
    self.penalty_scaling = penalty_scaling if penalty_scaling else lambda x: x**2
    self.nb_envs = nb_envs
    self.split_position = split_position
    self.add_trivial = add_trivial
    self.reg_strength = reg_strength
    self.err_strength = err_strength

  def forward(self, model, phase, batch, batch_idx):
    """
    pred : The output of the representaiton. Shape (batch, *other_output)
    target : Targets. Shape (batch,)
    e : Tensor containing the ID of the environment of each sample. Shape (batch,)
    """

    inp, target, env = batch
    if inp.size(0)==1:
      inp, target, env = inp.squeeze(0), target.squeeze(0), env.squeeze(0)
    model = IRMModelWrapper(model, self.add_trivial)
    representation, classifier = model.split_and_get_classifier(self.split_position)

    with torch.set_grad_enabled(True):
      pred = model(inp)
      nb_samples_per_env = torch.bincount(env, minlength=self.nb_envs).refine_names('env')
      env = F.one_hot(env, self.nb_envs).refine_names('batch', 'env') #('batch','env')

      # We compute the error term for each sample
      error = self.error_cost(pred, target).squeeze().refine_names('batch') # ('batch')

      # We sum together the losses from the samples in the same environments 
      error_per_env = (error.align_to('batch', 'env')*env.align_to('batch', 'env')).sum('batch')/nb_samples_per_env.align_to('env')

      # We compute the gradient with respect to each env. ('env', ...)
      # This corresponds to the jacobian of error_per_env with respect to the classifier parameters
      # We use a loop here but future pytorch features will allow for a faster implementation using forward mode differentiation
      # See https://github.com/pytorch/rfcs/pull/11
      penalty = torch.zeros(self.nb_envs, device=error_per_env.device).refine_names('env')
      for e in range(self.nb_envs):
        if nb_samples_per_env[e] > 0:
          param_grads = grad(error_per_env[e], classifier.parameters(), create_graph = True)
          for g in param_grads:
            # We apply the penalty loss
            penalty[e] += self.penalty_scaling(g.norm())

    return pred, RegularizedLoss(error, penalty)

  def reduce(self, losses):
    error, penalty = losses
    error = error.mean('batch')
    penalty = penalty.mean('env')
    return RegularizedLoss(error, penalty)
  
  def combine_to_scalar(self, losses):
    error, penalty = self.reduce(losses)
    return (self.err_strength*error + self.reg_strength*penalty).rename(None)