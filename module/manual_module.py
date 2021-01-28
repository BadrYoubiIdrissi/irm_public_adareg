from .base_module import BaseModule

"""
This module allows for manual optimization. 
This is needed for adaptive regularization.
"""
class ManualModule(BaseModule):

    def __init__(self, *args, gradient_manager=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.build_gradient_manager(gradient_manager)

    def build_gradient_manager(self, gradient_manager):
        self.gradient_manager = gradient_manager(model=self.model, optimizer=self.optimizer, backward=self.manual_backward)

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()
        losses = self.step("train", batch, batch_idx)
        self.gradient_manager(self.criterion.reduce(losses))
        self.on_after_backward()
        self.optimizer.step()

    @property
    def automatic_optimization(self) -> bool:
        return False