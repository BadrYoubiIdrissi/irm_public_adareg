from torch import nn
import pytorch_lightning as pl


"""
This module takes care of logging and calculating metrics along with any measure of interest.
It is for now the only quite messy part of the code. 
Since it is very dependent on what one wants to visualize I didn't spend too much time splitting this up into modules.
This will likely come in the future 
"""
class LinearIRMMetrics(nn.Module):
    def __init__(self, log_freq, max_epochs, metric_dict):
        super().__init__()
        self.log_freq = log_freq
        self.max_epochs = max_epochs
        self.metric_dict = metric_dict

    def should_not_log(self, current_epoch, phase):
        return phase=="train" and (current_epoch % self.log_freq[phase] != 0) and (current_epoch < self.max_epochs-1)

    def forward(self, module, phase, batch, batch_idx, out, losses):
        if self.should_not_log(module.current_epoch, phase): return
        _, y, _ = batch
        for metric_name, metric in self.metric_dict[phase].items():
            m_out = metric(out.squeeze(), y.squeeze())
            module.log(f"metrics/{phase}/{metric_name}",metric.compute()) 

        choices = module.hparams.choices
        if "irm" in choices['module/criterion']: 
            error, penalty = module.criterion.reduce(losses)
            module.log(f"losses/{phase}/erm", error.item())
            module.log(f"losses/{phase}/irm", penalty.item())
        elif "erm" in choices['module/criterion']: 
            loss = module.criterion.reduce(losses)
            module.log(f"losses/{phase}/erm", loss.item())
        
        if choices.get('module/model', "") == "linear" and phase=="train":
            if "adareg" == choices.get('module/gradient_manager', ""):
                for name, buffer in module.gradient_manager.reg_weights.named_buffers():
                    for i, r in enumerate(buffer.flatten().tolist()):
                        module.log(f"optim/reg_strength/{name}/{i}", r)
            if choices.get('module/gradient_manager', "") in ["adareg", "inverse_adareg"]:
                for name, buffer in module.gradient_manager.exp_avg_sq.named_buffers():
                    for i, r in enumerate(buffer.flatten().tolist()):
                        module.log(f"optim/exp_avg_sq/{name}/{i}", r)
            if choices.get('module/gradient_manager', "") in ["simple_adareg"]:
                module.log(f"optim/reg_strength", module.gradient_manager.reg_strength)

        if choices.get('module/model', "") == "nonlinear" and phase=="train":
            if "adareg" == choices.get('module/gradient_manager', ""):
                for name, buffer in module.gradient_manager.reg_weights.named_buffers():
                    module.log(f"optim/reg_strength/{name}", buffer.mean().item())
            if "adareg" in choices.get('module/gradient_manager', ""):
                for name, buffer in module.gradient_manager.exp_avg_sq.named_buffers():
                    module.log(f"optim/exp_avg_sq/{name}", buffer.mean().item())

    def on_after_backward(self, module):
        if self.should_not_log(module.current_epoch, "train"): return
        choices = module.hparams.choices
        if choices.get('module/model', "") == "linear":
            module.log_dict({
                    "params/phi_0_grad": module.model[0].weight.grad[0, 0].clone().item(),
                    "params/phi_0": module.model[0].weight[0, 0].clone().item(),
                    "params/phi_1_grad": module.model[0].weight.grad[0, 1].clone().item(),
                    "params/phi_1": module.model[0].weight[0, 1].clone().item()
                })
        for name, param in module.model.named_parameters():
            module.log(f"grads/{name}", param.grad.norm().item())