import pytorch_lightning as pl
import torch
from omegaconf import open_dict
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np
import torch
import math

class BaseLinearDataset(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, train_size_per_env, val_size_per_env, test_size_per_env, scramble, dim_inv, dim_in, dim_out, is_oracle, n_envs):
        super().__init__()
        self.batch_size = batch_size
        self.single_batch = batch_size >= n_envs*train_size_per_env
        self.num_workers = num_workers
        self.train_size_per_env = train_size_per_env
        self.val_size_per_env = val_size_per_env
        self.test_size_per_env = test_size_per_env
        self.dim_in = dim_in
        self.dim_inv = dim_inv
        self.dim_spu = dim_in - dim_inv
        self.is_oracle = is_oracle
        self.n_envs = n_envs
        if scramble:
            self.scramble, _ = torch.qr(torch.randn(self.dim_in, self.dim_in))
        else:
            self.scramble = self.scramble = torch.eye(self.dim_in)

    def setup(self, stage=None):
        self.train = self.generate_data(self.train_size_per_env, split="test" if self.is_oracle else "train")
        self.val = self.generate_data(self.val_size_per_env, split="test")
        self.test = self.generate_data(self.test_size_per_env, split="test")

    def train_dataloader(self):
        if self.single_batch:
            return DataLoader([self.train])
        return DataLoader(TensorDataset(self.train), batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.single_batch:
            return DataLoader([self.val])
        return DataLoader(TensorDataset(self.val), batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.single_batch:
            return DataLoader([self.test])
        return DataLoader(TensorDataset(self.test), batch_size=self.batch_size, num_workers=self.num_workers)

"""
Most of the code below comes from https://github.com/facebookresearch/InvarianceUnitTests.
Some changes were made to integrate with the rest of the code
"""

class RegressionSimple(BaseLinearDataset):
    """
    Cause and effect of a target with heteroskedastic noise
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.task = "regression"
        self.envs = []

        if self.n_envs >= 2:
            self.envs = [0.1, 1.5]
        if self.n_envs >= 3:
            self.envs.append(2)
        if self.n_envs > 3:
            for env in range(3, self.n_envs):
                var = 10 ** torch.zeros(1).uniform_(-2, 1).item()
                self.envs.append(var)
        print("Environments variables:", self.envs)
    
    def save_hparams(self, cfg):
        with open_dict(cfg):
            cfg.data = {
                'envs': self.envs,
                'wxy': 1,
                'wyz': 1
            }

    def generate_data(self, size=1000, split="train"):
        all_inputs = []
        all_outputs = []
        envs = []
        n = size
        for env, sdv in enumerate(self.envs):
            x = torch.randn(n, self.dim_inv) * sdv
            y = x + torch.randn(n, self.dim_inv) * sdv
            z = y + torch.randn(n, self.dim_spu)

            if split == "test":
                z = z[torch.randperm(len(z))]

            inputs = torch.cat((x, z), -1) @ self.scramble
            outputs = y.sum(1, keepdim=True)
            all_inputs.append(inputs)
            all_outputs.append(outputs)
            envs.append(env*torch.ones_like(outputs).long())

        inputs = torch.cat(all_inputs)
        outputs = torch.cat(all_outputs)
        envs = torch.cat(envs)

        return inputs, outputs, envs.squeeze()

class Regression(BaseLinearDataset):
    """
    Cause and effect of a target with heteroskedastic noise
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.task = "regression"
        self.envs = []

        if self.n_envs >= 2:
            self.envs = [0.1, 1.5]
        if self.n_envs >= 3:
            self.envs.append(2)
        if self.n_envs > 3:
            for env in range(3, self.n_envs):
                var = 10 ** torch.zeros(1).uniform_(-2, 1).item()
                self.envs.append(var)
        print("Environments variables:", self.envs)

        self.wxy = torch.randn(self.dim_inv, self.dim_inv) / self.dim_inv
        self.wyz = torch.randn(self.dim_inv, self.dim_spu) / self.dim_spu

    def save_hparams(self, cfg):
        with open_dict(cfg):
            cfg.data = {
                'envs': self.envs,
                'wxy': self.wxy.tolist(),
                'wyz': self.wyz.tolist()
            }

    def generate_data(self, size=1000, split="train"):
        all_inputs = []
        all_outputs = []
        envs = []
        n = size
        for env, sdv in enumerate(self.envs):
            x = torch.randn(n, self.dim_inv) * sdv
            y = x @ self.wxy + torch.randn(n, self.dim_inv) * sdv
            z = y @ self.wyz + torch.randn(n, self.dim_spu)

            if split == "test":
                z = z[torch.randperm(len(z))]

            inputs = torch.cat((x, z), -1) @ self.scramble
            outputs = y.sum(1, keepdim=True)
            all_inputs.append(inputs)
            all_outputs.append(outputs)
            envs.append(env*torch.ones_like(outputs).long())

        inputs = torch.cat(all_inputs)
        outputs = torch.cat(all_outputs)
        envs = torch.cat(envs)

        return inputs, outputs, envs.squeeze()


class CammelCow(BaseLinearDataset):
    """
    Cows and camels
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.envs = []

        if self.n_envs >= 2:
            self.envs = [
                {"p": 0.95, "s": 0.3},
                {"p": 0.97, "s": 0.5}
            ]
        if self.n_envs >= 3:
            self.envs.append({"p": 0.99, "s": 0.7})
        if self.n_envs > 3:
            for env in range(3, self.n_envs):
                self.envs.append({
                    "p": torch.zeros(1).uniform_(0.9, 1).item(),
                    "s": torch.zeros(1).uniform_(0.3, 0.7).item()
                })
        print("Environments variables:", self.envs)

        # foreground is 100x noisier than background
        self.snr_fg = 1e-2
        self.snr_bg = 1

        # foreground (fg) denotes animal (cow / camel)
        cow = torch.ones(1, self.dim_inv)
        self.avg_fg = torch.cat((cow, cow, -cow, -cow))

        # background (bg) denotes context (grass / sand)
        grass = torch.ones(1, self.dim_spu)
        self.avg_bg = torch.cat((grass, -grass, -grass, grass))
    
    def save_hparams(self, cfg):
        with open_dict(cfg):
            cfg.data = {
                'envs': self.envs,
                'snr_fg': self.snr_fg,
                'snr_bg': self.snr_bg
            }

    def generate_data(self, size=1000, split="train"):
        all_inputs = []
        all_outputs = []
        envs = []
        n = size
        for env in range(self.n_envs):
            p = self.envs[env]["p"]
            s = self.envs[env]["s"]
            w = torch.Tensor([p, 1 - p] * 2) * torch.Tensor([s] * 2 + [1 - s] * 2)
            i = torch.multinomial(w, n, True)
            x = torch.cat((
                (torch.randn(n, self.dim_inv) /
                    math.sqrt(10) + self.avg_fg[i]) * self.snr_fg,
                (torch.randn(n, self.dim_spu) /
                    math.sqrt(10) + self.avg_bg[i]) * self.snr_bg), -1)

            if split == "test":
                x[:, self.dim_spu:] = x[torch.randperm(len(x)), self.dim_spu:]

            inputs = x @ self.scramble
            outputs = x[:, :self.dim_inv].sum(1, keepdim=True).gt(0).float()
            
            all_inputs.append(inputs)
            all_outputs.append(outputs)
            envs.append(env*torch.ones_like(outputs).long())

        inputs = torch.cat(all_inputs)
        outputs = torch.cat(all_outputs)
        envs = torch.cat(envs)

        return inputs, outputs, envs.squeeze()


class SmallInvMargin(BaseLinearDataset):
    """
    Small invariant margin versus large spurious margin
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for env in range(self.n_envs):
            self.envs.append(torch.randn(1, dim_spu))
    
    def save_hparams(self, cfg):
        with open_dict(cfg):
            cfg.data = {
                'envs': self.envs,
            }

    def generate_data(self, n=1000, split="train"):
        all_inputs = []
        all_outputs = []
        envs = []
        n = size
        for env in range(self.n_envs):
            m = n // 2
            sep = .1

            invariant_0 = torch.randn(m, self.dim_inv) * .1 + \
                torch.Tensor([[sep] * self.dim_inv])
            invariant_1 = torch.randn(m, self.dim_inv) * .1 - \
                torch.Tensor([[sep] * self.dim_inv])

            shortcuts_0 = torch.randn(m, self.dim_spu) * .1 + self.envs[env]
            shortcuts_1 = torch.randn(m, self.dim_spu) * .1 - self.envs[env]

            x = torch.cat((torch.cat((invariant_0, shortcuts_0), -1),
                        torch.cat((invariant_1, shortcuts_1), -1)))

            if split == "test":
                x[:, self.dim_inv:] = x[torch.randperm(len(x)), self.dim_inv:]

            inputs = x @ self.scramble
            outputs = torch.cat((torch.zeros(m, 1), torch.ones(m, 1)))

            all_inputs.append(inputs)
            all_outputs.append(outputs)
            envs.append(env*torch.ones_like(outputs).long())

        inputs = torch.cat(all_inputs)
        outputs = torch.cat(all_outputs)
        envs = torch.cat(envs)

        return inputs, outputs, envs.squeeze()