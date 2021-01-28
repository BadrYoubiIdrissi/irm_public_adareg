from hydra._internal.utils import _locate
from hydra.core.hydra_config import HydraConfig
from functools import partial
from omegaconf import OmegaConf
import hydra
import random

def get_partial(*args, **kwargs):
    func_or_obj = _locate(kwargs.pop("_partial_"))
    return partial(func_or_obj, *args, **kwargs)

def set_job_name(cfg, choices_list):
    hydra_cfg = HydraConfig.get()
    cfg.job.name = '_'.join([hydra_cfg.choices[group] for group in choices_list if group in hydra_cfg.choices])

def instantiate_dict(cfg):
    return {callback_name : hydra.utils.instantiate(callback_cfg) for callback_name, callback_cfg in cfg.items()}
        
def configure_logger(cfg):
    if hasattr(cfg, "logger"): 
        logger = hydra.utils.instantiate(cfg.logger) 
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    else:
        logger = None
    return logger

OmegaConf.register_resolver('random_seed', lambda : random.randint(0, 10000))
