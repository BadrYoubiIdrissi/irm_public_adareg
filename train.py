import hydra
from omegaconf import OmegaConf
import configuration

@hydra.main(config_path='config', config_name='config')
def app(cfg):
    
    # We delay heavy imports for hydra to be fast
    import pytorch_lightning as pl
    
    # We instantiate the needed objects directly from the config
    # This allows us to change bits and pieces of the code easily and without 
    # having to manually go back and forth.
    # It also allows for better reproducibility as all the hyperparameters are saved in the config
    # There are no constants fixed in the code itself
    
    # What we refer to as module is a pytorch lightening module that encapsulates everything
    # except the data. So the model, the optimizer, the loss (criterion) and metrics 

    configuration.set_job_name(cfg, cfg.important_choices)
    pl.trainer.seed_everything(cfg.job.seed)

    module = hydra.utils.instantiate(cfg.module, cfg)

    # The data module is responsible for preparing the data and splitting it.
    # It is decoupled from the module above to allow for swapping datasets easily
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.save_hparams(cfg)
    # This is a logger object that takes care of logging the metrics etc
    logger = configuration.configure_logger(cfg)

    callbacks = list(configuration.instantiate_dict(cfg.callbacks).values()) if hasattr(cfg, "callbacks") else None

    # This is a pytorch lightening trainer that takes care of the training loop and the engineering
    # side of the pipeline (devices, distributed training, quantization ...)    
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    if hasattr(cfg, "logger") and cfg.activate.watch_model_grads: 
        logger.experiment.watch(module.model)
    # We fit the model to the data 
    trainer.fit(module, datamodule)

    # We test the model
    if cfg.activate.test:
        trainer.test(datamodule=datamodule)
    
    if hasattr(cfg, "logger"): 
        logger.experiment.finish()

if __name__ == "__main__":
    app()