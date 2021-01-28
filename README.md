## Improvements on IRM

This repo contains some ideas I've tried to implement to improve on IRM.

The code is very modular to encourage experimentation. I use [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/latest/) (a pytorch framework) to organize the code into modules. Then I use [hydra](https://hydra.cc/) (a configuration manager) to choose which modules to use and to configure them properly. This way, all parameters can be overriden and finetuned automatically. I also keep track of the (large) configuration object to be able to compare runs.

The structure of the repo is the following 

```
irm
├── config
│   ├── callbacks
│   ├── datamodule
│   ├── experiment
│   ├── hydra
│   │   └── launcher
│   └── module
│       ├── criterion
│       ├── gradient_manager
│       ├── metrics
│       ├── model
│       └── optimizer
├── datamodule
├── module
│   ├── criterion
│   ├── gradient_manager
│   ├── metrics
│   └── model
├── notes
└── visualization
```

Where the config directory contains the various options available for each submodule of the code. For example the module config has the following options

```
irm/config/module
├── criterion
│   ├── erm.yaml
│   └── irmv1.yaml
├── gradient_manager
│   ├── adareg.yaml
│   └── soft_adareg.yaml
├── metrics
│   └── linear_irm.yaml
├── model
│   ├── linear.yaml
│   └── nonlinear.yaml
├── optimizer
│   ├── adam.yaml
│   └── sgd.yaml
├── base.yaml
└── manual.yaml
```

Where one can choose for example the irmv1 criterion along with the adareg gradient manager, a linear module and the adam optimizer. The config directly contains the classes that need to be instantiated along with any configuration parameters. So the code combines these different building blocks. 

Here is a commandline example: 

```
python train.py -m hydra/launcher=submitit_slurm   
                   module=manual 
                   module/model=linear
                   module/criterion=irmv1 
                   module/optimizer=adam
                   module/gradient_manager=adareg 
                   job.seed=range\(100,108\)
```

This launches 8 experiments in parallel on a slurm cluster with seeds from 100 to 107. Where the module is the encapsulating component that builds the others and makes them interact. The manual module allows for manual optimization (manually backwarding and stepping).  

The modules' codes contain comments with more information on how they work.

To reproduce the report's results, create a conda environment using the provided environment file 

```
conda env create -f environment.yml
conda activate torch
```

Then create a [Weights and Biases](https://wandb.ai/site) account to easily visualize the results. Login using your api key

```
wandb login <API-KEY>
```

Once you do this the results of all the runs will be logged in a project named "irm".

If you have access to a slurm cluster, add `hydra/launcher=ruche` to the command line arguments of the following commands. This will launch 8 jobs in parallel until it finishes all jobs. Launching this on a local computer might take up to 4 hours since the jobs will be launched sequentially. Look into the hydra docs to launch this using the ray launcher if you want it to go faster. 

```
python train.py -m job.seed=range\(100,132\) +experiment/regression=base_erm 

python train.py -m job.seed=range\(100,132\) +experiment/regression=base_erm_oracle 

python train.py -m job.seed=range\(100,132\) +experiment/regression=irmv1 

python train.py -m job.seed=range\(100,132\) +experiment/regression=adareg_simple 

python train.py -m job.seed=range\(100,132\) +experiment/regression=adareg 
```