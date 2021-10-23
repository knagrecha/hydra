# INSTALLATION GUIDE

This guide will walk you through the process of installing and using Hydra for the first time.

## Prerequisites
---

You must have the version of PyTorch installed that matches your system. You can download PyTorch here: https://pytorch.org/. The examples in this guide also use torchtext, but it is not necessary to use the framework.

## Download
---

At the moment, Hydra is only available through this repository. Run `git clone https://github.com/knagrecha/mptorch`, then cd into the repo. 

Run `pip install -r requirements.txt`, then `pip install .`.

## Running
---

The files `examples/single-task-lm.py` and `examples/multi-task-lm.py` demonstrate how to setup a simple training job. Essentially, you need to define your dataloaders and model through the standard PyTorch APIs, then pass the objects into Hydra's wrappers as follows:

    model_hydra_0 = Model(model_0) # Call Hydra Model Wrapper
    dataloader_0 = get_data_loader_train(32) # Generate dataloader
    
    model_hydra_0 = Model(model_1)
    dataloader_1 = get_data_loader_train(32)
    
    task_0 = ModelTask(model_0, loss_fn, dataloader_0, lr_0, epochs_0)
    task_1 = ModelTask(model_1, loss_fn, dataloader_1, lr_1, epochs_1)


    # create orchestrator
    orchestra = ModelOrchestrator([task_0, task_1])
    orchestra.verbose = 1
    orchestra.buffer = None

    orchestra.generate()
    orchestra.train_models()


