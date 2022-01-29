# Hydra

Hydra is a model parallel execution engine optimized for multi-task execution. It enables arbitrarily large models to be trained on a single GPU by supporting sequential data spilling, and generates near-optimal training schedules for sharded multi-model training using a novel scheduling strategy.

Contact knagrech@ucsd.edu for more info.

## Install

To install Hydra, follow the [Installation Guide](https://github.com/knagrecha/hydra/blob/main/INSTALL.md).

## (For VLDB)
The files to run the end-to-end tests are twelve_model_task.py and twelve_model_task_vit.py. 

Please note that running 12-task 8-GPU single node experiments (like the paper) is an expensive operation that demands a great deal of DRAM and continuous, heavy GPU utilization (see debugging section for more details). If you want to run a smaller scale version (2-3 GPUs) just to observe the system, I have also prepared a file with a relatively lighter workload, three-task-lm.py.


## Debugging

Expensive multi-task operations can be very demanding on the hardware, and GPU failure can occur. If you notice in the training output that one task or another seems to have frozen in place (minibatch counter is not advancing) when you expect it to, it is likely that the GPU it is meant to be scheduled to is no longer executing properly.

Note that GPU failure WILL NOT cause Hydra to stop training - it will still work fine! However Hydra will no longer be able to use disconnected GPUs for parallelization. Use nvidia-smi occasionally to check if some GPUs have randomly dropped to 0%. All viable GPUs should be at 100% utilization, or close to it, so if you see lower, something is wrong. 

Adjusting the double-buffer size or even increasing/decreasing batch size may help as well.

## Running

The files `examples/single-task-vision.py` and `examples/multi-task-lm.py` demonstrate how to setup a simple training job. A vocab file for BERT and a vision benchmark dataset are also included. 

To use the system, first define your model and dataloaders using standard PyTorch APIs. Specification is easiest using PyTorch's `nn.Sequential` wrapper, but any module will do so long as the layers are listed in the order you wish them to execute.

Once the dataloader and model is defined, you can pass them into Hydra's wrappers as follows:

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


## Limitations

### Optimizers
Currently, this system only supports SGD (see hydra/nn/Model.py). However, extending it to support other optimizers
should not be too difficult, and we plan to implement it ASAP.


### Architectures
Recurrent architectures cannot be trained with Hydra, as we assume the architecture will be trained with a simple forward -> backward approach.
Residual architectures CAN be trained with Hydra, but must be defined in a way such that residual outputs are passed through intermediate layers
with an identity function, or else put into a single residual block.

### Multi-node
The system is implemented for single-node, multi-GPU execution. I have not implemented multi-node execution just yet.

## NOTE

This system is under development, it will likely change quite a bit in the coming weeks.

## Publications
If you use this system, please cite the following:
```
@misc{nagrecha2021hydra,
      title={Hydra: A System for Large Multi-Model Deep Learning}, 
      author={Kabir Nagrecha and Arun Kumar},
      year={2021},
      eprint={2110.08633},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}

@inproceedings{hydravision,
author = {Nagrecha, Kabir},
title = {Model-Parallel Model Selection for Deep Learning Systems},
year = {2021},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3448016.3450571},
doi = {10.1145/3448016.3450571},
journal = {Proceedings of the 2021 International Conference on Management of Data},
pages = {2929–2931},
}


```

