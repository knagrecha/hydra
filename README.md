# Hydra

Hydra is a model parallel execution engine optimized for multi-task execution. It enables arbitrarily large models to be trained on a single GPU by supporting sequential data spilling, and generates near-optimal training schedules for sharded multi-model training using a novel scheduling strategy.

Contact knagrech@ucsd.edu for more info.

## Install

To install Hydra, follow the [Installation Guide](https://github.com/knagrecha/hydra/blob/main/INSTALL.md).

## Saved Intermediates Mode
This branch takes a different approach to intermediate generation during scheduling. The standard branch uses a "checkpointed" approach to regenerate intemediates between forward and backward passes. This allows the forward pass to run much faster as it does not need to maintain gradients, and each shard execution stage is self-contained. Saved intermediates mode actually packes intermediates into shards by taking advantage of a new PyTorch hook from 1.10, but empirical testing has shown that this technique is not actually faster than checkpointing mode. However if you do wish to use this approach the option is there with this branch.

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

## FAQs

**Why do I get Out-of-Memory Errors sometimes? The OOM's are inconsistent, and change depending on my hardware.**

This is a known issue. The Pilot partitioner (default) attempts to estimate shard memory costs by running sample passes. 
However, during real execution, minibatch memory costs can and do vary! Occasionally, one minibatch or another causes memory usage peaks
that create OOM's when combined with the pre-loaded parameters from double-buffering. There are two quick fixes that are possible:

1) (Recommended) Increase the double-buffer space until the shard sizes are reduced. This will reduce per-shard memory costs by increasing the free space guarantee.
2) Turn off double-buffering (CACHE_SYSTEM flag in ModelOrchestrator). WARNING! This will induce a 1.5-2X slowdown. Use solution 1 if possible.

We are attempting to create a more exact partitioning algorithm to address this issue fully. In the meantime, use the quick fixes.

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

