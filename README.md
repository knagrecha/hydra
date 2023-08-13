# Hydra

Hydra is a large-model execution engine optimized for multi-task execution. It enables arbitrarily large models to be trained on a single GPU by supporting sequential data spilling, and generates near-optimal training schedules for sharded multi-model training using a novel scheduling strategy.

Contact knagrech@ucsd.edu for more info.

## Install

To install Hydra, follow the [Installation Guide](https://github.com/knagrecha/hydra/blob/main/INSTALL.md).

## Important
To run the model selection workload go to the examples folder and run  

``PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 hydra_12_models_sequential.py --seed 0.``

If you get block allocation errors (e.g.):

``Assertion `block.allocated' failed.  at ../aten/src/THC/THCCachingHostAllocator.cpp:158``

Try tuning the 128 parameter up.  We are looking into setting up our own caching allocator that does not rely on PyTorch.

To run the scalability workload use examples/scaling_hydra.py. We do not use the split prefix for that command.
    
Please note that running 12-task single node experiments (like the paper) is an expensive operation that demands a great deal of DRAM and continuous, heavy GPU utilization. If you want to run a smaller scale version (6 tasks) just to observe the system, I have also prepared two files which represent the 12 tasks split into two. These are examples/hydra_6_0_models_sequential.py and examples/hydra_6_1_models_sequential.py. Use these if you don't have enough DRAM to run the 12-task experiment (~300GB needed).

The PyTorch CUDA split command is very important. PyTorch isn't optimized for constant swaps, so their caching allocator produces a lot of fragmentation and unnecessary OOMs (e.g. 6 GB reserved, fails to allocate 1GB because no valid block size was found). The split command essentially reduces the cacher's flexibility in this regard.

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
    
## Benefits
In the single GPU environment, you should observe that you can train models several orders of magnitude larger than before. For example, with a K80 GPU and 500GB DRAM, naive execution only enables training an 850M parameter GPT-2 like architecture, but with Hydra, we were able to scale up to 110B parameters. 
    
For model selection workloads, multi-GPU users can enjoy linear strong scaling as task parallelism enables maximal processor utilization. Note that if you have fewer tasks than processors, some devices will go unused. We are looking to hybridize with data parallelism to mitigate this issue (contributions welcome). In particular we are interested in using ZeRO-DDP.


## Limitations

### Optimizers
Currently, this system only supports SGD (see hydra/nn/Model.py). However, extending it to support other optimizers
should not be too difficult, and we plan to implement it ASAP.


### Architectures
Recurrent architectures cannot be trained with Hydra, as we assume the architecture will be trained with a simple forward -> backward approach.
Residual architectures CAN be trained with Hydra, but must be defined in a way such that residual outputs are passed through intermediate layers
with an identity function, or else put into a single residual block.

### Multi-node
The system is implemented for single-node, multi-GPU execution. We are working to implement multi-node execution.

## CUDA Optimizations
We only use high-level PyTorch APIs for data transfer. While this helps with compatability as frameworks evolve, we're likely missing out on some major potential speedups for CPU-GPU optimization. In the future we plan to optimize this to enable better spilling performance.

## FAQs

**Why do I get Out-of-Memory Errors sometimes? The OOM's are inconsistent, and change depending on my hardware.**

This is a known issue. The Pilot partitioner (default) attempts to estimate shard memory costs by running sample passes. 
However, during real execution, minibatch memory costs can and do vary! Occasionally, one minibatch or another causes memory usage peaks
that create OOM's when combined with the pre-loaded parameters from double-buffering. There are three quick fixes that are possible:

1) (Recommended) Increase the double-buffer space until the shard sizes are reduced. This will reduce per-shard memory costs by increasing the free space guarantee.
2) Turn off double-buffering (CACHE_SYSTEM flag in ModelOrchestrator). WARNING! This will induce a 1.5-2X slowdown. Use solution 1 if possible.
3) Use Presharded partitioner and adjust the partition boundaries yourself. This gives you more fine-grained control at the cost of taking some time to input the values yourself.

We are attempting to create a more exact partitioning algorithm to address this issue fully. In the meantime, use the quick fixes.

## Publications
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

