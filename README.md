# Hydra

Hydra is a model parallel execution engine optimized for multi-task execution. It enables arbitrarily large models to be trained on a single GPU by supporting sequential data spilling, and generates near-optimal training schedules for sharded multi-model training using a novel scheduling strategy.

Contact knagrech@ucsd.edu for more info.

## Install

To install Hydra, clone this repo and import it as shown in the example.


## Usage

The main tools in Hydra are the "Model", "Task", and "Orchestrator". Wrap any model architecture that can run as a sequence (one layer after another), and wrap it in a Model container:

`model_hydra_0 = Model(model_0)`

Then, pass it to a Task container along with some training info:

`task_0 = ModelTask(model_0, loss_fn, dataloader_0, lr_0, epochs_0)`

Pass any number of tasks to an orchestrator:

`orchestra = ModelOrchestrator([task_0, task_1])`

Define orchestrator details and start training!

``` 
 orchestra.verbose = 1
 orchestra.buffer = None

 orchestra.generate()
 orchestra.train_models()
```
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

