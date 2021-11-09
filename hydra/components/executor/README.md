# Executor Modules

The executor module serves as a way to provide custom execution patterns
to given shards. Each ShardedTask is assigned an executor by the partitioner,
so new execution modules should be paired with new partitioner modules.

## Typical Module Patterns

There are three base modules covering three use-cases:

- Forward Pass Module
- Forward Pass with Loss Module
- Backward Pass Module

These three modules have different input/output specifications. If you wish to replace one of these modules
with a custom implementation, you must abide by the same specs so that the ModelOrchestrator can handle the
inputs and outputs appropriately.

Every module is a class with the following attributes:
- type (a simple name for the class)
- index (reference to the shard's index)

It must also have a function "run". The run function signature depends on the module type you are implementing.

---

### Forward Module

The forward module must support the following inputs (in order):
- a model
- a data batch
- a torch.device

It must return the following outputs (in order):
- the data batch to be passed to the next layer

### Forward-Loss Module
The forward-loss module must support the following inputs (in order):
- a model
- an optimizer
- a data batch
- labels
- a criterion
- a torch.device
- (optional) a scaler for Automatic Mixed Precision compute.

It must return the following outputs (in order):
- scaler (if not used, return None)
- the gradients being passed back to the next layer (return None if there is only a single shard)
- the loss (used in visualizations and debugging)

### Backward Module
The backward module must support the following inputs (in order):
- a model
- an optimizer
- a data batch (checkpointed data that would be used as input for that shard in a forward pass)
- a torch.device
- the gradients that will be used in the Jacobian-vector product.
- (optional) a scaler for Automatic Mixed Precision compute.

It must return the following outputs (in order):
- scaler (if not used, return None)
- the gradients being passed back to the next layer (return None if there is only a single shard)




