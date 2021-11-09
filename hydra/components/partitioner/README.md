# Partitioner Modules

The partitioner module enables the development of custom
partitioning algorithms for your models.

## Typical Module Patterns

The partitioner module is relatively simple. The class itself
has no attribute requirements - you may implement it however you wish.

It does, however, require one function: 'shard'. 

### Shard Function Specification

The shard function must comply with the following specification.

It must accept as input:

- a model
- a criterion
- a test batch (dataloader will send a batch in the form of a tuple (batch, label) ) 
- double-buffer space (necessary for the double-buffering to work)
- learning rate (assigned to ShardedTasks)
- verbosity

It must return as output:
- a list of forward-facing ShardedTasks
- a list of backward-facing ShardedTasks
- the estimated total runtime of the model.

Note that the final forward-facing ShardedTask usually covers both the last forward pass and the first backward pass, though this will depend on executor module implementations.


### ShardedTasks
The sharded tasks your partitioner returns must be assigned the correct executor module for them to function! The executor module will likely differ based on the shard's index and the shard's direction, so be sure to take this into account when devising new partitioner modules.

