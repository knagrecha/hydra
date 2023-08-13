Developers Guide
---

## Code Organization

In this section, we describe the structure of the codebase. 

### Representations

In hydra/components, we define the two key components of Hydra, the executor (which runs forward and backward passes), and the partitioner, which determines how to chunk a model architecture.

### Orchestrator/Task
In hydra/ModelOrchestrator.py and hydra/ModelTask.py, we introduce the critical operators of Hydra that determine execution tasks and orchestrate training.

## Tests & Examples
In the examples folder, we provide example training pipelines with Hydra.
