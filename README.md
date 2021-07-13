# hydra


Prepped for SIGMOD '22. Code is VERY unorganized. Plan to clean it up before open-sourcing.

Brief overview:

-Partitioner is in nn/Model.

-Container for arbitrary sequential execution is in nn/Container.

-Sharded-LRTF is in ModelOrchestrator.py

-SHARP is in ModelOrchestrator.py

-Double-buffering is in ModelOrchestrator.py by use of non-blocking execution

-Spilling is in ModelOrchestrator.py by use of .cpu() and .to() functions

-ModelTask.py contains ancillary functions and variables critical to execution.

