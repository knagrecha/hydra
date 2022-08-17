# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pynvml
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import gc

def get_free_space(idx=0):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(idx)
    info = nvmlDeviceGetMemoryInfo(h)
    return info.free

def get_used_space(idx=0):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(idx)
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
    mem = 0
    for p in procs:
        mem += p.usedGpuMemory / (1024 * 1024)
    return mem

"""
    Handles device movement with awareness of data type.

"""

def delete_batch(batch):
    if not isinstance(batch, torch.Tensor):
        while (len(batch) > 0):
            del batch[0]
    else:
        del batch

"""
    Handles device movement with awareness of data type.

"""

def move_batch_to_device(batch, device):
    if not (isinstance(batch, torch.Tensor)):
        return [x.to(device, non_blocking=True) for x in batch]
    else:
        return batch.to(device, non_blocking=True)
    
"""
    Add gradient tracking with awareness of data type.

"""

def track_gradients(batch):
    if not (isinstance(batch, torch.Tensor)):
        for x in batch:
            track_gradients(x)
    else:
        batch.requires_grad_(True)
        
"""
    Remove gradient tracking with awareness of data type.

"""

def untrack_gradients(batch):
    if (not isinstance(batch, torch.Tensor)):
        gradients = [i.grad for i in batch]
        for m_input in batch:
            m_input.requires_grad_(False)
    else:
        gradients = batch.grad
        toy_input.requires_grad_(False)

    return gradients
