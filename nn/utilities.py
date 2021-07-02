import pynvml
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import gc

def compute_weight_in_mem(layer, input_shape):
    temp = layer.instantiate(torch.device("cpu"))
    weight = torch.cuda.reserved_memory()
    del temp
    return weight

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
    
## MEM utils ##
import gc

import torch

## MEM utils ##
def mem_report_verbose():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' %(mem_type))
        print('-'*LEN)
        total_numel = 0
        total_mem = 0
        visited_data = set()
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.update([data_ptr])

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem) )
        print('-'*LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        print('-'*LEN)

    LEN = 65
    print('='*LEN)
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('='*LEN)


def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        total_numel = 0
        total_mem = 0
        visited_data = set()
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.update([data_ptr])

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size 
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())
         
        return total_mem

    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    host_tensors = [t for t in tensors if not t.is_cuda]
    return _mem_report(host_tensors, 'CPU')