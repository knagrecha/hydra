from hydra.utilities import delete_batch, move_batch_to_device
import torch

class Forward():
    def __init__(self, idx):
        self.type="Forward"
        self.idx = idx

    def run(self, model, batch_input, device):
    
        old = next(model.parameters()).device
        model.to(device, non_blocking=True)

        batch_input = move_batch_to_device(batch_input, device)
        
        with torch.no_grad() and torch.cuda.amp.autocast():
            ns_labels = model(batch_input)

        delete_batch(batch_input)
            
        return ns_labels