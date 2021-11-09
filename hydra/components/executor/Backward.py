from hydra.utilities import delete_batch, move_batch_to_device
import torch

class Backward():
    def __init__(self, idx):
        self.type="Backward"
        self.idx = idx

    def run(self, model, optimizer, batch_input, labels, criterion, device, scaler, back_input):
        
        model.to(device, non_blocking=True)
        model.zero_grad()  # zeroes the gradient buffers of all parameters
        optimizer.zero_grad()  # zero the gradient buffers

        if not isinstance(back_input, torch.Tensor):
            #print("Back input is a list")
            toy_input = [x.to(device, non_blocking=True) for x in back_input]

            if self.idx != 0:
                for m_input in toy_input:
                    #print(m_input)
                    if isinstance(m_input, torch.Tensor):
                        m_input.requires_grad_(True)

        else:
            toy_input = back_input.to(device, non_blocking=True)
            if self.idx != 0:
                toy_input.requires_grad_(True)     

        if not isinstance(batch_input, torch.Tensor):
            batch_input = [x.to(device, non_blocking=True) for x in batch_input]
        else:
            batch_input = batch_input.to(device, non_blocking=True) 


        with torch.cuda.amp.autocast():
            toy_output = model(toy_input)
        
        if scaler is not None:
            toy_output = scaler.scale(toy_output)
        torch.autograd.backward(toy_output, batch_input)
        del toy_output
        del batch_input
        pass_back_gradients = None
        if self.idx != 0: # the first backwards pass need not compute back pass gradients.
            if (not isinstance(toy_input, torch.Tensor)):
                pass_back_gradients = [i.grad for i in toy_input]
                for m_input in toy_input:
                    m_input.requires_grad_(False)
            else:
                pass_back_gradients = toy_input.grad
                toy_input.requires_grad_(False)
            # the user will pass in what WAS the input for this stage!
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad()


        if not isinstance(toy_input, torch.Tensor):
            while (len(toy_input) > 0):
                del toy_input[0]
            del toy_input
        else:
            del toy_input

        model.zero_grad()


        return scaler, pass_back_gradients 
