import torch
import copy
import numpy as np
def pcgrad(model, losses):
    num_tasks = len(losses)
    # First set all grads to None
    grad_dict = dict()
    # Store the task specific grads in a dict
    for lid, loss in enumerate(losses):
        zero_grad_model(model)
        loss.backward(retain_graph=True)
        grad_dict[lid] = {name:param.grad for name,param in model.named_parameters()}
    
    # Iterate through all grads and conduct surgeries
    for name,param in model.named_parameters():
        # get all grads w.r.t. to loss for this param
        this_grads = [grad_dict[lid].get(name, None) for lid in range(num_tasks)]
        modified_grad = grad_surgery(this_grads)
        param.grad = modified_grad

def zero_grad_model(model):
    for _, param in model.named_parameters():
        param.grad = None

def grad_surgery(grads):
    num_tasks = len(grads)
    grads_cp = copy.deepcopy(grads) # keeps original grad value
    for grad_idx, gradi in enumerate(grads):
        for grad_jdx in np.random.shuffle(np.arange(num_tasks)):
            gradj = grads_cp[grad_jdx]
            if grad_idx==grad_jdx or gradi is None or gradj is None:
                continue
            # First determine if the two have negative cosine similarity
            gradi_flat = gradi.flatten()
            gradj_flat = gradj.flatten()
            csim = torch.dot(gradi_flat, gradj_flat)
            if csim >= 0:
                continue
            else: # Perform gradient surgery
                gradi_flat = gradi_flat - gradj_flat * csim/(torch.linalg.vector_norm(gradj_flat,ord=2)**2)
                gradi = torch.reshape(gradi_flat, gradj.shape)
    return grads

