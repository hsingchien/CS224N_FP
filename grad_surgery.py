import torch
import copy
import numpy as np
def hmpcgrad(model, losses, log=False):
    num_tasks = len(losses)
    # First set all grads to None
    grad_dict = dict()
    # Numpy array to contain number of task conflicts across all params in this optimization step
    nconfs_total = np.zeros(num_tasks) if log else None
    # Store the task specific grads in a dict
    for kid, kloss in enumerate(losses):
        zero_grad_model(model)
        kloss.backward(retain_graph=True)
        grad_dict[kid] = {name:param.grad for name,param in model.named_parameters()}
    zero_grad_model(model)
    # Iterate through all grads and conduct surgeries
    for name,param in model.named_parameters():
        # get all grads w.r.t. to loss for this param
        this_grads = [grad_dict[kid].get(name, None) for kid in range(num_tasks)]
        grad_idx = [k for k,grad in enumerate(this_grads) if isinstance(grad, torch.Tensor)]
        # check if this_grads is filled with None
        if len(grad_idx)>1:
            modified_grad,nconfs = grad_surgery(this_grads)
            # take only valid grads
            modified_grad = [grad for grad in modified_grad if grad is not None]
            modified_grad = torch.stack(modified_grad, dim=0).sum(dim=0)
            param.grad = modified_grad
            # log conflicts
            if log:
                # nconfs -- Number of conflicts of this param
                nconfs_total += nconfs

        elif len(grad_idx)==1:
            # if there is only one valid grad, no need to do pcgrad, set this grad as the grad
            param.grad = this_grads[grad_idx[0]]
        else:
            param.grad = None
    return nconfs_total

def zero_grad_model(model):
    for _, param in model.named_parameters():
        param.grad = None

def grad_surgery(grads):
    num_tasks = len(grads)
    # container to store the number of conflicts
    nconfs = np.zeros(num_tasks)
    grads_cp = copy.deepcopy(grads) # keeps original grad value
    for grad_idx, gradi in enumerate(grads):
        for grad_jdx in np.random.permutation(num_tasks):
            gradj = grads_cp[grad_jdx]
            if grad_idx==grad_jdx or gradi is None or gradj is None:
                continue
            # First determine if the two have negative cosine similarity
            one_d = False
            if len(gradi.shape) == 1: # weights of layernorm is 1d
                gradi = gradi.unsqueeze(0)
                gradj = gradj.unsqueeze(0)
                one_d = True
            csim = torch.bmm(gradi.unsqueeze(1), gradj.unsqueeze(-1)).flatten()
            # Find idx of csim where the value is negative and do surgeries on the corresponding rows of gradi
            cflct_idx = torch.nonzero(csim<0, as_tuple=True)[0]
            nconf = len(cflct_idx)
            # print(grad_idx+grad_jdx-1, nconf)
            nconfs[grad_idx+grad_jdx-1] = np.max((nconf,nconfs[grad_idx+grad_jdx-1]))
            m = csim[cflct_idx]/(torch.linalg.vector_norm(gradj[cflct_idx,],ord=2,dim=1,keepdim=False)**2)
            gradi[cflct_idx,] = gradi[cflct_idx,] - gradj[cflct_idx,] * m.unsqueeze(-1)
            if one_d: 
                gradi = gradi.squeeze(0)
    return grads, nconfs
