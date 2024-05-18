import torch
def pcgrad(model, losses):
    # First set all grads to None
    grad_dict = dict()
    # Store the task specific grads in a dict
    for lid, loss in enumerate(losses):
        zero_grad_model(model)
        loss.backward(retain_graph=True)
        grad_dict[lid] = {name:param.grad for name,param in model.named_parameters()}
    
    # Iterate through all grads and conduct surgeries
    for lid, 




def zero_grad_model(model):
    for _, param in model.named_parameters():
        param.grad = None