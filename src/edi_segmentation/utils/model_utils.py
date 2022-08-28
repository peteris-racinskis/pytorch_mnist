from torch.nn import Module

def freeze(model: Module):
    for p in model.parameters():
        p.requires_grad = False
    return model