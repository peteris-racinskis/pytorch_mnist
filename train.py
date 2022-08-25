import torch, torchvision

x = torch.tensor([1.,2.], requires_grad=True)
y = torch.tensor([3.,4.], requires_grad=True)
''' 
Graph:
    x     y
    |     |
   pow   pow
    |     |
   mul    |
      \   /
       sub
        |
        z
        |
    sum (aggregate)
        |
    final.backward()

'''
z = x.pow(3).mul(3) - y.pow(2) 
#
z.sum().backward() # the thing you cann backward on has to be a vector gradient of a scalar function (external) or scalar valued
dx = x.grad # partial derivative dz/dxi for each xi
dy = y.grad # partial derivative dz/dyi for each yi


# finetuning - load a model, freeze all its parameters you don't want to tune
model = torchvision.models.resnet18(pretrained=True)
for p in model.parameters(): # all the tensors registered in the parameter iterator
    p.requires_grad = False

old_in_features = model.fc.in_features
old_out_features = model.fc.out_features
## NN LINEAR: has a PARAMETER for weights and a PARAMETER for biases
model.fc = torch.nn.Linear(old_in_features, old_out_features) # swap the pretrained, frozen layer with a new one