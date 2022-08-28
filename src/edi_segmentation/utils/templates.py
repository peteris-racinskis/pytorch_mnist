from ast import Param
import torch
from torch.nn import Parameter, Linear, Conv2d, Sequential
from torchvision.models.resnet import resnet50, ResNet
from edi_segmentation.utils.model_utils import freeze


def restructure_resnet(resnet: ResNet):
    """
    Layers up to avg_pool
    """
    return Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4
    )

def get_fb_transformer(root=None, path="transformer"):
    """
    Reproduce the transformer instance found in:
    https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=h91rsIPl7tVl
    """
    hidden_dim = 256
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    transformer = torch.nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

    if not root is None:
        transformer.load_state_dict(torch.load(root+path))
        transformer = freeze(transformer)

    return transformer

def get_fb_resnet(root=None, path="backbone"):
    """
    Reproduce the resnet instance found in:
    https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=h91rsIPl7tVl
    """

    resnet = resnet50()
    del resnet.fc

    if not root is None:
        resnet.load_state_dict(torch.load(root+path))
        resnet = freeze(resnet)
    
    return restructure_resnet(resnet)

def get_fb_intermediates(root=None, suffixes=["conv", "queries", "row_embed", "col_embed"], freeze_queries=True):
    conv = Conv2d(2048, 256, 1)
    if root is not None:
        conv.load_state_dict(torch.load(f"{root}{suffixes[0]}"))
        conv = freeze(conv)
        params = []
        for suffix in suffixes[1:]:
            p = torch.load(f"{root}{suffix}")
            if freeze_queries:
                p.requires_grad = False
            params.append(p)
    else:
        params = [
            Parameter(100, 256),
            Parameter(50, 128),
            Parameter(50, 128)
        ]
    return [conv] + params