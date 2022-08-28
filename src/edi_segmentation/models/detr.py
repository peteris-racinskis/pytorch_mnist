import torch
import torch.nn.functional as nf
from torch.nn import Module, Linear, Sequential
from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import resnet50, ResNet

from edi_segmentation.datasets.vfw import VfwDataset


class DetrWithMask(Module):

    def __init__(self, dl_weights=False, classes=3):
        super().__init__()
        self.backbone = self._restructure_resnet(
            resnet50(ResNet50_Weights) if dl_weights else resnet50())
        self.fc_test = Linear(self.backbone.fc.out_features, classes)
    
    def forward(self, img):
        x = self.backbone(img)
        x = nf.relu(self.fc_test(x))
        return x
    
    def _restructure_resnet(self, resnet: ResNet):
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

if __name__ == "__main__":
    gpu = torch.device('cuda')
    cpu = torch.device('cpu')
    ds = VfwDataset()
    img, label = ds[0]
    model = DetrWithMask(True).to(gpu)
    model.eval()
    model(img.to(gpu))
    pass