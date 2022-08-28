import torch
import torch.nn.functional as nf

from torch.utils.data import DataLoader
from torch.nn import Module, Linear, Sequential
from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import resnet50, ResNet

from edi_segmentation.datasets.vfw import VfwDataset


class DetrWithMask(Module):

    def __init__(self, dl_weights=False, classes=3):
        super().__init__()
        # Find these by running the resnet in debugger
        conv_output_channels = 2048
        conv_output_height = 15
        conv_output_width = 20
        # Load weights when no saved version of the model is available
        self.backbone = self._restructure_resnet(
            resnet50(ResNet50_Weights) if dl_weights else resnet50())
        
        self.fc_class = Linear(conv_output_channels*conv_output_height*conv_output_width, classes)
        self.fc_bbox = Linear(conv_output_channels*conv_output_height*conv_output_width, 4)
    
    def forward(self, img):
        x = self.backbone(img)
        x = torch.flatten(x, -3) # C x H x W
        # FUCKING BRILLIANT -- nn.Linear(input_dim=(x),output_dim=(y))(Tensor(n,x)) -> Tensor(n,y)
        logits = nf.softmax(nf.relu(self.fc_class(x)))
        bboxes = nf.relu(self.fc_bbox(x))
        return x
    
    @staticmethod
    def _restructure_resnet(resnet: ResNet):
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
    dl = DataLoader(ds, 1, False)
    img, label = next(iter(dl))
    model = DetrWithMask(True).to(gpu)
    model.eval()
    model(img.to(gpu))
    pass