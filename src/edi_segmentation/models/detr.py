import torch
import torch.nn.functional as nf

from torch.utils.data import DataLoader
from torch.nn import Module, Linear, Sequential
from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import resnet50, ResNet

from edi_segmentation.datasets.vfw import VfwDataset
from edi_segmentation.utils.templates import get_fb_resnet, get_fb_transformer, get_fb_intermediates


class DetrWithMask(Module):

    def __init__(self, fb_root="saved_models/detr_", classes=3):
        super().__init__()
        
        # Load weights from detr's standalone colab
        self.backbone = get_fb_resnet(fb_root)
        self.transformer = get_fb_transformer(fb_root)
        self.conv, self.query, self.r_emb, self.c_emb = get_fb_intermediates(fb_root)
        
        transformer_output_dim = 256
        self.fc_class = Linear(transformer_output_dim, classes)
        self.fc_bbox = Linear(transformer_output_dim, 4)
    
    def forward(self, img):
        # img -> 2048 channel patches (15x20 for 480x640 img)
        x = self.backbone(img)
        # 2048 channel patches -> 256 channel patches
        x = self.conv(x)
        # Dynamically shaped transformer input depending on image size
        h, w = x.shape[-2:]
        # Get batch count for query shaping
        b = x.shape[0]
        # pick h first embedding vectors (128 dim each), fill out h x w x 128 tensor
        r_emb = self.r_emb[:h].unsqueeze(1).repeat(1,w,1)
        # pick w first embedding vectors (128 dim each), fill out h x w x 128 tensor
        c_emb = self.c_emb[:w].unsqueeze(0).repeat(h,1,1)
        # concatenate 128+128 vectors into 256dim encoder token format. Order determined by fb training. Stack batches.
        pos_emb = torch.cat([c_emb, r_emb], dim=-1).flatten(0,1).unsqueeze(1).repeat(1,b,1)
        # reshape patches to fit into transformer
        x = x.flatten(-2).permute(2,0,1)
        encoder_query = pos_emb + 0.1 * x
        decoder_query = self.query.unsqueeze(1).repeat(1,b,1)
        tokens = self.transformer(encoder_query, decoder_query)

        logits = nf.softmax(nf.relu(self.fc_class(tokens)))
        bboxes = nf.relu(self.fc_bbox(tokens))

        return {"pred_logits": logits, "pred_bboxes": bboxes}
    
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
    model = DetrWithMask().to(gpu)
    ds = VfwDataset()
    dl = DataLoader(ds, 10, False)
    img, label = next(iter(dl))
    model.eval()
    model(img.to(gpu))
    pass