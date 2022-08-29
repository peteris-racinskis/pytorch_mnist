import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.ops import complete_box_iou_loss, box_convert
from torch.utils.data import DataLoader

from edi_segmentation.models.detr import DetrPartialPreload
from edi_segmentation.datasets.vfw import VfwDataset

ds = VfwDataset()
train_dl = DataLoader(ds, 16, True)
valid_dl = DataLoader(ds, 1, False)

cpu = torch.device('cpu')
dev = torch.device('cuda') if torch.cuda.is_available() else cpu

model = DetrPartialPreload()
#torch.save(model.state_dict(), "saved_models/detr_refactored_notrain")
model.train()
model.to(dev)

cls_loss = torch.nn.CrossEntropyLoss()
box_loss = complete_box_iou_loss

opt = SGD(model.parameters(), 1e-3, 0.9)
scd = ExponentialLR(opt, 0.9)

# for ind, (i, l) in enumerate(train_dl):
#     print(f"Hit item {ind} Img batch shape: {i.shape}")

EPOCHS=100
EPS = 1e-6
for e in range(EPOCHS):
    print(f"Starting epoch {e}...")
    total_loss = 0
    categorical_loss = 0
    bbox_loss = 0
    for img_batch, target_batch in train_dl:
        opt.zero_grad()

        t_types_batch = target_batch["obj_type"].to(dev)
        t_boxes_batch = target_batch["bbox"].to(dev)
        t_valid_batch = target_batch["n_valid"]
        infer = model(img_batch.to(dev))
        i_types_batch = infer["pred_logits"]
        i_boxes_batch = infer["pred_bboxes"]

        # Add epsilon to the x2, y2 coordinates of the bounding box to make the loss not shit the bed
        #eps = torch.zeros_like(i_boxes_batch).to(dev)
        #eps[:,:,2:] += EPS
        i_boxes_batch = box_convert(i_boxes_batch, "cxcywh", "xyxy")
        #i_boxes_batch = i_boxes_batch + eps
        
        loss = torch.tensor(0.).to(dev)

        for i in range(len(t_boxes_batch)):
            closs = cls_loss(t_types_batch[i,:t_valid_batch[i]], i_types_batch[i,:t_valid_batch[i]])
            bloss = box_loss(t_boxes_batch[i,:t_valid_batch[i]], i_boxes_batch[i,:t_valid_batch[i]], reduction="sum")
            categorical_loss += closs.item()
            bbox_loss += bloss.item()
            loss += closs + bloss
        if torch.isnan(loss):
            print()

        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch: {e} loss at the end: {total_loss} categorical: {categorical_loss} bbox: {bbox_loss}")
    scd.step()

torch.save(model.state_dict(), "saved_models/detr_refactored_cls_bbox_unfrozen-queries-embeddings_boxconvert_ep100")
