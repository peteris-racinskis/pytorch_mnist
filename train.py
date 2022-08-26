import torch
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
from torchvision.models.resnet import resnet18


from datasets.mnist import MnistDataset
from models.conv import SimpleCNN
from train_utils import show_imgs

EPOCHS=10

def colorize(batch: torch.Tensor) -> torch.Tensor:
    # new = torch.zeros_like(batch)
    # new = torch.cat((new, new), dim=1)
    return torch.cat((batch, batch, batch), dim=1)

transforms = T.Compose(
    [
        T.Normalize(0.5,0.5, True)
    ]
)

train_ds = MnistDataset("datasets/MNIST/train-images.idx3-ubyte", "datasets/MNIST/train-labels.idx1-ubyte", transform=transforms)
overfit_ds = MnistDataset("datasets/MNIST/t10k-images.idx3-ubyte", "datasets/MNIST/t10k-labels.idx1-ubyte", transform=transforms, take_first=500)
valid_ds = MnistDataset("datasets/MNIST/train-images.idx3-ubyte", "datasets/MNIST/train-labels.idx1-ubyte", transform=transforms, take_first=500)

train_dl = DataLoader(train_ds, 64, shuffle=True)
valid_dl = DataLoader(valid_ds, 1)
overfit_dl = DataLoader(overfit_ds, 1)

# show_imgs(list(train_ds[i] for i in range(5)))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# use custom model
# model = SimpleCNN()
# use a pre-trained model 
model = resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model = model.to(device)

sgd = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(sgd, gamma=0.9)

lf = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):

    print(f"At epoch {epoch}")
    model.train()
    for i, (img_batch, label_batch) in enumerate(train_dl):
        img_batch = colorize(img_batch)
        sgd.zero_grad() # reset the gradients of all parameters to zero for new accumulation
        infer_batch = torch.nn.functional.softmax(model(img_batch.to(device)), dim=1)
        loss = lf(infer_batch, label_batch.to(device))
        loss.backward()
        sgd.step()
        if i % 250 == 0:
            print(f"Batch: {i} Loss: {loss.item()}")
            # model.eval()
            # vl = 0
            # corr = 0
            # total = 0
            # with torch.no_grad():
            #     for v_img_batch, v_lab_batch in valid_dl:
            #         v_img_batch = colorize(v_img_batch)
            #         infer_batch = model(v_img_batch.to(device))
            #         corr = corr + 1 if torch.argmax(infer_batch) == torch.argmax(v_lab_batch) else corr
            #         total += 1
            #         vl += torch.nn.functional.cross_entropy(infer_batch, v_lab_batch.to(device))
            # print(f"Batch: {i}")
            # print(f"Total validation loss: {vl.item()}")
            # print(f"Accuracy: {corr} / {total} : {corr/total * 100}")
            # model.train()
    scheduler.step()

    model.eval()
    vl = 0
    corr = 0
    total = 0
    vl_of = 0
    corr_of = 0
    total_of = 0
    with torch.no_grad():
        for v_img_batch, v_lab_batch in valid_dl:
            v_img_batch = colorize(v_img_batch)
            infer_batch = torch.nn.functional.softmax(model(v_img_batch.to(device)), dim=1)
            corr = corr + 1 if torch.argmax(infer_batch) == torch.argmax(v_lab_batch) else corr
            total += 1
            vl += torch.nn.functional.cross_entropy(infer_batch, v_lab_batch.to(device))
        for v_img_batch, v_lab_batch in overfit_dl:
            v_img_batch = colorize(v_img_batch)
            infer_batch = torch.nn.functional.softmax(model(v_img_batch.to(device)), dim=1)
            corr_of = corr + 1 if torch.argmax(infer_batch) == torch.argmax(v_lab_batch) else corr
            total_of += 1
            vl_of += torch.nn.functional.cross_entropy(infer_batch, v_lab_batch.to(device))
    print(f"Total validation loss: {vl.item()}")
    print(f"Accuracy: {corr} / {total} : {corr/total * 100}%")
    print(f"Total train loss: {vl_of.item()}")
    print(f"Accuracy (train/overfit): {corr_of} / {total_of} : {corr_of/total_of * 100}%")

torch.save(model.state_dict(), "saved_model_resnet")
