import torch
from torch.utils.data.dataloader import DataLoader
from train_utils import show_imgs
from models.conv import SimpleCNN
from datasets.mnist import MnistDataset

INSTANCES=32

cpu = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available else cpu
check_ds = MnistDataset("datasets/MNIST/t10k-images.idx3-ubyte", "datasets/MNIST/t10k-labels.idx1-ubyte", take_first=INSTANCES)
check_dl = DataLoader(check_ds, INSTANCES, shuffle=False)

model = SimpleCNN()
model.load_state_dict(torch.load("saved_model"))
model.to(device)
model.eval()

for imgs, labels in check_dl:
    predicted = model(imgs.to(device))

show_imgs(list(zip(imgs.to(cpu), predicted.to(cpu))))
pass