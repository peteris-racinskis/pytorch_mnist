import torch
import torchvision
import torchvision.transforms.functional as vf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from typing import Tuple, List

def show_img(mat: torch.Tensor):
    img = fit_to_range(mat, 0, 255).numpy().astype(np.uint8)
    cv.imshow("image", img.reshape(img.shape[-2:]))
    cv.waitKey()

def fit_to_range(mat: torch.Tensor, newmin, newmax):
    old_scale = mat.max() - mat.min()
    old_offs = mat.max() - (old_scale / 2)
    new_scale = newmax - newmin
    new_offs = newmax - (new_scale / 2)
    return (mat - old_offs) * (new_scale / old_scale) + new_offs

def show_imgs(items: List[Tuple[torch.Tensor, torch.Tensor]]):
    imgs = []
    labels = []
    for item in items:
        img, label = item
        imgs.append(fit_to_range(img, 0, 255))
        labels.append(str(label.argmax().item()))
    title = labels
    grid = torchvision.utils.make_grid(imgs)
    grid = np.asarray(grid).transpose(1,2,0)
    plt.title(title)
    plt.imshow(grid)
    plt.show()
    pass

    
if __name__ == "__main__":


    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(0.5,0.5, True)
        ]
    )
    from datasets.mnist import MnistDataset
    train_ds = MnistDataset("datasets/MNIST/train-images.idx3-ubyte", "datasets/MNIST/train-labels.idx1-ubyte", transform=transforms)

    show_imgs(list(train_ds[i] for i in range(25)))