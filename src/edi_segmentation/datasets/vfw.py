import torch
import torchvision.transforms as vt
from PIL import Image
import nestedtext as nt
from os import listdir

def pop_fn():
    i = 0
    while True:
        yield i
        i+=1

g = pop_fn()

class VfwScene():

    cls_names = {"BOTTLE":1, "CAN":2}
    transforms = [
        vt.PILToTensor(),
        vt.ConvertImageDtype(torch.float),
    ]

    def __init__(self, path):
        self._t = vt.Compose(VfwScene.transforms)
        self._parse_directory(path)
        self._img = None
        self._masks = None
    
    def _parse_directory(self, path):
        fnames = [f"{path}/{p}" for p in listdir(path)]
        image_names = list(filter(lambda s: ".png" in s, fnames))
        self._mask_fnames = list(filter(lambda s: "obj" in s, image_names))
        self._img_fname = list(filter(lambda s: s not in self._mask_fnames, image_names))[0]
        self._annotations = nt.load(list(filter(lambda s: ".nt" in s, fnames))[0])

    def load(self, retain=True):

        if retain and self._img is not None:
            return (self._img, {
                "obj_type": self._cls_tensor,
                "bbox":     self._bbox_tensor,
                "masks":    self._masks
            })

        img = self._t(Image.open(self._img_fname))

        img_norm = vt.Normalize(img.mean(dim=(1,2)), img.std(dim=(1,2)))
        img = img_norm(img)

        classes = []
        bboxes = []
        masks = []

        for mask_fname in self._mask_fnames:

            index = mask_fname.split(".")[0][-3:]
            info = self._annotations[f"object_{index}"]

            if float(info["percent_visible"]) < 95:
                continue 

            obj_class = torch.zeros(1,3)
            obj_class[0,VfwScene.cls_names[info["object_type"]]] += 1
            bbox = torch.tensor([float(info["bounding_box"][k]) for k in ["bb_X", "bb_Y", "bb_width", "bb_height"]]).reshape(1,-1)

            classes.append(obj_class)
            bboxes.append(bbox)
            masks.append(self._t(Image.open(mask_fname)))
        
        zero_class = torch.zeros(1,3)
        zero_bbox = torch.zeros(1,4)
        zero_mask = torch.zeros((1,)+img.shape[-2:])
        zero_count = 100 - len(classes)

        cls_tensor = torch.cat(classes + [zero_class] * zero_count)
        bbox_tensor = torch.cat(bboxes + [zero_bbox]* zero_count)
        mask_tensor = torch.cat(masks + [zero_mask] * zero_count)

        label = {
            "obj_type": cls_tensor,
            "bbox":     bbox_tensor,
            "masks":    mask_tensor
        }

        if retain:
            self._bbox_tensor = bbox_tensor
            self._cls_tensor = cls_tensor
            self._img = img
            self._masks = mask_tensor

        return (img, label)

class VfwSubset():

    def __init__(self, path):
        self.scenes = self._populate_scenes(path)

    def _populate_scenes(self, path):
        scenes = []
        for scene in listdir(path):
            scenes.append(VfwScene(f"{path}/{scene}"))
        return scenes

    def load_item(self, idx):
        return self.scenes[idx].load()

    def __len__(self):
        return len(self.scenes)

class VfwDataset():

    def __init__(self, root="data/VFW"):
        self._subsets = self._populate_subsets(root)
    
    def _populate_subsets(self, root):
        subsets = []
        for d in listdir(root):
            subsets.append(VfwSubset(f"{root}/{d}"))
        return self._link_indices(subsets)

    def _link_indices(self, subsets):
        running_total = 0
        subsets_with_ranges = []
        for subset in subsets:
            start = running_total
            running_total += len(subset)
            stop = running_total
            subsets_with_ranges.append((subset, start, stop))
        return subsets_with_ranges

    def __getitem__(self, idx):
        for subset, start, stop in self._subsets:
            if idx < 0:
                idx = len(self) + idx
            if start <= idx < stop:
                return subset.load_item(idx-start)
        raise IndexError("Dataset index out of range")

    def __len__(self):
        return self._subsets[-1][2]


if __name__ == "__main__":
    ds = VfwDataset()
    x= ds[0]
    ds[51]
    ds[-1]
    ds[-51]
    ds[-100]
    len(ds)
    pass