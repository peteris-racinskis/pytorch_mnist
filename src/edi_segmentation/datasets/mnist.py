import torch
import torchvision.transforms as T
import numpy as np

class MnistFile():
    """
    Format documented in: http://yann.lecun.com/exdb/mnist/
    """
    def __init__(self, filename):
        self._to_memory(filename)
    
    def _to_memory(self, file):
        with open(file, 'rb') as fb:
            self.buf = fb.read()
        self.words = self._to_ints(self._to_words(self.buf[:16], 4))
        self.bytes = np.frombuffer(self.buf, dtype=np.uint8)
        self.magic = self.words[0]
        self.items = self.words[1]
    
    @staticmethod
    def _to_int(bytes, ord="big"):
        return int.from_bytes(bytes, ord)

    @classmethod
    def _to_ints(cls, words, ord="big"):
        return [cls._to_int(x) for x in words]

    @staticmethod
    def _to_words(buffer, stride):
        words = []
        for i in range(0, len(buffer), stride):
            words.append(buffer[i:i+stride])
        return words

class LabelFile(MnistFile):
    """
    Format documented in: http://yann.lecun.com/exdb/mnist/
    """

    def __init__(self, filename):
        super().__init__(filename)
        assert self.magic == 2049
        self.labels = np.zeros((self.items, 10), dtype=np.float32)
        for i, b in enumerate(self.bytes[8:]): # LABELS START AT 8!!!!
            self.labels[i,b] = 1

class ImageFile(MnistFile):
    """
    Format documented in: http://yann.lecun.com/exdb/mnist/
    """
    
    def __init__(self, filename):
        super().__init__(filename)
        assert self.magic == 2051
        self.images = self._get_images()
        assert len(self.images) == self.items
    
    def _get_images(self):
        n_rows, n_cols = self.words[2:4]
        image_pixels = n_rows * n_cols
        total_pixels = image_pixels * self.items
        images = []
        for i in range(16, total_pixels, image_pixels):
            pxl_list = self.bytes[i:i+image_pixels]
            images.append(pxl_list.reshape((1, n_rows, n_cols)))
        return np.asarray(images)


class MnistDataset():

    def __init__(self, datapath, labelpath, transform=T.Normalize(0.5, 0.5, True), take_first=None):
        self.data = ImageFile(datapath)
        self.labels = LabelFile(labelpath)
        self._length = self.data.items
        self._transform = transform
        assert self._length == self.labels.items
        if not take_first is None:
            self._length = take_first
            self.data.images = self.data.images[:take_first]
            self.labels.labels = self.labels.labels[:take_first]
    
    def __getitem__(self, idx):
        img = torch.tensor(self.data.images[idx].astype(np.float32))
        label = torch.tensor(self.labels.labels[idx].astype(np.float32))
        img = self._transform(img)
        return (img, label)
    
    def __len__(self):
        return self._length    


if __name__ == "__main__":
    m = ImageFile("data/MNIST/t10k-images.idx3-ubyte")
    l = LabelFile("data/MNIST/t10k-labels.idx1-ubyte")
    train = MnistDataset("data/MNIST/train-images.idx3-ubyte", "data/MNIST/train-labels.idx1-ubyte")
    test = MnistDataset("data/MNIST/t10k-images.idx3-ubyte", "data/MNIST/t10k-labels.idx1-ubyte")