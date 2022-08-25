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
        self.labels = np.asarray(self.bytes[8:])
        assert len(self.labels) == self.items

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
            images.append(pxl_list.reshape((n_rows, n_cols)))
        return np.asarray(images)


class MnistDataset():

    def __init__(self, datapath, labelpath):
        self.data = ImageFile(datapath)
        self.labels = LabelFile(labelpath)
        self._length = self.data.items
        assert self._length == self.labels.items
    
    def __getitem__(self, idx):
        return (self.data.images[idx], self.labels.labels[idx])
    
    def __len__(self):
        return self._length    


if __name__ == "__main__":
    m = ImageFile("datasets/MNIST/t10k-images.idx3-ubyte")
    l = LabelFile("datasets/MNIST/t10k-labels.idx1-ubyte")
    train = MnistDataset("datasets/MNIST/train-images.idx3-ubyte", "datasets/MNIST/train-labels.idx1-ubyte")
    test = MnistDataset("datasets/MNIST/t10k-images.idx3-ubyte", "datasets/MNIST/t10k-labels.idx1-ubyte")