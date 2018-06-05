import os, glob, random
import skimage.io
import skimage.transform
import numpy as np
from sklearn.model_selection import train_test_split


def make_batch(iterable, batch_size=1):
    length = len(iterable)
    for index in range(0, length, batch_size):
        yield iterable[index:min(index + batch_size, length)]
    # Dataset classes
    # B  = Benign
    #       A = Adenosis
    #  	    F = Fibroadenoma
    #       TA = Tubular Adenoma
    #       PT = Phyllodes Tumor
    #  M  = Malignant
    #	    DC = Ductal Carcinoma
    #       LC = Lobular Carcinoma
    #       MC = Mucinous Carcinoma (Colloid)
    #       PC = Papillary Carcinoma
class CancerDataset(object):
    """ Cancer dataset reader """

    def __init__(self, data_dir="/home/murilo/dataset/BreaKHis_v1/img40/", test_size=0.3, dataset_mode=1):
        if dataset_mode == 1:
            self.categories = {"B": 0, "M": 1}
            class_position = 1
        else:
            self.categories = {"A": 0, "F": 1, "TA": 2, "PT": 3, "DC": 4, "LC": 5, "MC": 6, "PC": 7}
            class_position = 2
        # self.image_files = list(glob.glob(os.path.join(data_dir, split, "*.png")))
        self.image_files = list(glob.glob(os.path.join(data_dir, "*.png")))
        self.labels = [self.categories.get(os.path.basename(path).strip().replace("-","_").split("_")[class_position], -1)
                       for path in self.image_files]
        # Do dataset split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.image_files, self.labels,
                                                                                stratify=self.labels,
                                                                                test_size=test_size,
                                                                                random_state=159)

    def rescale(self, img, input_height, input_width):
        aspect = img.shape[1] / float(img.shape[0])
        if aspect > 1:
            return skimage.transform.resize(img, (input_width, int(aspect * input_height)))
        elif aspect < 1:
            return skimage.transform.resize(img, (int(input_width / aspect), input_height))
        else:
            return skimage.transform.resize(img, (input_width, input_height))

    def crop_center(self, img, cropx, cropy):
        y, x, c = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]

    def prepare_image(self, img_path):
        img = skimage.io.imread(img_path)
        img = skimage.img_as_float(img)
        img = self.rescale(img, 227, 227)
        img = self.crop_center(img, 227, 227)
        img = img.swapaxes(1, 2).swapaxes(0, 1)  # HWC to CHW dimension
        img = img[(2, 1, 0), :, :]  # RGB to BGR color order
        img = img * 255 - 128  # Subtract mean = 128
        return img.astype(np.float32)

    def getitem(self, index, train=True):
        if train:
            image = self.prepare_image(self.X_train[index])
            label = self.y_train[index]
            file = self.X_train[index]
        else:
            image = self.prepare_image(self.X_test[index])
            label = self.y_test[index]
            file = self.X_test[index]

        return image, label, file

    def len(self, train=True):
        if train:
            return len(self.y_train)
        else:
            return len(self.y_test)

    def read(self, batch_size=50, shuffle=False, train=True):
        """Read (image, label) pairs in batch"""
        order = list(range(self.len(train)))
        if shuffle:
            random.shuffle(order)
        for batch in make_batch(order, batch_size):
            images, labels, files = [], [], []
            for index in batch:
                # image, label = self[index]
                image, label, file = self.getitem(index, train)
                images.append(image)
                labels.append(label)
                files.append(file)

            if len(labels) < batch_size:
                batch_size = len(labels)
            yield np.stack(images).astype(np.float32), \
                  np.stack(labels).astype(np.int32).reshape((batch_size,)), \
                  np.stack(files)
