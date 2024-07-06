import subprocess
import os
import abc
import hashlib
import zipfile
import glob
import logging
import tarfile
from skimage.io import imread
from PIL import Image
from tqdm import tqdm
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import pandas as pd

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
DATASETS_DICT = {"mnist": "MNIST",
                 "dsprites": "DSprites",
                 "celeba": "CelebA",
                 "til23": "TIL23",
                 "brca": "TCGA_BRCA",
                 "consep": "ConSep"}
DATASETS = list(DATASETS_DICT.keys())


def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).img_size


def get_background(dataset):
    """Return the image background color."""
    return get_dataset(dataset).background_color


def get_dataloaders(dataset, root=None, shuffle=True, pin_memory=True,
                    batch_size=128, logger=logging.getLogger(__name__), **kwargs):
    """A generic data loader

    Parameters
    ----------
    dataset : {"mnist", "fashion", "dsprites", "celeba", "chairs"}
        Name of the dataset to load

    root : str
        Path to the dataset root. If `None` uses the default one.

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)
    dataset = Dataset(logger=logger) if root is None else Dataset(root=root, logger=logger)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, **kwargs)


class DisentangledDataset(Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass


class TIL23():
    """TIL-23 Dataset from [1].

    description of TIL-23 ....

    Notes
    -----
    - Link : 

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] 

    """
    img_size = (3, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self, data_dir = "/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/data/classification/TCGA-TILs", \
    dataframe_path = "/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/data/classification/TCGA-TILs/images-tcga-tils-metadata.csv", \
    study_name = 'stad', logger=logging.getLogger(__name__)):
        self.directory = data_dir
        self.img_size = 64
        self.data = pd.read_csv(dataframe_path)
        if study_name != 'all':
            self.data = self.data[self.data['study'] == study_name]

        self.transform =  transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])

        self.kwargs = {'num_workers': 1, 'pin_memory': True}
        self.logger = logger

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path  = os.path.join(self.directory, self.data.iloc[idx]['path'])
        image = Image.open(path).convert("RGB")
        label = self.data.iloc[idx]['label']

        if label == 'til-negative':
            label_id = 0
        elif label == 'til-positive':
            label_id = 1

        if self.transform is not None:
            image = self.transform(image)
        return image, label_id
    

class TCGA_BRCA():
    img_size = (3, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self, data_dir = "/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/data/segmentation/TCGA_BRCA/CHC_VAE/TILE_1024_1024_Patch_128_128/synthetic/Train", logger=logging.getLogger(__name__)):
        self.img_size = 64
        self.data_dir = data_dir
        class_0_files = glob.glob(f"{self.data_dir}/0/*.npy")
        class_0_labels = [0 for i in range(len(class_0_files))]

        class_1_files = glob.glob(f"{self.data_dir}/1/*.npy")
        class_1_labels = [1 for i in range(len(class_1_files))]
 
        all_files = class_0_files + class_1_files
        all_labels = class_0_labels + class_1_labels
        temp = list(zip(all_files, all_labels))

        random.shuffle(temp)

        all_files, all_labels = zip(*temp)
        self.all_files, self.all_labels = list(all_files), list(all_labels)

        self.transform =  transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])

        self.kwargs = {'num_workers': 1, 'pin_memory': True}
        self.logger = logger

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        image_path = self.all_files[idx]
        data = np.load(image_path)
        if data.shape[2] == 4:
            image = data[:, :, :3].astype(np.uint8)
        else:
            image = data.astype(np.uint8)
        # image = np.load(image_path).astype(np.uint8)
        image = Image.fromarray(image)
        label = image_path.split('/')[-2]
        image = self.transform(image)
        return image, int(label)
    

class ConSep():
    img_size = (3, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self, data_dir = "/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/data/segmentation/CoNSeP/TILE_1024_1024_Patch_128_128/Binary/synthetic/Train", logger=logging.getLogger(__name__)):
        self.img_size = 64
        self.data_dir = data_dir
        class_0_files = glob.glob(f"{self.data_dir}/0/*.npy")
        class_0_labels = [0 for i in range(len(class_0_files))]

        class_1_files = glob.glob(f"{self.data_dir}/1/*.npy")
        class_1_labels = [1 for i in range(len(class_1_files))]
 
        all_files = class_0_files + class_1_files
        all_labels = class_0_labels + class_1_labels
        temp = list(zip(all_files, all_labels))

        random.shuffle(temp)

        all_files, all_labels = zip(*temp)
        self.all_files, self.all_labels = list(all_files), list(all_labels)

        self.transform =  transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])

        self.kwargs = {'num_workers': 1, 'pin_memory': True}
        self.logger = logger

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        image_path = self.all_files[idx]
        data = np.load(image_path)
        if data.shape[2] == 4:
            image = data[:, :, :3].astype(np.uint8)
        else:
            image = data.astype(np.uint8)
        # image = np.load(image_path).astype(np.uint8)
        image = Image.fromarray(image)
        label = image_path.split('/')[-2]
        image = self.transform(image)
        return image, int(label)


# HELPERS
def preprocess(root, size=(64, 64), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.

    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.

    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)


if __name__ == '__main__':
    tcga_brca = TCGA_BRCA()
    for batch in tcga_brca:
        image, label = batch
        print(image.shape)
        print(label)
        break
