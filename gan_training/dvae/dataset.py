"""dataset.py"""

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import pandas as pd


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class TIL23():
    def __init__(self, data_dir = "/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/data/classification/TCGA-TILs", \
    dataframe_path = "/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/data/classification/TCGA-TILs/images-tcga-tils-metadata.csv", \
    study_name = 'stad'):
        self.directory = data_dir
        self.img_size = 64
        self.data = pd.read_csv(dataframe_path)
        if study_name != 'all':
            self.data = self.data[self.data['study'] == study_name]

        self.transform =  transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])

        self.kwargs = {'num_workers': 1, 'pin_memory': True}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        extracted_path = self.data.iloc[idx]['path'][17:]
        # path  = os.path.join(self.directory, self.data.iloc[idx]['path'])
        path = f"{self.directory}/stain_normalized_images/{extracted_path}"
        image = Image.open(path).convert("RGB")
        label = self.data.iloc[idx]['label']

        if label == 'til-negative':
            label_id = 0
        elif label == 'til-positive':
            label_id = 1

        if self.transform is not None:
            image = self.transform(image)
        return image


def return_data(args):
    dsetname = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = 64

    if dsetname.lower() == 'chairs':
        root = os.path.join(dset_dir, 'Chairs_64')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif dsetname.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA_64')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif dsetname.lower() == 'cars':
        root = os.path.join(dset_dir, 'Cars_64')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif dsetname.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset

    elif dsetname.lower() == 'til23':
        til23_fileloader = TIL23(data_dir=dset_dir, dataframe_path=os.path.join(dset_dir, "images-tcga-tils-metadata.csv"), study_name='stad')
        train_loader = DataLoader(til23_fileloader, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        return train_loader

    else:
        raise NotImplementedError


    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader

    return data_loader
