import glob
import random
import cv2
import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

class Paired_BIN_File_Loader():
    def __init__(self, dataset_1, dataset_2):        
        self.class_0_files = glob.glob(f"{dataset_1}/0/*.npy")
        self.class_1_files = glob.glob(f"{dataset_2}/1/*.npy")
        random.shuffle(self.class_0_files)
        random.shuffle(self.class_1_files)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.class_0_files)

    def __getitem__(self, idx):
        class_1_random_idx = random.randint(0, len(self.class_1_files) - 1)

        class_0_path = self.class_0_files[idx]
        class_0_image = np.load(class_0_path).astype(np.uint8)
        class_0_image_label = class_0_path.split('/')[-2]

        class_1_path = self.class_1_files[class_1_random_idx]
        class_1_image = np.load(class_1_path).astype(np.uint8)
        class_1_image_label = class_1_path.split('/')[-2]

        class_0_image = (class_0_image / 255).astype('float32')
        class_0_image = self.transform(class_0_image)

        class_1_image = (class_1_image / 255).astype('float32')
        class_1_image = self.transform(class_1_image)

        return class_0_image, int(class_0_image_label), class_1_image, int(class_1_image_label)

    def shuffle(self):
        random.shuffle(self.class_0_files)
        random.shuffle(self.class_1_files)

class File_Loader():
    def __init__(self, dataset_path):
        self.img_files = glob.glob(f"{dataset_path}/0/*.npy")
        self.img_files += glob.glob(f"{dataset_path}/1/*.npy")
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        image_path = self.img_files[idx]
        image = np.load(image_path).astype(np.uint8)
        label = image_path.split('/')[-2]

        image = (image / 255).astype('float32')
        image = self.transform(image)

        return image, int(label)
    

class MulticlassFileLoader():
    def __init__(self, dataset_path):
        self.img_files = glob.glob(f"{dataset_path}/*/*.npy")
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        image_path = self.img_files[idx]
        image = np.load(image_path)[:,:,:3].astype(np.uint8)
        label = image_path.split('/')[-2]

        image = (image / 255).astype('float32')
        image = self.transform(image)

        return image, int(label)
    

class MulticlassFileLoaderWithAug():
    def __init__(self, dataset_path, augmentation_path):
        self.img_files = glob.glob(f"{dataset_path}/*/*.npy")
        self.augmentation_files = glob.glob(f"{augmentation_path}/*/*.npy")
        self.img_files += self.augmentation_files
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        image_path = self.img_files[idx]
        image = np.load(image_path)[:,:,:3].astype(np.uint8)
        label = image_path.split('/')[-2]

        image = (image / 255).astype('float32')
        image = self.transform(image)

        # random_aug_idx = random.randint(0, len(self.augmentation_files) - 1)
        # aug_image_path = self.augmentation_files[random_aug_idx]
        # aug_image = np.load(aug_image_path)[:,:,:3].astype(np.uint8)
        # aug_label = aug_image_path.split('/')[-2]

        # aug_image = (aug_image / 255).astype('float32')
        # aug_image = self.transform(aug_image)

        return image, int(label) #, aug_image, int(aug_label)


class BalancedMulticlassFileLoader():
    def __init__(self, dataset_path, num_classes = 3) -> None:
        self.transform = transforms.Compose([transforms.ToTensor()])
        classes = [i for i in range(num_classes)]
        self.img_files = {}
        min_len = 100000
        for class_ in classes:
            self.img_files[class_] = glob.glob(f"{dataset_path}/{class_}/*.npy")
            min_len = min(min_len, len(self.img_files[class_]))
        
        self.balanced_img_files = []

        for class_ in classes:
            random.shuffle(self.img_files[class_])
            self.balanced_img_files += self.img_files[class_][:min_len]

    def __len__(self):
        return len(self.balanced_img_files)
    
    def __getitem__(self, idx):
        image_path = self.balanced_img_files[idx]
        image = np.load(image_path).astype(np.uint8)
        label = image_path.split('/')[-2]

        image = (image / 255).astype('float32')
        image = self.transform(image)

        return image, int(label)


if __name__ == '__main__':
    random.seed(42)
    config = read_yaml('./../configs/config_downstream_classifier_brca_consep.yaml')['PRETRAINING']
    train_dataset_path = config['train_dir']
    train_ds = BalancedMulticlassFileLoader(train_dataset_path, num_classes=2)
    train_loader = DataLoader(train_ds, batch_size = 64, shuffle = True, num_workers = 4)

    for batch in train_loader:
        image, label = batch
        print(image.shape)
        print(label)

    # test_dataset_path = config['test_dir']
    # test_ds = BalancedMulticlassFileLoader(test_dataset_path)