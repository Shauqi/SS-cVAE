import os
from typing import Any
import numpy as np
from torchvision import transforms
import glob
import random
import cv2
import yaml
import torch
random.seed(42)

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

class BRCA_BIN_File_Loader():
    def __init__(self, data_dir, shuffle=True):
        self.data_dir = data_dir
        
        class_0_files = glob.glob(f"{self.data_dir}/0/*.npy")
        class_0_labels = [0 for i in range(len(class_0_files))]

        class_1_files = glob.glob(f"{self.data_dir}/1/*.npy")
        class_1_labels = [1 for i in range(len(class_1_files))]
 
        all_files = class_0_files + class_1_files
        all_labels = class_0_labels + class_1_labels
        temp = list(zip(all_files, all_labels))
        if shuffle:
            random.shuffle(temp)
        all_files, all_labels = zip(*temp)
        self.all_files, self.all_labels = list(all_files), list(all_labels)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        image_path = self.all_files[idx]
        
        image = np.load(image_path).astype(np.uint8)
        label = self.all_labels[idx]

        image_label = image_path.split('/')[-2]
        image_basename = os.path.basename(image_path)
        if image_label == '0':
            mask = np.zeros((image.shape[0], image.shape[1])).astype('float32')
        else:
            image_mask_path = f"{self.data_dir}/{image_label}_mask/{image_basename}"
            mask = np.load(image_mask_path)
            # mask = (mask > 0) * 1.0
            # mask[mask > 1] = 2
            mask[mask > 0] = 1
            mask = mask.astype('float32')

        image = (image / 255).astype('float32')
        image = self.transform(image)
        return image, label, mask

    def shuffle(self):
        temp = list(zip(self.all_files, self.all_labels))
        random.shuffle(temp)
        all_files, all_labels = zip(*temp)
        self.all_files, self.all_labels = list(all_files), list(all_labels)

class BRCA_GAN_File_Loader():
    def __init__(self, data_dir, shuffle=True):
        self.data_dir = data_dir
        
        # class_0_files = glob.glob(f"{self.data_dir}/0/*.npy")
        # class_0_labels = [0 for i in range(len(class_0_files))]

        class_1_files = glob.glob(f"{self.data_dir}/1/*.npy")
        class_1_labels = [1 for i in range(len(class_1_files))]
 
        all_files = class_1_files
        all_labels = class_1_labels
        temp = list(zip(all_files, all_labels))
        random.shuffle(temp)
        all_files, all_labels = zip(*temp)
        self.all_files, self.all_labels = list(all_files), list(all_labels)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        image_path = self.all_files[idx]
        
        image = np.load(image_path).astype(np.uint8)
        label = self.all_labels[idx]

        image_label = image_path.split('/')[-2]
        image_basename = os.path.basename(image_path)
        if image_label == '0':
            mask = np.zeros((image.shape[0], image.shape[1])).astype('float32')
        else:
            image_mask_path = f"{self.data_dir}/{image_label}_mask/{image_basename}"
            mask = np.load(image_mask_path)
            mask[mask > 1] = 2
            mask = mask.astype('float32')

        image = (image / 255).astype('float32')
        image = self.transform(image)
        return image, label, mask

    def shuffle(self):
        temp = list(zip(self.all_files, self.all_labels))
        random.shuffle(temp)
        all_files, all_labels = zip(*temp)
        self.all_files, self.all_labels = list(all_files), list(all_labels)

def get_til_count(type_mask):
    contours, _ = cv2.findContours((type_mask == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    til_count = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area >= 5:
            til_count += 1
    return til_count

class BRCA_BIN_to_TIL_vs_Other_File_Loader():
    def __init__(self, data_dir, shuffle=True):
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
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        image_path = self.all_files[idx]
        
        image = np.load(image_path).astype(np.uint8)
        # label = self.all_labels[idx]
        til_density_label = 0

        image_label = image_path.split('/')[-2]
        image_basename = os.path.basename(image_path)
        if image_label == '0':
            mask = np.zeros((image.shape[0], image.shape[1])).astype('float32')
        else:
            image_mask_path = f"{self.data_dir}/{image_label}_mask/{image_basename}"
            mask = np.load(image_mask_path)
            mask[mask > 1] = 2
            til_counts = get_til_count(mask)
            if til_counts > 5:
                til_density_label = 1
            mask = mask.astype('float32')

        image = (image / 255).astype('float32')
        image = self.transform(image)
        return image, til_density_label, mask

    def shuffle(self):
        temp = list(zip(self.all_files, self.all_labels))
        random.shuffle(temp)
        all_files, all_labels = zip(*temp)
        self.all_files, self.all_labels = list(all_files), list(all_labels)

class BRCA_TIL_VS_Other_File_Loader():
    def __init__(self, data_dir, shuffle=True):
        self.data_dir = data_dir
        
        class_0_files = glob.glob(f"{self.data_dir}/0/*.npy")
        random.shuffle(class_0_files)
        class_1_files = glob.glob(f"{self.data_dir}/1/*.npy")
        random.shuffle(class_1_files)

        min_len = min(len(class_0_files), len(class_1_files))
        class_0_files = class_0_files[:min_len]
        class_1_files = class_1_files[:min_len]

        class_0_labels = [0 for i in range(len(class_0_files))]
        class_1_labels = [1 for i in range(len(class_1_files))]
 
        all_files = class_0_files + class_1_files
        all_labels = class_0_labels + class_1_labels
        temp = list(zip(all_files, all_labels))
        random.shuffle(temp)
        all_files, all_labels = zip(*temp)
        self.all_files, self.all_labels = list(all_files), list(all_labels)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        data_path = self.all_files[idx]
        
        data = np.load(data_path).astype(np.uint8)
        label = self.all_labels[idx]

        image = data[:, :, :3]
        image = (image / 255).astype('float32')
        image = self.transform(image)
        mask = data[:, :, 3]
        mask[mask > 1] == 2
        mask = mask.astype('float32')

        return image, label, mask
    
    def shuffle(self):
        temp = list(zip(self.all_files, self.all_labels))
        random.shuffle(temp)
        all_files, all_labels = zip(*temp)
        self.all_files, self.all_labels = list(all_files), list(all_labels)

class BRCA_BIN_Paired_File_Loader():
    def __init__(self, data_dir, shuffle=True):
        self.data_dir = data_dir
        self.class_0_files = glob.glob(f"{self.data_dir}/0/*.npy")
        self.class_1_dir = f"{self.data_dir}/1"
        self.class_1_mask_dir = f"{self.data_dir}/1_mask"
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.class_0_files)
    
    def __getitem__(self, idx):
        class_0_file = self.class_0_files[idx]
        class_0_base_name = os.path.basename(class_0_file)
        class_1_file = f"{self.class_1_dir}/{class_0_base_name}"
        class_1_mask_file = f"{self.class_1_mask_dir}/{class_0_base_name}"
        class_0_image = np.load(class_0_file).astype(np.uint8)
        class_1_image = np.load(class_1_file).astype(np.uint8)
        class_0_mask = np.zeros((class_0_image.shape[0], class_0_image.shape[1])).astype('float32')
        class_1_mask = np.load(class_1_mask_file).astype('float32')
        
        class_0_image = (class_0_image / 255).astype('float32')
        class_1_image = (class_1_image / 255).astype('float32')
        class_0_image = self.transform(class_0_image)
        class_1_image = self.transform(class_1_image)

        return class_0_image, class_0_mask, class_1_image, class_1_mask
    
    def shuffle(self):
        random.shuffle(self.class_0_files)

class BRCA_MTL2BIN_Paired_File_Loader():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_size = 64
        self.class_0_files = glob.glob(f"{self.data_dir}/0/*.npy")
        self.class_1_dir = f"{self.data_dir}/1"
        self.class_1_mask_dir = f"{self.data_dir}/1_mask"
        self.class_2_dir = f"{self.data_dir}/2"
        self.class_2_mask_dir = f"{self.data_dir}/2_mask"
        self.class_3_dir = f"{self.data_dir}/3"
        self.class_3_mask_dir = f"{self.data_dir}/3_mask"
        self.class_4_dir = f"{self.data_dir}/4"
        self.class_4_mask_dir = f"{self.data_dir}/4_mask"
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.class_0_files)

    def __getitem__(self, idx):
        class_0_path = self.class_0_files[idx]
        class_0_base_name = os.path.basename(class_0_path)

        random_int = random.randint(1, 4)
        neg_image = np.load(class_0_path).astype(np.uint8)
        neg_image = cv2.resize(neg_image, (self.img_size, self.img_size))
        neg_image = (neg_image / 255).astype('float32') 
        neg_image = self.transform(neg_image)

        if random_int == 1:
            pos_image = np.load(f"{self.class_1_dir}/{class_0_base_name}").astype(np.uint8)
            pos_image = cv2.resize(pos_image, (self.img_size, self.img_size))
            pos_image = (pos_image / 255).astype('float32')
            pos_image = self.transform(pos_image)

            pos_mask = (np.load(f"{self.class_1_mask_dir}/{class_0_base_name}") > 0) * 1.0
            pos_mask = cv2.resize(pos_mask, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR).astype('float32')
        elif random_int == 2:
            pos_image = np.load(f"{self.class_2_dir}/{class_0_base_name}").astype(np.uint8)
            pos_image = cv2.resize(pos_image, (self.img_size, self.img_size))
            pos_image = (pos_image / 255).astype('float32')
            pos_image = self.transform(pos_image)

            pos_mask = (np.load(f"{self.class_2_mask_dir}/{class_0_base_name}") > 0) * 1.0
            pos_mask = cv2.resize(pos_mask, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR).astype('float32')
        elif random_int == 3:
            pos_image = np.load(f"{self.class_3_dir}/{class_0_base_name}").astype(np.uint8)
            pos_image = cv2.resize(pos_image, (self.img_size, self.img_size))
            pos_image = (pos_image / 255).astype('float32')
            pos_image = self.transform(pos_image)

            pos_mask = (np.load(f"{self.class_3_mask_dir}/{class_0_base_name}") > 0) * 1.0
            pos_mask = cv2.resize(pos_mask, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR).astype('float32')
        elif random_int == 4:
            pos_image = np.load(f"{self.class_4_dir}/{class_0_base_name}").astype(np.uint8)
            pos_image = cv2.resize(pos_image, (self.img_size, self.img_size))
            pos_image = (pos_image / 255).astype('float32')
            pos_image = self.transform(pos_image)

            pos_mask = (np.load(f"{self.class_4_mask_dir}/{class_0_base_name}") > 0) * 1.0
            pos_mask = cv2.resize(pos_mask, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR).astype('float32')

        return pos_image, neg_image, pos_mask

    def shuffle(self):
        random.shuffle(self.class_0_files)

class BRCA_MTL_File_Loader():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_size = 64
        
        class_0_files = glob.glob(f"{self.data_dir}/0/*.npy")
        class_0_labels = [0 for i in range(len(class_0_files))]

        class_1_files = glob.glob(f"{self.data_dir}/1/*.npy")
        class_1_labels = [1 for i in range(len(class_1_files))]

        class_2_files = glob.glob(f"{self.data_dir}/2/*.npy")
        class_2_labels = [2 for i in range(len(class_2_files))]

        all_files = class_0_files + class_1_files + class_2_files
        all_labels = class_0_labels + class_1_labels + class_2_labels
        temp = list(zip(all_files, all_labels))
        random.shuffle(temp)
        all_files, all_labels = zip(*temp)
        self.all_files, self.all_labels = list(all_files), list(all_labels)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        data_path = self.all_files[idx]
        data = np.load(data_path).astype(np.uint8)
        label = self.all_labels[idx]

        image = data[:, :, :3]
        image = (image / 255).astype('float32')
        image = self.transform(image)
        mask = data[:, :, 3]
        mask[mask > 1] == 2
        mask = mask.astype('float32')

        return image, label, mask

    def shuffle(self):
        temp = list(zip(self.all_files, self.all_labels))
        random.shuffle(temp)
        all_files, all_labels = zip(*temp)
        self.all_files, self.all_labels = list(all_files), list(all_labels)

class BRCA_MTL2BIN_File_Loader():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_size = 64
        
        class_0_files = glob.glob(f"{self.data_dir}/0/*.npy")
        class_0_labels = [0 for i in range(len(class_0_files))]

        class_1_files = glob.glob(f"{self.data_dir}/1/*.npy")
        random.shuffle(class_1_files)
        class_1_files = class_1_files[:int(0.25 * len(class_1_files))]
        class_1_labels = [1 for i in range(len(class_1_files))]
 
        class_2_files = glob.glob(f"{self.data_dir}/2/*.npy")
        random.shuffle(class_2_files)
        class_2_files = class_2_files[:int(0.25 * len(class_2_files))]
        class_2_labels = [1 for i in range(len(class_2_files))]

        class_3_files = glob.glob(f"{self.data_dir}/3/*.npy")
        random.shuffle(class_3_files)
        class_3_files = class_3_files[:int(0.25 * len(class_3_files))]
        class_3_labels = [1 for i in range(len(class_3_files))]

        class_4_files = glob.glob(f"{self.data_dir}/4/*.npy")
        random.shuffle(class_4_files)
        class_4_files = class_4_files[:int(0.25 * len(class_4_files))]
        class_4_labels = [1 for i in range(len(class_4_files))]

        all_files = class_0_files + class_1_files + class_2_files + class_3_files + class_4_files
        all_labels = class_0_labels + class_1_labels + class_2_labels + class_3_labels + class_4_labels
        temp = list(zip(all_files, all_labels))
        random.shuffle(temp)
        all_files, all_labels = zip(*temp)
        self.all_files, self.all_labels = list(all_files), list(all_labels)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        image_path = self.all_files[idx]
        
        image = np.load(image_path).astype(np.uint8)
        label = self.all_labels[idx]

        image_label = image_path.split('/')[-2]
        image_basename = os.path.basename(image_path)
        if image_label == '0':
            mask = np.zeros((self.img_size, self.img_size)).astype('float32')
        else:
            image_mask_path = f"{self.data_dir}/{image_label}_mask/{image_basename}"
            mask = np.load(image_mask_path)
            mask = (mask > 0) * 1.0
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR).astype('float32')

        image = cv2.resize(image, (self.img_size, self.img_size))
        image = (image / 255).astype('float32')

        image = self.transform(image)
        return image, label, mask

    def shuffle(self):
        temp = list(zip(self.all_files, self.all_labels))
        random.shuffle(temp)
        all_files, all_labels = zip(*temp)
        self.all_files, self.all_labels = list(all_files), list(all_labels)

if __name__ == '__main__':
    config = read_yaml('./configs/brca/ss_cvae_one_stage_ablation.yaml')
    train_dir = config['CVAE_MODEL_TRAIN']['PROJECT_DIR'] + config['CVAE_MODEL_TRAIN']['train_dir']

    # train_dataset = BRCA_BIN_File_Loader(train_dir)
    train_dataset = BRCA_MTL_File_Loader(train_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    for image, label, mask in train_loader:
        print(image.shape, label.shape, mask.shape)
        print(label)
        break

    # test_dir = config['CVAE_MODEL_TRAIN']['train_dir']
    # test_dataset = BRCA_BIN_Paired_File_Loader(test_dir)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)
    # for class_0_image, class_0_mask, class_1_image, class_1_mask in test_loader:
    #     print(class_0_image.shape, class_0_mask.shape, class_1_image.shape, class_1_mask.shape)