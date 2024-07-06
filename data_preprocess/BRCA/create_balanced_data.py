import glob
import random
import cv2
import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import random

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def create_real_data(config, mode = 'Valid'):
    INPUT_DIR = config['REAL_DATA_CREATION']['INPUT_DIR']
    input_dir = f'{INPUT_DIR}/{mode}'
    OUTPUT_DIR = config['REAL_DATA_CREATION']['OUTPUT_DIR']
    output_dir = f'{OUTPUT_DIR}/{mode}'
    os.makedirs(output_dir, exist_ok = True)

    class_0_files = glob.glob(f'{input_dir}/0/*.npy')
    class_1_files = glob.glob(f'{input_dir}/1/*.npy')

    min_len = min(len(class_0_files), len(class_1_files))
    
    random.shuffle(class_0_files)
    random.shuffle(class_1_files)

    class_0_files = class_0_files[:min_len]
    class_1_files = class_1_files[:min_len]

    class_0_dir = f'{output_dir}/0'
    os.makedirs(class_0_dir, exist_ok = True)
    class_1_dir = f'{output_dir}/1'
    os.makedirs(class_1_dir, exist_ok = True)

    class_1_mask_dir = f'{output_dir}/1_mask'
    os.makedirs(class_1_mask_dir, exist_ok = True)

    for img_indx in tqdm(range(len(class_0_files))):
        class_0_data = np.load(class_0_files[img_indx])
        class_0_image = class_0_data[:, :, :3]
        np.save(f'{class_0_dir}/{img_indx}.npy', class_0_image)

    for img_indx in tqdm(range(len(class_1_files))):
        class_1_data = np.load(class_1_files[img_indx])
        class_1_image = class_1_data[:, :, :3]
        class_1_mask = class_1_data[:, :, 3]
        np.save(f'{class_1_dir}/{img_indx}.npy', class_1_image)
        np.save(f'{class_1_mask_dir}/{img_indx}.npy', class_1_mask)

if __name__ == '__main__':
    mode = 'Test'
    config = read_yaml('./../../configs/config_brca_ss_cvae_64_64.yaml')
    seed_value = config['RANDOM_SEED_VALUE']
    random.seed(seed_value)
    create_real_data(config, mode = mode)