import glob
import random
import numpy as np
import os
from tqdm import tqdm
import yaml
import cv2
import matplotlib.pyplot as plt

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def create_synth_data(config, mode = 'Train'):
    INPUT_DIR = f"{config['PROJECT_DIR']}{config['SYNTHETIC_DATA_CREATION']['INPUT_DIR']}"
    input_dir = f'{INPUT_DIR}/{mode}'
    OUTPUT_DIR = f"{config['PROJECT_DIR']}{config['SYNTHETIC_DATA_CREATION']['OUTPUT_DIR']}"
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
        class_1_data = np.load(class_1_files[img_indx])
        
        class_0_image = class_0_data[:, :, :3]
        # cv2.imwrite('background.png', cv2.cvtColor(class_0_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
        np.save(f'{class_0_dir}/{img_indx}.npy', class_0_image)

        class_1_image = class_0_image.copy()
        # canvas = np.ones(class_0_image.shape, dtype=np.uint8) * 255
        # cv2.imwrite('target.png', cv2.cvtColor(class_1_data[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
        class_1_mask = class_1_data[:, :, 3]
        class_1_image[class_1_mask > 0] = class_1_data[:, :, :3][class_1_mask > 0]
        # canvas[class_1_mask > 0] = class_1_data[:, :, :3][class_1_mask > 0]
        # cv2.imwrite('target_mask.png', cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2BGR))
        # cv2.imwrite('modified_target.png', cv2.cvtColor(class_1_image.astype(np.uint8), cv2.COLOR_RGB2BGR))

        np.save(f'{class_1_dir}/{img_indx}.npy', class_1_image)
        np.save(f'{class_1_mask_dir}/{img_indx}.npy', class_1_mask)


if __name__ == '__main__':
    mode = 'Train'
    config = read_yaml('./../../configs/consep/config_consep.yaml')
    seed_value = config['RANDOM_SEED_VALUE']
    random.seed(seed_value)
    create_synth_data(config, mode = mode)