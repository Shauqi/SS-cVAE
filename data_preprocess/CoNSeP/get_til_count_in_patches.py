import glob
import os
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from visualize_data import visualize_type_mask
import yaml

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_til_count(resnet_config, mode='Train'):
    if mode == 'Train':
        data_dir = resnet_config['CONTRASTIVE_MODEL_TRAIN']['train_dir']
    elif mode == 'Valid':
        data_dir = resnet_config['CONTRASTIVE_MODEL_TRAIN']['val_dir']
    elif mode == 'Test':
        data_dir = resnet_config['CONTRASTIVE_MODEL_TRAIN']['test_dir']

    class_1_files = glob.glob(f"{resnet_config['PROJECT_DIR']}{data_dir}/1/*.npy")

    total_til_over_all_patches = []
    for file in tqdm(class_1_files):
        data = np.load(file)
        image = data[:,:,:3]
        mask = data[:,:,3]
        til_mask = mask == 1
        contours, _ = cv2.findContours((til_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_til_over_all_patches.append(len(contours))

    print(f"Total TILs mean in patches: {np.mean(total_til_over_all_patches)}")
    print(f"Total TILs std in patches: {np.std(total_til_over_all_patches)}")


if __name__ == '__main__':
    mode = 'Test'
    config = read_yaml('./../../configs/consep/config_consep.yaml')
    resnet_config = read_yaml('./../../configs/consep/resnet_cvae.yaml')
    classes = config['VISUALIZATION_PROPERTIES']['CLASSES']
    class_to_id = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_IDS']
    id_to_class = {value: key for key, value in class_to_id.items()}
    class_with_colors = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_COLORS']
    get_til_count(resnet_config, mode = mode)