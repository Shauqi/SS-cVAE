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

def get_cell_count(resnet_config, mode='Train'):
    if mode == 'Train':
        data_dir = resnet_config['CVAE_MODEL_TRAIN']['train_dir']
    elif mode == 'Valid':
        data_dir = resnet_config['CVAE_MODEL_TRAIN']['val_dir']
    elif mode == 'Test':
        data_dir = resnet_config['CVAE_MODEL_TRAIN']['test_dir']

    class_1_files = glob.glob(f"{resnet_config['PROJECT_DIR']}{data_dir}/1/*.npy")

    total_cells_over_all_patches = []
    for file in tqdm(class_1_files):
        base_name = os.path.basename(file)
        mask_file = f"{resnet_config['PROJECT_DIR']}{data_dir}/1_mask/{base_name}"
        mask = np.load(mask_file)
        type_freq = {class_name:0 for class_name in classes}
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_instances = 0
        for i, contour in enumerate(contours):
            class_id = mask[contour[0][0][1], contour[0][0][0]]
            class_name = id_to_class[class_id]
            area = cv2.contourArea(contour)
            if area > 5:
                type_freq[class_name] += 1
                total_instances += 1
        total_cells = sum(type_freq.values())
        total_cells_over_all_patches.append(total_cells)

    print(f"Total cells: {np.mean(total_cells_over_all_patches)}")
    print(f"Total cells: {np.std(total_cells_over_all_patches)}")

if __name__ == '__main__':
    mode = 'Test'
    config = read_yaml('./../../configs/brca/general_config.yaml')
    resnet_config = read_yaml('./../../configs/brca/resnet_cvae.yaml')
    classes = config['VISUALIZATION_PROPERTIES']['CLASSES']
    class_to_id = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_IDS']
    id_to_class = {value: key for key, value in class_to_id.items()}
    class_with_colors = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_COLORS']
    get_cell_count(resnet_config, mode = mode)