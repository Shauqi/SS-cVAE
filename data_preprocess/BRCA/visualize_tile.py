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

def visualize_tile(config, mode = 'Train'):
    if mode == 'Train':
        data_dir = config['PATCH_EXTRACTION']['TRAIN_DIR']
    elif mode == 'Valid':
        data_dir = config['PATCH_EXTRACTION']['VALID_DIR']
    elif mode == 'Test':
        data_dir = config['PATCH_EXTRACTION']['TEST_DIR']
    image_paths = glob.glob(f"{data_dir}/Images/TCGA-A8-A09B-01Z-00-DX1_11001_39001_1000_1000_0.93.png")
    label_dir = f"{data_dir}/Labels"
    tile_size = (config['PATCH_EXTRACTION']['TILE_WIDTH'], config['PATCH_EXTRACTION']['TILE_HEIGHT'])
    patch_size = (config['PATCH_EXTRACTION']['PATCH_WIDTH'], config['PATCH_EXTRACTION']['PATCH_HEIGHT'])

    classes = config['VISUALIZATION_PROPERTIES']['CLASSES']
    class_to_id = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_IDS']
    id_to_class = {value: key for key, value in class_to_id.items()}
    class_with_colors = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_COLORS']

    for image_path in tqdm(image_paths):
        base_name = os.path.basename(image_path)[:-4]
        label_path = f"{label_dir}/{base_name}.mat"
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label_data = sio.loadmat(label_path)
        type_map = label_data['type_map']
        inst_map = label_data['inst_map']
        print(np.unique(type_map))
        visualize_type_mask(image, type_map, config['PATCH_EXTRACTION']['OUTPUT_DIR'], base_name, id_to_class, class_with_colors)

if __name__ == '__main__':
    mode = 'Test'
    config = read_yaml('./../../configs/brca/general_config.yaml')
    visualize_tile(config, mode = mode)