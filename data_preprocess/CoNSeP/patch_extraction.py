import glob
import os
from tqdm import tqdm
import numpy as np
import glob
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from visualize_data import visualize_type_mask
import yaml

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def patch_extract(config, mode = 'Train'):
    project_dir = str(config['PROJECT_DIR'])
    if mode == 'Train':
        data_dir = f"{project_dir}{config['PATCH_EXTRACTION']['TRAIN_DIR']}"
    elif mode == 'Valid':
        data_dir = f"{project_dir}{config['PATCH_EXTRACTION']['VALID_DIR']}"
    elif mode == 'Test':
        data_dir = f"{project_dir}{config['PATCH_EXTRACTION']['TEST_DIR']}"

    image_paths = glob.glob(f"{data_dir}/Images/*.png")
    label_dir = f"{data_dir}/Labels"
    out_dir = f"{project_dir}{config['PATCH_EXTRACTION']['OUTPUT_DIR']}/{mode}"
    os.makedirs(out_dir, exist_ok = True)
    tile_size = (config['PATCH_EXTRACTION']['TILE_WIDTH'], config['PATCH_EXTRACTION']['TILE_HEIGHT'])
    patch_size = (config['PATCH_EXTRACTION']['PATCH_WIDTH'], config['PATCH_EXTRACTION']['PATCH_HEIGHT'])

    classes = config['VISUALIZATION_PROPERTIES']['CLASSES']
    class_to_id = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_IDS']
    id_to_class = {value: key for key, value in class_to_id.items()}
    class_with_colors = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_COLORS']

    for image_path in tqdm(image_paths):
        base_name = os.path.basename(image_path)[:-4]
        out_dir_with_image_file_name = f'{out_dir}/{base_name}'
        os.makedirs(out_dir_with_image_file_name, exist_ok = True)
        label_path = f"{label_dir}/{base_name}.mat"
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label_data = sio.loadmat(label_path)
        type_map = label_data['type_map']
        inst_map = label_data['inst_map']
        image_resized = zoom(image, (tile_size[0]/image.shape[0], tile_size[1]/image.shape[1], 1))
        type_map_resized = zoom(type_map, (tile_size[0]/type_map.shape[0], tile_size[1]/type_map.shape[1]), order=0)
        inst_map_resized = zoom(inst_map, (tile_size[0]/inst_map.shape[0], tile_size[1]/inst_map.shape[1]), order=0)
        type_map_resized[type_map_resized == 1] = 8
        type_map_resized[type_map_resized == 2] = 1
        type_map_resized[type_map_resized == 3] = 2
        type_map_resized[type_map_resized == 4] = 2
        type_map_resized[type_map_resized == 5] = 3
        type_map_resized[type_map_resized == 6] = 3
        type_map_resized[type_map_resized == 7] = 3
        type_map_resized[type_map_resized == 8] = 3
        for y in range(0, tile_size[0], patch_size[0]):
            for x in range(0, tile_size[1], patch_size[1]):
                patch = image_resized[y:y + patch_size[0], x: x + patch_size[1]]
                type_map_patch = type_map_resized[y:y + patch_size[0], x: x + patch_size[1]]
                img_with_type_patch = np.dstack((patch, type_map_patch))
                patch_name = f'{out_dir_with_image_file_name}/{y}_{x}.npy'
                np.save(patch_name, img_with_type_patch)

        # visualize_type_mask(image_resized, type_map_resized, config['PATCH_EXTRACTION']['OUTPUT_DIR'], base_name, id_to_class, class_with_colors)

if __name__ == '__main__':
    config = read_yaml('./../../configs/consep/config_consep.yaml')
    mode = 'Train'
    patch_extract(config, mode = mode)