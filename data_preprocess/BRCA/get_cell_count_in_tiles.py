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

def get_cell_count(config, mode='Train'):
    if mode == 'Train':
        data_dir = config['PATCH_EXTRACTION']['TRAIN_DIR']
    elif mode == 'Valid':
        data_dir = config['PATCH_EXTRACTION']['VALID_DIR']
    elif mode == 'Test':
        data_dir = config['PATCH_EXTRACTION']['TEST_DIR']

    image_paths = glob.glob(f"{data_dir}/Images/*.png")
    label_dir = f"{data_dir}/Labels"

    tile_size = (config['PATCH_EXTRACTION']['TILE_WIDTH'], config['PATCH_EXTRACTION']['TILE_HEIGHT'])
    patch_size = (config['PATCH_EXTRACTION']['PATCH_WIDTH'], config['PATCH_EXTRACTION']['PATCH_HEIGHT'])

    classes = config['VISUALIZATION_PROPERTIES']['CLASSES']
    class_to_id = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_IDS']
    id_to_class = {value: key for key, value in class_to_id.items()}
    class_with_colors = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_COLORS']
    instance_of_each_type_for_all_tile = [] #{1: [], 2: [], 3: []}
    for image_path in tqdm(image_paths):
        base_name = os.path.basename(image_path)[:-4]
        label_path = f"{label_dir}/{base_name}.mat"
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label_data = sio.loadmat(label_path)
        type_map = label_data['type_map']
        inst_map = label_data['inst_map']
        image_resized = zoom(image, (tile_size[0]/image.shape[0], tile_size[1]/image.shape[1], 1))
        type_map_resized = zoom(type_map, (tile_size[0]/type_map.shape[0], tile_size[1]/type_map.shape[1]), order=0)
        inst_map_resized = zoom(inst_map, (tile_size[0]/inst_map.shape[0], tile_size[1]/inst_map.shape[1]), order=0)
        unique_instances = np.unique(inst_map_resized)[1:]
        instance_of_each_type = {1: 0, 2: 0, 3: 0, 4: 0}
        for instance in unique_instances:
            if np.unique(type_map_resized[inst_map_resized == instance])[0] == 0:
                type_map_resized[inst_map_resized == instance] = 4
            instance_of_each_type[int(np.median(type_map_resized[inst_map_resized == instance]))] += 1

        total_cells = sum(instance_of_each_type.values())
        instance_of_each_type_for_all_tile.append(total_cells)

        # for key in instance_of_each_type_for_all_tile.keys():
        #     if key != 3:
        #         instance_of_each_type_for_all_tile[key].append(instance_of_each_type[key])

        # instance_of_each_type_for_all_tile[3].append(instance_of_each_type[3] + instance_of_each_type[4])
    
    print(f"Total cells: {np.mean(instance_of_each_type_for_all_tile)}")
    print(f"Total cells: {np.std(instance_of_each_type_for_all_tile)}")

if __name__ == '__main__':
    mode = 'Test'
    config = read_yaml('./../../configs/brca/general_config.yaml')
    get_cell_count(config, mode = mode)