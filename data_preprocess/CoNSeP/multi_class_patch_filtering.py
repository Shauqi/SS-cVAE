import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import glob
import shutil
import yaml
from visualize_data import visualize_type_mask
import cv2

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_cell_and_background_percentage(type_mask, classes, id_to_class):
    type_freq = {class_name:0 for class_name in classes}

    background_percentage = 0
    patch_area = type_mask.shape[0] * type_mask.shape[1]

    contours, _ = cv2.findContours((type_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_instances = 0
    for i, contour in enumerate(contours):
        class_id = type_mask[contour[0][0][1], contour[0][0][0]]
        class_name = id_to_class[class_id]
        area = cv2.contourArea(contour)
        if area > 20:
            type_freq[class_name] += 1
            total_instances += 1
    
    type_percentage = {class_name:0 for class_name in classes}
    if total_instances != 0:
        type_percentage = {class_name: type_freq[class_name]/total_instances for class_name in classes}
    else:
        type_percentage['BACKGROUND'] = 1.0

    unique_types, type_freqs = np.unique(type_mask, return_counts = True)
    for unique_type, type_freq in zip(unique_types, type_freqs):
        if unique_type == 0:
            background_percentage += type_freq / patch_area

    if background_percentage >= 0.99:
        type_percentage['BACKGROUND'] = 1.0

    return type_percentage

def filter_patches(config, mode = 'Train'):
    INPUT_DIR = config['MULTI_CLASS_PATCH_FILTER']['INPUT_DIR']
    OUTPUT_DIR = config['MULTI_CLASS_PATCH_FILTER']['OUTPUT_DIR']
    VISUALIZATION_DIR = config['MULTI_CLASS_PATCH_FILTER']['VISUALIZATION_DIR']
    data_dir = f'{INPUT_DIR}/{mode}'
    img_files = glob.glob(f'{data_dir}/*')
    output_dir = f'{OUTPUT_DIR}/{mode}'
    os.makedirs(output_dir, exist_ok = True)
    visualization_dir = f'{VISUALIZATION_DIR}/{mode}'
    os.makedirs(visualization_dir, exist_ok = True)

    classes = config['VISUALIZATION_PROPERTIES']['CLASSES']
    class_to_id = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_IDS']
    id_to_class = {value: key for key, value in class_to_id.items()}
    class_with_colors = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_COLORS']

    for img_file in tqdm(img_files):
        img_name = os.path.basename(img_file)
        output_dir_img = f'{output_dir}/{img_name}'
        os.makedirs(output_dir_img, exist_ok = True)
        class_0_dir = f'{output_dir_img}/0'
        os.makedirs(class_0_dir, exist_ok = True)
        class_1_dir = f'{output_dir_img}/1'
        os.makedirs(class_1_dir, exist_ok = True)
        class_2_dir = f'{output_dir_img}/2'
        os.makedirs(class_2_dir, exist_ok = True)
        # class_3_dir = f'{output_dir_img}/3'
        # os.makedirs(class_3_dir, exist_ok = True)
        # visualization_dir_img = f'{visualization_dir}/{img_name}'
        # os.makedirs(visualization_dir_img, exist_ok = True)
        patch_files = glob.glob(f'{img_file}/*.npy')
        for patch_file in patch_files:
            patch_name = os.path.basename(patch_file)[:-4] 
            data = np.load(patch_file)
            image = data[:,:,0:3].astype(np.uint8)
            type_mask = data[:,:,3]
            type_percentage = get_cell_and_background_percentage(type_mask, classes, id_to_class)
            if type_percentage['BACKGROUND'] >= 0.95:
                dest_file_name = f'{class_0_dir}/{patch_name}.npy'
                shutil.copy(patch_file, dest_file_name)
            elif type_percentage['LYMPHOCYTE'] >= 0.20:
                dest_file_name = f'{class_1_dir}/{patch_name}.npy'
                shutil.copy(patch_file, dest_file_name)
            # elif type_percentage['TUMOR'] >= 0.55:
            #     dest_file_name = f'{class_2_dir}/{patch_name}.npy'
            #     shutil.copy(patch_file, dest_file_name)
            else:
                dest_file_name = f'{class_2_dir}/{patch_name}.npy'
                shutil.copy(patch_file, dest_file_name)
            # visualize_type_mask(image, type_mask, visualization_dir_img, patch_name, id_to_class, class_with_colors)

def combine_filtered_patches(config, mode = 'Train'):
    INPUT_DIR = config['MULTI_CLASS_PATCH_FILTER']['OUTPUT_DIR']
    OUTPUT_DIR = config['MULTI_CLASS_PATCH_FILTER']['OUTPUT_DIR']
    input_dir = f'{INPUT_DIR}/{mode}'
    img_files = glob.glob(f'{input_dir}/*')
    combined_output_dir = f'{OUTPUT_DIR}/combined'
    os.makedirs(combined_output_dir, exist_ok = True)
    combined_output_dir = f'{combined_output_dir}/{mode}'
    class_0_dir = f'{combined_output_dir}/0'
    os.makedirs(class_0_dir, exist_ok = True)
    class_1_dir = f'{combined_output_dir}/1'
    os.makedirs(class_1_dir, exist_ok = True)
    class_2_dir = f'{combined_output_dir}/2'
    os.makedirs(class_2_dir, exist_ok = True)
    # class_3_dir = f'{combined_output_dir}/3'
    # os.makedirs(class_3_dir, exist_ok = True)

    for img_file in tqdm(img_files):
        img_name = os.path.basename(img_file)

        class_0_patch_files = glob.glob(f'{img_file}/0/*.npy')
        for src_file_name in class_0_patch_files:
            base_name = os.path.basename(src_file_name)
            dst_file_name = f'{class_0_dir}/{img_name}_{base_name}'
            shutil.copy(src_file_name, dst_file_name)

        class_1_patch_files = glob.glob(f'{img_file}/1/*.npy')
        for src_file_name in class_1_patch_files:
            base_name = os.path.basename(src_file_name)
            dst_file_name = f'{class_1_dir}/{img_name}_{base_name}'
            shutil.copy(src_file_name, dst_file_name)

        class_2_patch_files = glob.glob(f'{img_file}/2/*.npy')
        for src_file_name in class_2_patch_files:
            base_name = os.path.basename(src_file_name)
            dst_file_name = f'{class_2_dir}/{img_name}_{base_name}'
            shutil.copy(src_file_name, dst_file_name)

        # class_3_patch_files = glob.glob(f'{img_file}/3/*.npy')
        # for src_file_name in class_3_patch_files:
        #     base_name = os.path.basename(src_file_name)
        #     dst_file_name = f'{class_3_dir}/{img_name}_{base_name}'
        #     shutil.copy(src_file_name, dst_file_name)


if __name__ == '__main__':
    mode = 'Train'
    config = read_yaml('./../../configs/config_consep.yaml')
    filter_patches(config, mode = mode)
    combine_filtered_patches(config, mode = mode)