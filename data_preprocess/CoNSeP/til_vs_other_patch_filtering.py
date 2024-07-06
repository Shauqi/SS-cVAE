import os
import numpy as np
from tqdm import tqdm
import glob
import shutil
import yaml
import cv2

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_til_count(type_mask, class_to_id):
    contours, _ = cv2.findContours((type_mask == class_to_id['LYMPHOCYTE']).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    til_count = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area >= 5:
            til_count += 1
    return til_count

def filter_patches(config, mode = 'Train'):
    INPUT_DIR = f"{config['PROJECT_DIR']}{config['TIL_VS_OTHER_PATCH_FILTER']['INPUT_DIR']}"
    OUTPUT_DIR = f"{config['PROJECT_DIR']}{config['TIL_VS_OTHER_PATCH_FILTER']['OUTPUT_DIR']}"
    VISUALIZATION_DIR = f"{config['PROJECT_DIR']}{config['TIL_VS_OTHER_PATCH_FILTER']['VISUALIZATION_DIR']}"
    data_dir = f'{INPUT_DIR}/{mode}'
    img_files = glob.glob(f'{data_dir}/*')
    output_dir = os.path.join(OUTPUT_DIR, mode)
    os.makedirs(output_dir, exist_ok = True)
    # visualization_dir = f'{VISUALIZATION_DIR}/{mode}'
    # os.makedirs(visualization_dir, exist_ok = True)

    classes = config['VISUALIZATION_PROPERTIES']['CLASSES']
    class_to_id = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_IDS']
    id_to_class = {value: key for key, value in class_to_id.items()}
    class_with_colors = config['VISUALIZATION_PROPERTIES']['CLASS_WITH_COLORS']

    for img_file in tqdm(img_files):
        img_name = os.path.basename(img_file)
        class_0_dir = os.path.join(output_dir, img_name, '0')
        os.makedirs(class_0_dir, exist_ok = True)
        class_1_dir = os.path.join(output_dir, img_name, '1')
        os.makedirs(class_1_dir, exist_ok = True)
        # visualization_dir_img = f'{visualization_dir}/{img_name}'
        # os.makedirs(visualization_dir_img, exist_ok = True)
        patch_files = glob.glob(f'{img_file}/*.npy')
        for patch_file in patch_files:
            patch_name = os.path.basename(patch_file)[:-4] 
            data = np.load(patch_file)
            image = data[:,:,0:3].astype(np.uint8)
            type_mask = data[:,:,3]
            til_count = get_til_count(type_mask, class_to_id)
            if til_count <= 2:
                dest_file_name = f'{class_0_dir}/{patch_name}.npy'
                shutil.copy(patch_file, dest_file_name)
            else:
                dest_file_name = f'{class_1_dir}/{patch_name}.npy'
                shutil.copy(patch_file, dest_file_name)
            # visualize_type_mask(image, type_mask, visualization_dir_img, patch_name, id_to_class, class_with_colors)

def combine_filtered_patches(config, mode = 'Train'):
    INPUT_DIR = f"{config['PROJECT_DIR']}{config['TIL_VS_OTHER_PATCH_FILTER']['OUTPUT_DIR']}"
    OUTPUT_DIR = f"{config['PROJECT_DIR']}{config['TIL_VS_OTHER_PATCH_FILTER']['OUTPUT_DIR']}"
    input_dir = os.path.join(INPUT_DIR, mode)
    img_files = glob.glob(f'{input_dir}/*')
    combined_output_dir = os.path.join(OUTPUT_DIR, 'combined', mode)
    class_0_dir = os.path.join(combined_output_dir, '0')
    os.makedirs(class_0_dir, exist_ok = True)
    class_1_dir = os.path.join(combined_output_dir, '1')
    os.makedirs(class_1_dir, exist_ok = True)

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


if __name__ == '__main__':
    mode = 'Train'
    config = read_yaml('./../../configs/consep/config_consep.yaml')
    filter_patches(config, mode = mode)
    combine_filtered_patches(config, mode = mode)