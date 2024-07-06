import glob
import yaml
import random
import os
import shutil
from tqdm import tqdm

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    config = read_yaml('./../../configs/consep/config_consep.yaml')
    random.seed(config['RANDOM_SEED_VALUE'])
    mode = 'Train'
    input_dir = f"{config['PROJECT_DIR']}{config['TIL_VS_OTHER_PATCH_FILTER']['OUTPUT_DIR']}/combined"
    output_dir = f'{input_dir}/Test'
    input_dir = f'{input_dir}/{mode}'
    os.makedirs(output_dir, exist_ok = True)
    input_sub_dirs = glob.glob(f'{input_dir}/*')
    for input_sub_dir in input_sub_dirs:
        output_sub_dir = f'{output_dir}/{os.path.basename(input_sub_dir)}'
        os.makedirs(output_sub_dir, exist_ok = True)
        files = glob.glob(f'{input_sub_dir}/*')
        random.shuffle(files)
        for file in tqdm(files[:55]):
            base_name = os.path.basename(file)
            src = f'{input_sub_dir}/{base_name}'
            dst = f'{output_sub_dir}/{base_name}'
            shutil.move(src, dst)