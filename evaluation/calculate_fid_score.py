import argparse
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import yaml
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import helper
from utils import set_seeds
import cv2
import yaml
from fid_score import calculate_fid_given_paths
from models import get_model_from_checkpoints

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_dataloader(config):
    test_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['test_dir']}"
    test_batch_size = config['CVAE_MODEL_TRAIN']['test_batch_size']

    from dataloader.brca_loader import BRCA_BIN_File_Loader
    test_ds = BRCA_BIN_File_Loader(test_dir)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle = False, num_workers = 4)
    return test_loader


if __name__ == '__main__':
    config = read_yaml('./configs/brca/ss_cvae.yaml')
    model_name = config['CVAE_MODEL_TRAIN']['model_name']

    model, device = get_model_from_checkpoints(config['CVAE_MODEL_TRAIN'])

    out_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['output_dir']}"
    fid_dir = os.path.join(out_dir, 'fid')
    os.makedirs(fid_dir, exist_ok = True)
    orig_dir = os.path.join(fid_dir, 'orig_for_fid')
    cVAE_dir = os.path.join(fid_dir, 'cVAE_for_fid')
    os.makedirs(orig_dir, exist_ok = True)
    os.makedirs(cVAE_dir, exist_ok = True)

    test_loader = get_dataloader(config)

    pos_num_of_samples = 0
    neg_num_of_samples = 0
    with torch.no_grad():
        for img, label, mask in test_loader:
            img = img.to(device)
            if model_name == 'mm_cvae' or model_name == 'cvae' or model_name == 'resnet_cvae':
                z_mu, logvar_z, s_mu, logvar_s = model.encode(img)
                recon_img = model.decode_combined(torch.cat([z_mu, s_mu], dim=1))
            elif model_name == 'double_infogan':
                _, _ , z_mu, s_mu = model.discriminator_step(img)
                recon_img = model.generator_step(torch.cat([z_mu, s_mu], dim=1))

            img = (img.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
            recon_img = (recon_img.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
            if label == 1:
                cv2.imwrite(f"{orig_dir}/pos_{pos_num_of_samples}.png",cv2.cvtColor(np.squeeze(img), cv2.COLOR_BGR2RGB))
                cv2.imwrite(f"{cVAE_dir}/pos_{pos_num_of_samples}.png",cv2.cvtColor(np.squeeze(recon_img), cv2.COLOR_BGR2RGB))
                pos_num_of_samples += 1
            elif label == 0:
                cv2.imwrite(f"{orig_dir}/neg_{neg_num_of_samples}.png",cv2.cvtColor(np.squeeze(img), cv2.COLOR_BGR2RGB))
                cv2.imwrite(f"{cVAE_dir}/neg_{neg_num_of_samples}.png",cv2.cvtColor(np.squeeze(recon_img), cv2.COLOR_BGR2RGB))
                neg_num_of_samples += 1
            

    cVAE_paths = [orig_dir, cVAE_dir]
    cuda = True
    batch_size = 4
    dims = 2048
    cVAE_fid = calculate_fid_given_paths(cVAE_paths, batch_size, cuda, dims)
    print(cVAE_fid)