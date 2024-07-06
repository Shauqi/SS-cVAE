import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from os import path
import copy
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision
from dataloader import BRCA_BIN_File_Loader
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import random

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_model(config):
    model_name = config['CVAE_MODEL_TRAIN']['model_name']
    version_number = config['CVAE_MODEL_TRAIN']['version_number']
    version_name = f'v_{version_number}'
    chkpt_dir = f"{config['CVAE_MODEL_TRAIN']['chkpt_dir']}/{version_name}/epoch={config['CVAE_MODEL_TRAIN']['epoch_number']}.ckpt"
    background_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['background_disentanglement_penalty']
    salient_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_disentanglement_penalty']
    salient_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size']
    background_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['background_latent_size']

    if model_name == 'mtl_cvae':
        from models.mtl_cvae import MTL_cVAE
        model = MTL_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, background_disentanglement_penalty = background_disentanglement_penalty, salient_disentanglement_penalty = salient_disentanglement_penalty)
    elif model_name == 'mm_cvae':
        from models.mm_cvae import MM_cVAE
        model = MM_cVAE.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'chc_vae':
        from models.chc_vae import CHC_VAE
        model = CHC_VAE.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'ch_vae':
        from models.ch_vae import CH_VAE
        model = CH_VAE.load_from_checkpoint(chkpt_dir, config = config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


config = read_yaml('./../configs/config_brca_mm_cvae.yaml')
model_name = config['CVAE_MODEL_TRAIN']['model_name']
dvae, device = get_model(config)
out_dir = config['CVAE_MODEL_TRAIN']['output_dir']
test_img_dir = path.join(out_dir, 'test')

os.makedirs(test_img_dir, exist_ok = True)

test_batch_size = 1 #config['CVAE_MODEL_TRAIN']['test_batch_size']
test_dir = config['CVAE_MODEL_TRAIN']['in_distribution']['test_dir']
test_ds = BRCA_BIN_File_Loader(test_dir)
test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers = 4)

def draw_salient_vs_background(img_list, dvae_recon_list, dvae_clean_list, dvae_dirty_list, output_dir = "", sample_name = "pos"):
    plt.figure(figsize=[100, 30])
    fig,axs =  plt.subplots(5,4)

    for i in range(5):
        axs[i][0].imshow(img_list[i])
        axs[i][0].axis('off')
        if i == 0:
            axs[i][0].set_title("Original")

        axs[i][1].imshow(dvae_recon_list[i])
        axs[i][1].axis('off')
        if i == 0:
            axs[i][1].set_title("SScVAE")
        
        axs[i][2].imshow(dvae_clean_list[i])
        axs[i][2].axis('off')
        if i == 0:
            axs[i][2].set_title("Back")

        axs[i][3].imshow(dvae_dirty_list[i])
        axs[i][3].axis('off')
        if i == 0:
            axs[i][3].set_title("Salient")

    plt.savefig(f"{output_dir}/cvae_background_and_salient_{sample_name}_sample.png")

with torch.no_grad():
    pos_img_list, pos_dvae_recon_list, pos_dvae_clean_list, pos_dvae_dirty_list = [], [], [], []
    neg_img_list, neg_dvae_recon_list, neg_dvae_clean_list, neg_dvae_dirty_list = [], [], [], []
    pos_num_of_samples = 0
    neg_num_of_samples = 0
    for img, label, mask in test_loader:
        img = img.to(device)
        mask = mask.unsqueeze(1).to(device)
        if model_name == 'mtl_cvae':
            z_mu, logvar_z, F_z, s_mu, logvar_s, F_s = dvae.encode(img)
        else:
            z_mu, z_var, s_mu, s_var = dvae.encode(img)
        z_ = torch.cat([z_mu, s_mu], dim=1)
        dvae_recon = dvae.decode_combined(z_)
        dvae_clean = dvae.decode_combined(torch.cat([z_mu, torch.zeros_like(s_mu)], dim=1))
        dvae_dirty = dvae.decode_combined(torch.cat([torch.zeros_like(z_mu), s_mu], dim=1))

        img = (img.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
        dvae_recon = (dvae_recon.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
        dvae_clean = (dvae_clean.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
        dvae_dirty = (dvae_dirty.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)

        if label == 1:
            # cv2.imwrite(f"{test_img_dir}/pos_{pos_num_of_samples}_orig.png",cv2.cvtColor(np.squeeze(img), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(f"{test_img_dir}/pos_{pos_num_of_samples}_cvae.png",cv2.cvtColor(np.squeeze(dvae_recon), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(f"{test_img_dir}/pos_{pos_num_of_samples}_gan.png",cv2.cvtColor(np.squeeze(gan_recon), cv2.COLOR_BGR2RGB))
            pos_img_list.append(np.squeeze(img))
            pos_dvae_recon_list.append(np.squeeze(dvae_recon))
            pos_dvae_clean_list.append(np.squeeze(dvae_clean))
            pos_dvae_dirty_list.append(np.squeeze(dvae_dirty))
            pos_num_of_samples += 1
        elif label == 0:
            # cv2.imwrite(f"{test_img_dir}/neg_{neg_num_of_samples}_orig.png",cv2.cvtColor(np.squeeze(img), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(f"{test_img_dir}/neg_{neg_num_of_samples}_cvae.png",cv2.cvtColor(np.squeeze(dvae_recon), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(f"{test_img_dir}/neg_{neg_num_of_samples}_gan.png",cv2.cvtColor(np.squeeze(gan_recon), cv2.COLOR_BGR2RGB))
            neg_img_list.append(np.squeeze(img))
            neg_dvae_recon_list.append(np.squeeze(dvae_recon))
            neg_dvae_clean_list.append(np.squeeze(dvae_clean))
            neg_dvae_dirty_list.append(np.squeeze(dvae_dirty))
            neg_num_of_samples += 1

        # if pos_num_of_samples == 5 and neg_num_of_samples == 5:
        #     break


pos_filtered_sample =  [23,27,34,52,59] #[1,10,12,13,14,18,46,62]
pos_img_list = [pos_img_list[index] for index in pos_filtered_sample]
pos_dvae_recon_list = [pos_dvae_recon_list[index] for index in pos_filtered_sample]
pos_dvae_clean_list = [pos_dvae_clean_list[index] for index in pos_filtered_sample]
pos_dvae_dirty_list = [pos_dvae_dirty_list[index] for index in pos_filtered_sample]

draw_salient_vs_background(pos_img_list, pos_dvae_recon_list, pos_dvae_clean_list, pos_dvae_dirty_list, output_dir = test_img_dir, sample_name = 'pos')

neg_filtered_sample = [3,8,14,17,18]
neg_img_list = [neg_img_list[index] for index in neg_filtered_sample]
neg_dvae_recon_list = [neg_dvae_recon_list[index] for index in neg_filtered_sample]
neg_dvae_clean_list = [neg_dvae_clean_list[index] for index in neg_filtered_sample]
neg_dvae_dirty_list = [neg_dvae_dirty_list[index] for index in neg_filtered_sample]

draw_salient_vs_background(neg_img_list, neg_dvae_recon_list, neg_dvae_clean_list, neg_dvae_dirty_list, output_dir = test_img_dir, sample_name = 'neg')