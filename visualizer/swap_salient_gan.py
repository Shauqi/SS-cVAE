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
import cv2
import matplotlib.pyplot as plt
import yaml
from utils import set_seeds
from models import get_model_from_checkpoints, get_gan_model_from_checkpoints
from dataloader import get_datasets
import random

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def swap_salient_features(cvae_model, gan_model, batch, device, output_dir):
    x, labels, _ = batch
    background = x[labels == 0].to(device)
    targets = x[labels != 0].to(device)

    background = background[neg_indices]
    targets = targets[pos_indices]

    mu_z, logvar_z, mu_s, logvar_s = cvae_model.test_encode(batch, device)

    mu_z_bg = mu_z[labels == 0][neg_indices]
    mu_s_bg = mu_s[labels == 0][neg_indices]
    mu_z_t = mu_z[labels != 0][pos_indices]
    mu_s_t = mu_s[labels != 0][pos_indices]

    img_recon_bg = gan_model(torch.cat([mu_z_bg, mu_s_bg], dim=1))
    img_recon_t = gan_model(torch.cat([mu_z_t, mu_s_t], dim=1))

    salient_var_vector = torch.zeros_like(mu_s_bg)

    swap_img_zbg_st = gan_model(torch.cat([mu_z_bg, mu_s_t], dim=1))
    swap_img_zt_zeros = gan_model(torch.cat([mu_z_t, salient_var_vector], dim=1))

    img_name = f'{output_dir}/swap_salient_features_gan.png'

    reshape_background = background.permute(0,2,3,1).detach().cpu().numpy()
    reshape_targets = targets.permute(0,2,3,1).detach().cpu().numpy()
    reshape_img_recon_bg = img_recon_bg.permute(0,2,3,1).detach().cpu().numpy()
    reshape_img_recon_t = img_recon_t.permute(0,2,3,1).detach().cpu().numpy()
    reshape_swap_img_zbg_st = swap_img_zbg_st.permute(0,2,3,1).detach().cpu().numpy()
    reshape_swap_img_zt_zeros = swap_img_zt_zeros.permute(0,2,3,1).detach().cpu().numpy()

    fig, ax = plt.subplots(4*2, 3, figsize=(3, 4*2))

    for i in range(len(neg_indices)):
        ax[2*i, 0].imshow(reshape_targets[i]) 
        ax[2*i, 0].axis('off')
        ax[2*i, 1].imshow(reshape_img_recon_t[i]) 
        ax[2*i, 1].axis('off')
        ax[2*i, 2].imshow(reshape_swap_img_zt_zeros[i]) 
        ax[2*i, 2].axis('off')
        ax[2*i+1, 0].imshow(reshape_background[i])
        ax[2*i+1, 0].axis('off')
        ax[2*i+1, 1].imshow(reshape_img_recon_bg[i])
        ax[2*i+1, 1].axis('off')
        ax[2*i+1, 2].imshow(reshape_swap_img_zbg_st[i])
        ax[2*i+1, 2].axis('off')

    fig.savefig(img_name)
    plt.close(fig)


def swap_salient_features_cvae(cvae_model, gan_model, batch, device, output_dir):
    x, labels, _ = batch
    background = x[labels == 0].to(device)
    targets = x[labels != 0].to(device)

    background = background[neg_indices]
    targets = targets[pos_indices]

    mu_z, logvar_z, mu_s, logvar_s = cvae_model.test_encode(batch, device)

    mu_z_bg = mu_z[labels == 0][neg_indices]
    mu_s_bg = mu_s[labels == 0][neg_indices]
    mu_z_t = mu_z[labels != 0][pos_indices]
    mu_s_t = mu_s[labels != 0][pos_indices]

    img_recon_bg = cvae_model.decode_combined(torch.cat([mu_z_bg, mu_s_bg], dim=1))
    img_recon_t = cvae_model.decode_combined(torch.cat([mu_z_t, mu_s_t], dim=1))

    salient_var_vector = torch.zeros_like(mu_s_bg)

    swap_img_zbg_st = cvae_model.decode_combined(torch.cat([mu_z_bg, mu_s_t], dim=1))
    swap_img_zt_zeros = cvae_model.decode_combined(torch.cat([mu_z_t, salient_var_vector], dim=1))

    img_name = f'{output_dir}/swap_salient_features_cvae.png'

    reshape_background = background.permute(0,2,3,1).detach().cpu().numpy()
    reshape_targets = targets.permute(0,2,3,1).detach().cpu().numpy()
    reshape_img_recon_bg = img_recon_bg.permute(0,2,3,1).detach().cpu().numpy()
    reshape_img_recon_t = img_recon_t.permute(0,2,3,1).detach().cpu().numpy()
    reshape_swap_img_zbg_st = swap_img_zbg_st.permute(0,2,3,1).detach().cpu().numpy()
    reshape_swap_img_zt_zeros = swap_img_zt_zeros.permute(0,2,3,1).detach().cpu().numpy()

    fig, ax = plt.subplots(4*2, 3, figsize=(3, 4*2))

    for i in range(len(neg_indices)):
        ax[2*i, 0].imshow(reshape_targets[i]) 
        ax[2*i, 0].axis('off')
        ax[2*i, 1].imshow(reshape_img_recon_t[i]) 
        ax[2*i, 1].axis('off')
        ax[2*i, 2].imshow(reshape_swap_img_zt_zeros[i]) 
        ax[2*i, 2].axis('off')
        ax[2*i+1, 0].imshow(reshape_background[i])
        ax[2*i+1, 0].axis('off')
        ax[2*i+1, 1].imshow(reshape_img_recon_bg[i])
        ax[2*i+1, 1].axis('off')
        ax[2*i+1, 2].imshow(reshape_swap_img_zbg_st[i])
        ax[2*i+1, 2].axis('off')

    fig.savefig(img_name)
    plt.close(fig)    

if __name__ == "__main__":
    config = read_yaml('./../configs/brca/resnet_cvae.yaml')
    cvae_model, device = get_model_from_checkpoints(config['CVAE_MODEL_TRAIN'])
    gan_model = get_gan_model_from_checkpoints(config)
    output_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['output_dir']}"

    test_batch_size = 512 #config['CVAE_MODEL_TRAIN']['test_batch_size']
    salient_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size']
    background_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['background_latent_size']

    train_ds, val_ds, test_ds = get_datasets(config)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle = False, num_workers = 4)

    neg_indices = np.array([1,4,5,6]) # 9,8,0
    pos_indices = np.array([50,33,3,41]) # 9,16,22,25,26,31,55,57,64,67,74,45,36

    with torch.no_grad():
        for batch in test_loader:
            swap_salient_features(cvae_model, gan_model, batch, device, output_dir)
            swap_salient_features_cvae(cvae_model, gan_model, batch, device, output_dir)
            break