import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import set_seeds

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_dataloader(config):
    dataset = config['CVAE_MODEL_TRAIN']['dataset']
    test_dir = config['CVAE_MODEL_TRAIN']['in_distribution']['test_dir']
    test_batch_size = config['CVAE_MODEL_TRAIN']['test_batch_size']

    from dataloader.brca_loader import BRCA_BIN_File_Loader
    test_ds = BRCA_BIN_File_Loader(test_dir)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle = False, num_workers = 4)
    return test_loader

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

def draw_salient_vs_background(img_list, recon_list, x_clean_list, x_dirty_list, output_dir = "", sample_name = "pos", num_rows = 4):
    plt.figure(figsize=[100, 40])
    fig,axs =  plt.subplots(num_rows,4)

    for i in range(num_rows):
        axs[i][0].imshow(img_list[i])
        axs[i][0].axis('off')
        if i == 0:
            axs[i][0].set_title("Original")

        axs[i][1].imshow(recon_list[i])
        axs[i][1].axis('off')
        if i == 0:
            axs[i][1].set_title("Reconstructed")

        axs[i][2].imshow(x_clean_list[i])
        axs[i][2].axis('off')
        if i == 0:
            axs[i][2].set_title("Background")

        axs[i][3].imshow(x_dirty_list[i])
        axs[i][3].axis('off')
        if i == 0:
            axs[i][3].set_title("Salient")

    plt.savefig(f"{output_dir}/background_and_salient_extract_{sample_name}_sample.png")

if __name__ == '__main__':
    config = read_yaml('./../configs/config_brca_mm_cvae.yaml')
    model_name = config['CVAE_MODEL_TRAIN']['model_name']
    output_dir = config['CVAE_MODEL_TRAIN']['output_dir']
    version_number = config['CVAE_MODEL_TRAIN']['version_number']
    test_loader = get_dataloader(config)
    model, device = get_model(config)

    with torch.no_grad():
        pos_img_list, pos_recon_list, pos_x_clean_list, pos_x_dirty_list = [], [], [], []
        neg_img_list, neg_recon_list, neg_x_clean_list, neg_x_dirty_list = [], [], [], []
        pos_num_of_samples = 0
        neg_num_of_samples = 0
        for img, label, mask in tqdm(test_loader):
            img = img.to(device)
            mask = mask.unsqueeze(1).to(device)
            if model_name == 'mtl_cvae':
                z_mu, logvar_z, F_z, s_mu, logvar_s, F_s = model.encode(img)
            elif model_name == 'mtl_cvae_ablation':
                z_mu, logvar_z, s_mu, logvar_s = model.model.encode(img)
            elif model_name == 'mm_cvae':
                z_mu, logvar_z, s_mu, logvar_s = model.encode(img)
            elif model_name == 'chc_vae' or model_name == 'ch_vae':
                recon, z, z_mu, z_logvar, s, s_mu, s_logvar = model.forward(img, mask)
                s_mu_flat = []
                for i in range(len(s_mu)):
                    s_mu_flat.append(s_mu[i].view(s_mu[i].size(0), -1))

                s_mu = torch.cat(s_mu_flat, dim=1)

            z_ = torch.cat([z_mu, s_mu], dim=1)
            if model_name == 'mtl_cvae_ablation' or model_name == 'chc_vae' or model_name == 'ch_vae':
                # recon = model.decode_combined(z_)
                x_clean = model.decode_combined(torch.cat([z_mu, torch.zeros_like(s_mu)], dim=1))
                x_dirty = model.decode_combined(torch.cat([torch.zeros_like(z_mu), s_mu], dim=1))
            else:
                recon = model.decode_combined(z_)
                x_clean = model.decode_combined(torch.cat([z_mu, torch.zeros_like(s_mu)], dim=1))
                x_dirty = model.decode_combined(torch.cat([torch.zeros_like(z_mu), s_mu], dim=1))
            img = img.permute(0,2,3,1).cpu().numpy()
            recon = recon.permute(0,2,3,1).cpu().numpy()
            x_clean = x_clean.permute(0,2,3,1).cpu().numpy()
            x_dirty = x_dirty.permute(0,2,3,1).cpu().numpy()

            if label == 1:
                pos_img_list.append(np.squeeze(img))
                pos_recon_list.append(np.squeeze(recon))
                pos_x_clean_list.append(np.squeeze(x_clean))
                pos_x_dirty_list.append(np.squeeze(x_dirty))
                pos_num_of_samples += 1
            elif label == 0:
                neg_img_list.append(np.squeeze(img))
                neg_recon_list.append(np.squeeze(recon))
                neg_x_clean_list.append(np.squeeze(x_clean))
                neg_x_dirty_list.append(np.squeeze(x_dirty))
                neg_num_of_samples += 1
            
            if pos_num_of_samples >= 4 and neg_num_of_samples >=4:
                break

        draw_salient_vs_background(pos_img_list[:4], pos_recon_list[:4], pos_x_clean_list[:4], pos_x_dirty_list[:4], output_dir = output_dir, sample_name = 'pos', num_rows = 4)
        draw_salient_vs_background(neg_img_list[:4], neg_recon_list[:4], neg_x_clean_list[:4], neg_x_dirty_list[:4], output_dir = output_dir, sample_name = 'neg', num_rows = 4)