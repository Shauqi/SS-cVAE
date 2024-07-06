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
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import cv2
from dataloader import Til_File_Loader, Brats_File_Loader, BRCA_File_Loader
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import yaml
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from fid_score import calculate_fid_given_paths

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    config = read_yaml('./../configs/config_consep.yaml')

    dataset = config['MODEL_TRAIN']['dataset']
    model_name = config['MODEL_TRAIN']['model_name']
    version_number = config['MODEL_TRAIN']['version_number']
    version_name = f'v_{version_number}'
    chkpt_dir = f"{config['MODEL_TRAIN']['chkpt_dir']}/{version_name}/epoch={config['MODEL_TRAIN']['epoch_number']}.ckpt"
    output_dir = f"{config['MODEL_TRAIN']['output_dir']}/{dataset}_{model_name}"
    os.makedirs(output_dir, exist_ok = True)
    output_dir = f"{output_dir}/{version_name}"
    os.makedirs(output_dir, exist_ok = True)
    orig_dir = os.path.join(output_dir, 'orig_for_fid')
    cVAE_dir = os.path.join(output_dir, 'cVAE_for_fid')
    os.makedirs(orig_dir, exist_ok = True)
    os.makedirs(cVAE_dir, exist_ok = True)
    background_disentanglement_penalty = config['MODEL_TRAIN']['model_parameters']['background_disentanglement_penalty']
    salient_disentanglement_penalty = config['MODEL_TRAIN']['model_parameters']['salient_disentanglement_penalty']
    salient_latent_size = config['MODEL_TRAIN']['model_parameters']['salient_latent_size']
    background_latent_size = config['MODEL_TRAIN']['model_parameters']['background_latent_size']
    train_batch_size = config['MODEL_TRAIN']['train_batch_size']
    valid_batch_size = config['MODEL_TRAIN']['val_batch_size']
    test_batch_size = config['MODEL_TRAIN']['test_batch_size']

    if model_name == 'guided_mmcvae':
        from models.Guided_MM_cVAE import Conv_MM_cVAE
        model = Conv_MM_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, salient_disentanglement_penalty = salient_disentanglement_penalty, background_disentanglement_penalty = background_disentanglement_penalty)
    elif model_name == 'mmcvae_original':
        from models.MM_cVAE import Conv_MM_cVAE
        model = Conv_MM_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, salient_disentanglement_penalty = salient_disentanglement_penalty, background_disentanglement_penalty = background_disentanglement_penalty)
    elif model_name == 'mmcvae_paired':
        from models.MM_cVAE_Paired import Conv_MM_cVAE
        model = Conv_MM_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, salient_disentanglement_penalty = salient_disentanglement_penalty, background_disentanglement_penalty = background_disentanglement_penalty)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_dir = config['MODEL_TRAIN']['test_dir']

    if dataset == 'Brats_Synth':
        test_ds = Brats_File_Loader(data_dir = test_dir)
    elif dataset == 'TIL_23_Synth':
        test_ds = Til_File_Loader(data_dir = test_dir)
    elif dataset == 'BRCA_Synth' or dataset == 'CoNSeP_Synth':
        test_ds = BRCA_File_Loader(test_dir)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle = False, num_workers = 4)
    pos_num_of_samples = 0
    neg_num_of_samples = 0
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            if model_name == 'guided_mmcvae' or model_name == 'mmcvae_paired':
                z_mu, logvar_z, F_z, s_mu, logvar_s, F_s = model.encode(img)
                recon_img = model.decode_combined(torch.cat([z_mu, s_mu], dim=1))
            elif model_name == 'mmcvae_original':
                z_mu, logvar_z, s_mu, logvar_s = model.encode(img)
                recon_img = model.decode(torch.cat([z_mu, s_mu], dim=1))

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