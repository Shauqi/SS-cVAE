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
from models.cVAE import cVAE, Conv_cVAE
from models.MM_cVAE_v2 import Conv_MM_cVAE
from torch.utils.data import DataLoader
import helper
from utils import set_seeds
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import cv2
from dataloader import Til_File_Loader, Brats_File_Loader
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import yaml
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

config = read_yaml('./../configs/config_til_23.yaml')

dataset = config['dataset']
model_name = config['model_name']
version_number = config['version_number']
version_name = f'v_{version_number}'
chkpt_dir = f"{config['chkpt_dir']}/{version_name}/epoch={config['epoch_number']}.ckpt"
output_dir = config['output_dir']
background_disentanglement_penalty = config['model_parameters']['background_disentanglement_penalty']
salient_disentanglement_penalty = config['model_parameters']['salient_disentanglement_penalty']
salient_latent_size = config['model_parameters']['salient_latent_size']
background_latent_size = config['model_parameters']['background_latent_size']
train_batch_size = config['train_batch_size']
valid_batch_size = config['val_batch_size']
test_batch_size = config['test_batch_size']

logger = TensorBoardLogger(save_dir=output_dir, version=version_number, name=f'{dataset}_{model_name}')

if model_name == 'mmcvae':
    model = Conv_MM_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, salient_disentanglement_penalty = salient_disentanglement_penalty, background_disentanglement_penalty = background_disentanglement_penalty)
elif model_name == 'cvae':
    model = Conv_cVAE.load_from_checkpoint(chkpt_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if dataset == 'Brats_Synth':
    train_dir = config['train_dir']
    val_dir = config['val_dir']
    test_dir = config['test_dir']
    train_ds = Brats_File_Loader(data_dir = train_dir)
    valid_ds = Brats_File_Loader(data_dir = val_dir)
    test_ds = Brats_File_Loader(data_dir = test_dir)
elif dataset == 'TIL_23_Synth':
    train_dir = config['train_dir']
    val_dir = config['val_dir']
    test_dir = config['test_dir']
    train_ds = Til_File_Loader(data_dir = train_dir)
    valid_ds = Til_File_Loader(data_dir = val_dir)
    test_ds = Til_File_Loader(data_dir = test_dir)


train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle = False, num_workers = 4)
val_loader = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle = False, num_workers = 4)
test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle = False, num_workers = 4)

S_recon = []
Labels = []
with torch.no_grad():
    for img, label in test_loader : 
        img = img.to(device)
        z_mu, logvar_z, F_z, s_mu, logvar_s, F_s = model.encode(img)
        s_recon = model.decode_s(s_mu)
        S_recon.extend(s_recon.cpu().numpy())
        Labels.extend(label.cpu().numpy())

S_recon = np.array(S_recon)
Labels = np.array(Labels)
pred = []

for s_recon in S_recon:
    if np.sum(s_recon > 0.5) > 0:
        pred.append(1)
    else:
        pred.append(0)

pred = np.array(pred)

print(f"Salient Latent Space Accuracy: {accuracy_score(Labels, pred)}")