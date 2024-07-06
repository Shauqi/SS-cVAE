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
from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (load_config, build_models)
from gan_training.inputs import get_dataset
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision
from dataloader import BRCA_BIN_File_Loader
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt

config = load_config('./../configs/brca/resnet_cvae.yaml')

out_dir = f"{config['PROJECT_DIR']}{config['GAN_TRAIN']['OUTPUT_DIR']}"
batch_size = 64
checkpoint_dir = f"{config['PROJECT_DIR']}{config['GAN_TRAIN']['CHECKPOINT_DIR']}"
test_img_dir = path.join(out_dir, 'test')
os.makedirs(test_img_dir, exist_ok = True)

gpu_list = config['GPU_LIST']

checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")

dvae, generator, discriminator = build_models(config)
dvae = dvae.to(device)
generator = generator.to(device)
discriminator = discriminator.to(device)

checkpoint_io.register_modules(generator=generator, discriminator=discriminator,)

if config['GAN_TRAIN']['use_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Load checkpoint if existant
it = checkpoint_io.load('model.pt')
zdist = get_zdist(config['GAN_TRAIN']['z_dist']['type'], config['GAN_TRAIN']['z_dist']['dim'], device=device)

test_batch_size = 1 #config['CVAE_MODEL_TRAIN']['test_batch_size']
test_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['test_dir']}"
test_ds = BRCA_BIN_File_Loader(test_dir)
test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers = 4)

def draw_salient_vs_background(img_list, dvae_recon_list, dvae_clean_list, dvae_dirty_list, gan_recon_list, gan_clean_list, gan_dirty_list, output_dir = "", sample_name = "pos"):
    plt.figure(figsize=[100, 30])
    fig,axs =  plt.subplots(5,7)

    for i in range(5):
        axs[i][0].imshow(img_list[i])
        axs[i][0].axis('off')
        if i == 0:
            axs[i][0].set_title("Original")

        axs[i][1].imshow(dvae_recon_list[i])
        axs[i][1].axis('off')
        if i == 0:
            axs[i][1].set_title("RESVAE")
        
        axs[i][2].imshow(dvae_clean_list[i])
        axs[i][2].axis('off')
        if i == 0:
            axs[i][2].set_title("Back")

        axs[i][3].imshow(dvae_dirty_list[i])
        axs[i][3].axis('off')
        if i == 0:
            axs[i][3].set_title("Salient")

        axs[i][4].imshow(gan_recon_list[i])
        axs[i][4].axis('off')
        if i == 0:
            axs[i][4].set_title("GAN")

        axs[i][5].imshow(gan_clean_list[i])
        axs[i][5].axis('off')
        if i == 0:
            axs[i][5].set_title("Back")

        axs[i][6].imshow(gan_dirty_list[i])
        axs[i][6].axis('off')
        if i == 0:
            axs[i][6].set_title("Salient")

    plt.savefig(f"{output_dir}/cvae_and_gan_background_and_salient_{sample_name}_sample.png")

with torch.no_grad():
    pos_img_list, pos_dvae_recon_list, pos_dvae_clean_list, pos_dvae_dirty_list, pos_gan_recon_list, pos_gan_clean_list, pos_gan_dirty_list = [], [], [], [], [], [], []
    neg_img_list, neg_dvae_recon_list, neg_dvae_clean_list, neg_dvae_dirty_list, neg_gan_recon_list, neg_gan_clean_list, neg_gan_dirty_list = [], [], [], [], [], [], []
    pos_num_of_samples = 0
    neg_num_of_samples = 0
    for batch in test_loader:
        img, label, mask = batch
        # img = img.to(device)
        # mask = mask.unsqueeze(1).to(device)
        z_mu, z_var, s_mu, s_var = dvae.test_encode(batch, device)
        z_ = torch.cat([z_mu, s_mu], dim=1)
        dvae_recon = dvae.decode_combined(z_)
        dvae_clean = dvae.decode_combined(torch.cat([z_mu, torch.zeros_like(s_mu)], dim=1))
        dvae_dirty = dvae.decode_combined(torch.cat([torch.zeros_like(z_mu), s_mu], dim=1))
        gan_recon = generator_test(z_)
        gan_clean = generator_test(torch.cat([z_mu, torch.zeros_like(s_mu)], dim=1))
        gan_dirty = generator_test(torch.cat([torch.zeros_like(z_mu), s_mu], dim=1))

        img = (img.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
        dvae_recon = (dvae_recon.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
        dvae_clean = (dvae_clean.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
        dvae_dirty = (dvae_dirty.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)

        gan_recon = (gan_recon.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
        gan_clean = (gan_clean.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
        gan_dirty = (gan_dirty.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)

        if label == 1:
            # cv2.imwrite(f"{test_img_dir}/pos_{pos_num_of_samples}_orig.png",cv2.cvtColor(np.squeeze(img), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(f"{test_img_dir}/pos_{pos_num_of_samples}_cvae.png",cv2.cvtColor(np.squeeze(dvae_recon), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(f"{test_img_dir}/pos_{pos_num_of_samples}_gan.png",cv2.cvtColor(np.squeeze(gan_recon), cv2.COLOR_BGR2RGB))
            pos_img_list.append(np.squeeze(img))
            pos_dvae_recon_list.append(np.squeeze(dvae_recon))
            pos_dvae_clean_list.append(np.squeeze(dvae_clean))
            pos_dvae_dirty_list.append(np.squeeze(dvae_dirty))
            pos_gan_recon_list.append(np.squeeze(gan_recon))
            pos_gan_clean_list.append(np.squeeze(gan_clean))
            pos_gan_dirty_list.append(np.squeeze(gan_dirty))
            pos_num_of_samples += 1
        elif label == 0:
            # cv2.imwrite(f"{test_img_dir}/neg_{neg_num_of_samples}_orig.png",cv2.cvtColor(np.squeeze(img), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(f"{test_img_dir}/neg_{neg_num_of_samples}_cvae.png",cv2.cvtColor(np.squeeze(dvae_recon), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(f"{test_img_dir}/neg_{neg_num_of_samples}_gan.png",cv2.cvtColor(np.squeeze(gan_recon), cv2.COLOR_BGR2RGB))
            neg_img_list.append(np.squeeze(img))
            neg_dvae_recon_list.append(np.squeeze(dvae_recon))
            neg_dvae_clean_list.append(np.squeeze(dvae_clean))
            neg_dvae_dirty_list.append(np.squeeze(dvae_dirty))
            neg_gan_recon_list.append(np.squeeze(gan_recon))
            neg_gan_clean_list.append(np.squeeze(gan_clean))
            neg_gan_dirty_list.append(np.squeeze(gan_dirty))
            neg_num_of_samples += 1

        # if pos_num_of_samples == 5 and neg_num_of_samples == 5:
        #     break

pos_filtered_sample =  [10,11,12,13,14] #4
pos_img_list = [pos_img_list[index] for index in pos_filtered_sample]
pos_dvae_recon_list = [pos_dvae_recon_list[index] for index in pos_filtered_sample]
pos_dvae_clean_list = [pos_dvae_clean_list[index] for index in pos_filtered_sample]
pos_dvae_dirty_list = [pos_dvae_dirty_list[index] for index in pos_filtered_sample]
pos_gan_recon_list = [pos_gan_recon_list[index] for index in pos_filtered_sample]
pos_gan_clean_list = [pos_gan_clean_list[index] for index in pos_filtered_sample]
pos_gan_dirty_list = [pos_gan_dirty_list[index] for index in pos_filtered_sample]

draw_salient_vs_background(pos_img_list, pos_dvae_recon_list, pos_dvae_clean_list, pos_dvae_dirty_list, pos_gan_recon_list, pos_gan_clean_list, pos_gan_dirty_list, output_dir = test_img_dir, sample_name = 'pos')

# neg_filtered_sample = [3,8,14,17,18]
# neg_img_list = [neg_img_list[index] for index in neg_filtered_sample]
# neg_dvae_recon_list = [neg_dvae_recon_list[index] for index in neg_filtered_sample]
# neg_dvae_clean_list = [neg_dvae_clean_list[index] for index in neg_filtered_sample]
# neg_dvae_dirty_list = [neg_dvae_dirty_list[index] for index in neg_filtered_sample]
# neg_gan_recon_list = [neg_gan_recon_list[index] for index in neg_filtered_sample]
# neg_gan_clean_list = [neg_gan_clean_list[index] for index in neg_filtered_sample]
# neg_gan_dirty_list = [neg_gan_dirty_list[index] for index in neg_filtered_sample]

# draw_salient_vs_background(neg_img_list, neg_dvae_recon_list, neg_dvae_clean_list, neg_dvae_dirty_list, neg_gan_recon_list, neg_gan_clean_list, neg_gan_dirty_list, output_dir = test_img_dir, sample_name = 'neg')