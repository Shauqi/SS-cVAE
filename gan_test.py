import os
import sys
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
from gan_training.metrics.fid_score import calculate_fid_given_paths
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision
from dataloader import BRCA_BIN_File_Loader
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    config_path = './configs/brca/resnet_cvae.yaml'
    config = load_config(config_path)
    gpu_list = config['GPU_LIST']

    out_dir = f"{config['PROJECT_DIR']}{config['GAN_TRAIN']['OUTPUT_DIR']}"
    checkpoint_dir = f"{config['PROJECT_DIR']}{config['GAN_TRAIN']['CHECKPOINT_DIR']}"
    test_img_dir = path.join(out_dir, 'test')
    orig_dir = path.join(test_img_dir, 'orig')
    cVAE_dir = path.join(test_img_dir, 'cVAE')
    GAN_dir = path.join(test_img_dir, 'GAN')

    os.makedirs(test_img_dir, exist_ok = True)
    os.makedirs(orig_dir, exist_ok = True)
    os.makedirs(cVAE_dir, exist_ok = True)
    os.makedirs(GAN_dir, exist_ok = True)

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

    test_batch_size = config['GAN_TRAIN']['test_batch_size']

    test_dir = f"{config['PROJECT_DIR']}{config['GAN_TRAIN']['test_dir']}"

    test_ds = BRCA_BIN_File_Loader(test_dir)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle = False, num_workers = 4)

    with torch.no_grad():
        pos_img_list, pos_dvae_recon_list, pos_gan_recon_list = [], [], []
        neg_img_list, neg_dvae_recon_list, neg_gan_recon_list = [], [], []
        pos_num_of_samples = 0
        neg_num_of_samples = 0
        for img, label, mask in test_loader:
            img = img.to(device)
            mask = mask.unsqueeze(1).to(device)
            s, s_mu, s_logvar, z, z_mu, z_logvar = dvae.gan_encode(img)
            z_ = torch.cat([z, s], dim=1)
            dvae_recon = dvae.decode_combined(z_)
            gan_recon = generator_test(z_)
            img = (img.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
            dvae_recon = (dvae_recon.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
            gan_recon = (gan_recon.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)

            if label == 1:
                cv2.imwrite(f"{orig_dir}/pos_{pos_num_of_samples}.png",cv2.cvtColor(np.squeeze(img), cv2.COLOR_BGR2RGB))
                cv2.imwrite(f"{cVAE_dir}/pos_{pos_num_of_samples}.png",cv2.cvtColor(np.squeeze(dvae_recon), cv2.COLOR_BGR2RGB))
                cv2.imwrite(f"{GAN_dir}/pos_{pos_num_of_samples}.png",cv2.cvtColor(np.squeeze(gan_recon), cv2.COLOR_BGR2RGB))
                pos_num_of_samples += 1
            elif label == 0:
                cv2.imwrite(f"{orig_dir}/neg_{neg_num_of_samples}.png",cv2.cvtColor(np.squeeze(img), cv2.COLOR_BGR2RGB))
                cv2.imwrite(f"{cVAE_dir}/neg_{neg_num_of_samples}.png",cv2.cvtColor(np.squeeze(dvae_recon), cv2.COLOR_BGR2RGB))
                cv2.imwrite(f"{GAN_dir}/neg_{neg_num_of_samples}.png",cv2.cvtColor(np.squeeze(gan_recon), cv2.COLOR_BGR2RGB))
                neg_num_of_samples += 1


    ###### Frechet Inception Distance (FID) ######
    cVAE_paths = [orig_dir, cVAE_dir]
    GAN_paths = [orig_dir, GAN_dir]
    cuda = True
    batch_size = 4
    dims = 2048
    cVAE_fid = calculate_fid_given_paths(cVAE_paths, batch_size, cuda, dims)
    GAN_fid = calculate_fid_given_paths(GAN_paths, batch_size, cuda, dims)
    print(cVAE_fid, GAN_fid)


    ####### Inception Score ########

    # # Distributions
    # ydist = get_ydist(nlabels, device=device)
    # zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
    #                   device=device)
    # cdist = get_zdist('gauss', config['dvae']['c_dim'], device=device)

    # # Evaluator
    # evaluator = Evaluator(generator_test, zdist, ydist, batch_size=batch_size, device=device)

    # # Inception score
    # if config['test']['compute_inception']:
    #     print('Computing inception score...')
    #     inception_mean, inception_std = evaluator.compute_inception_score()
    #     print('Inception score: %.4f +- %.4f' % (inception_mean, inception_std))