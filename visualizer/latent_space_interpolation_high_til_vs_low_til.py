import argparse
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import yaml
import copy
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import helper
from utils import set_seeds
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from models import get_model_from_checkpoints, get_gan_model_from_checkpoints, get_contrastive_model_from_checkpoints
from gan_training.checkpoints import CheckpointIO
from gan_training.config import (load_config, build_models)
from dataloader import get_datasets, get_til_vs_other_datasets
import cv2
from umap import UMAP

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    
def draw_latent_features_with_class(tsne_feature_space, label, output_dir, fig_name, dot_size = 5, pos_index_1 = 0, neg_index_1 = 0, pos_index_2 = 0, neg_index_2 = 0, arrow_head_width = 0.1, arrow_head_length = 0.05):
    neg_features = tsne_feature_space[label == 0]
    pos_features = tsne_feature_space[label == 1]
    pos_to_neg_interpolated_features = tsne_feature_space[label == 3]
    neg_to_pos_interpolated_features = tsne_feature_space[label == 4]
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(neg_features[:,0], neg_features[:,1], c = 'brown', s = dot_size, label = 'low til')
    plt.scatter(pos_features[:,0], pos_features[:,1], c = 'green', s = dot_size, label = 'high til')
    plt.arrow(tsne_feature_space[pos_index_1, 0], tsne_feature_space[pos_index_1, 1], pos_to_neg_interpolated_features[1, 0] - tsne_feature_space[pos_index_1, 0], pos_to_neg_interpolated_features[1, 1] - tsne_feature_space[pos_index_1, 1], color='olive', length_includes_head=True, head_width=arrow_head_width, head_length=arrow_head_length)
    for i in range(1, pos_to_neg_interpolated_features.shape[0] - 2):
        plt.arrow(pos_to_neg_interpolated_features[i, 0], pos_to_neg_interpolated_features[i, 1], pos_to_neg_interpolated_features[i+1, 0] - pos_to_neg_interpolated_features[i, 0], pos_to_neg_interpolated_features[i+1, 1] - pos_to_neg_interpolated_features[i, 1], color='olive', length_includes_head=True, head_width=arrow_head_width, head_length=arrow_head_length)
    plt.arrow(pos_to_neg_interpolated_features[-2, 0], pos_to_neg_interpolated_features[-2, 1], tsne_feature_space[neg_index_1, 0] - pos_to_neg_interpolated_features[-2, 0], tsne_feature_space[neg_index_1, 1] - pos_to_neg_interpolated_features[-2, 1], color='olive', length_includes_head=True, head_width=arrow_head_width, head_length=arrow_head_length)
    plt.scatter(tsne_feature_space[pos_index_1, 0], tsne_feature_space[pos_index_1, 1], c = 'None', edgecolors='black', s = dot_size*10)
    plt.scatter(tsne_feature_space[neg_index_1, 0], tsne_feature_space[neg_index_1, 1], c = 'None', edgecolors='black', s = dot_size*10)

    plt.arrow(tsne_feature_space[neg_index_2, 0], tsne_feature_space[neg_index_2, 1], neg_to_pos_interpolated_features[1, 0] - tsne_feature_space[neg_index_2, 0], neg_to_pos_interpolated_features[1, 1] - tsne_feature_space[neg_index_2, 1], color='pink', length_includes_head=True, head_width=arrow_head_width, head_length=arrow_head_length)
    for i in range(1, neg_to_pos_interpolated_features.shape[0] - 2):
        plt.arrow(neg_to_pos_interpolated_features[i, 0], neg_to_pos_interpolated_features[i, 1], neg_to_pos_interpolated_features[i+1, 0] - neg_to_pos_interpolated_features[i, 0], neg_to_pos_interpolated_features[i+1, 1] - neg_to_pos_interpolated_features[i, 1], color='pink', length_includes_head=True, head_width=arrow_head_width, head_length=arrow_head_length)

    plt.arrow(neg_to_pos_interpolated_features[-2, 0], neg_to_pos_interpolated_features[-2, 1], tsne_feature_space[pos_index_2, 0] - neg_to_pos_interpolated_features[-2, 0], tsne_feature_space[pos_index_2, 1] - neg_to_pos_interpolated_features[-2, 1], color='pink', length_includes_head=True, head_width=arrow_head_width, head_length=arrow_head_length)
    plt.scatter(tsne_feature_space[pos_index_2, 0], tsne_feature_space[pos_index_2, 1], c = 'None', edgecolors='black', s = dot_size*10)
    plt.scatter(tsne_feature_space[neg_index_2, 0], tsne_feature_space[neg_index_2, 1], c = 'None', edgecolors='black', s = dot_size*10)

    plt.title('Latent Space Visualization')
    plt.xlabel('latent component 1')
    plt.ylabel('latent component 2')
    plt.legend()
    plt.savefig(f"{output_dir}/{fig_name}.png")

def interpolate_latent(latent_1, latent_2, n_steps = 4):
    interpolated_latent = [latent_1]
    for i in range(1, n_steps):
        interpolated_latent.append(latent_1 + (latent_2 - latent_1) * i / n_steps)
    interpolated_latent.append(latent_2)
    return interpolated_latent

def draw_interpolated_latent_with_gan(gan_model, salient_interpolated_latents, background_interpolated_latents, orig_pos_image, orig_neg_image, output_dir, fig_name):
    fig, axs =  plt.subplots(1, len(salient_interpolated_latents)+2, figsize=(10, 10))
    axs[0].imshow(orig_pos_image)
    axs[0].axis('off')

    for i, salient_latent in enumerate(salient_interpolated_latents):
        salient_latent = salient_latent.unsqueeze(0)
        background_latent = background_interpolated_latents[i].unsqueeze(0)
        z_ = torch.cat((background_latent, salient_latent), dim = 1)
        img = gan_model(z_)
        img = (img.permute(0,2,3,1).squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
        axs[i+1].imshow(img)
        axs[i+1].axis('off')

    axs[-1].imshow(orig_neg_image)
    axs[-1].axis('off')

    plt.savefig(f"{output_dir}/{fig_name}.png")
    plt.close(fig)


def draw_interpolated_latent_with_cvae(cvae_model, salient_interpolated_latents, background_interpolated_latents, orig_pos_image, orig_neg_image, output_dir, fig_name):
    fig, axs =  plt.subplots(1, len(salient_interpolated_latents)+2, figsize=(10, 10))
    axs[0].imshow(orig_pos_image)
    axs[0].axis('off')

    for i, salient_latent in enumerate(salient_interpolated_latents):
        salient_latent = salient_latent.unsqueeze(0)
        background_latent = background_interpolated_latents[i].unsqueeze(0)
        z_ = torch.cat((background_latent, salient_latent), dim = 1)
        img = cvae_model.decode_combined(z_)
        img = (img.permute(0,2,3,1).squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
        axs[i+1].imshow(img)
        axs[i+1].axis('off')

    axs[-1].imshow(orig_neg_image)
    axs[-1].axis('off')

    plt.savefig(f"{output_dir}/{fig_name}.png")
    plt.close(fig)


if __name__ == "__main__":
    config = read_yaml('./../configs/brca/resnet_cvae.yaml')
    cvae_model, device = get_model_from_checkpoints(config['CVAE_MODEL_TRAIN'])
    # gan_model = get_gan_model_from_checkpoints(config)
    # contrastive_model, device = get_contrastive_model_from_checkpoints(config)
    output_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['output_dir']}"

    test_batch_size = config['CVAE_MODEL_TRAIN']['test_batch_size']
    salient_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size']
    background_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['background_latent_size']

    train_ds, val_ds, test_ds = get_til_vs_other_datasets(config)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle = False, num_workers = 4)

    S_MU_arr = []
    Z_MU_arr = []
    S_arr = np.empty((0, salient_latent_size))
    Z_arr = np.empty((0, background_latent_size))
    label_arr = np.empty((0))
    image_arr = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            img, label, mask = batch
            z_mu, z_log_var, s_mu, s_log_var = cvae_model.test_encode(batch, device) #contrastive_model.test_encode(batch, device) 
            Z_MU_arr += z_mu
            S_MU_arr += s_mu
            Z_arr = np.vstack((Z_arr, np.squeeze(z_mu.cpu().numpy())))
            S_arr = np.vstack((S_arr, np.squeeze(s_mu.cpu().numpy())))
            label_arr = np.hstack((label_arr, np.squeeze(label.numpy())))
            image_arr += img
            # if index == 10:
            #     break

    pos_index_1 = np.where(label_arr == 1)[0][1] #
    neg_index_1 = np.where(label_arr == 0)[0][33] # 31, 18, 22, 11, 30, 29

    neg_index_2 = np.where(label_arr == 0)[0][11] # 18, 13, 15
    pos_index_2 = np.where(label_arr == 1)[0][8] #

    pos_s_mu_1 = S_MU_arr[pos_index_1]
    neg_s_mu_1 = S_MU_arr[neg_index_1]

    pos_z_mu_1 = Z_MU_arr[pos_index_1]
    neg_z_mu_1 = Z_MU_arr[neg_index_1]

    pos_s_mu_2 = S_MU_arr[pos_index_2]
    neg_s_mu_2 = S_MU_arr[neg_index_2]

    pos_z_mu_2 = Z_MU_arr[pos_index_2]
    neg_z_mu_2 = Z_MU_arr[neg_index_2]

    orig_pos_image_1 = (image_arr[pos_index_1].squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    orig_neg_image_1 = (image_arr[neg_index_1].squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

    orig_pos_image_2 = (image_arr[pos_index_2].squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    orig_neg_image_2 = (image_arr[neg_index_2].squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

    # pos_to_neg_interpolated_latents = interpolate_latent(pos_s_mu_1, neg_s_mu_1, n_steps = 4)
    # pos_to_neg_background_interpolated_latents = interpolate_latent(pos_z_mu_1, neg_z_mu_1, n_steps = 4)

    # neg_to_pos_interpolated_latents = interpolate_latent(neg_s_mu_2, pos_s_mu_2, n_steps = 4)
    # neg_to_pos_background_interpolated_latents = interpolate_latent(neg_z_mu_2, pos_z_mu_2, n_steps = 4)

    neg_to_pos_interpolated_latents = interpolate_latent(neg_s_mu_1, pos_s_mu_1, n_steps = 4)
    neg_to_pos_background_interpolated_latents = interpolate_latent(neg_z_mu_1, pos_z_mu_1, n_steps = 4)

    # draw_interpolated_latent_with_cvae(cvae_model, pos_to_neg_interpolated_latents, pos_to_neg_background_interpolated_latents, orig_pos_image_1, orig_neg_image_1, output_dir, "interpolated_latent_cvae_high_to_low")

    # draw_interpolated_latent_with_cvae(cvae_model, neg_to_pos_interpolated_latents, neg_to_pos_background_interpolated_latents, orig_neg_image_2, orig_pos_image_2, output_dir, "interpolated_latent_cvae_low_to_high")

    draw_interpolated_latent_with_cvae(cvae_model, neg_to_pos_interpolated_latents, neg_to_pos_background_interpolated_latents, orig_neg_image_1, orig_pos_image_1, output_dir, "interpolated_latent_cvae_high_to_low")

    # draw_interpolated_latent_with_gan(gan_model, pos_to_neg_interpolated_latents, pos_to_neg_background_interpolated_latents, orig_pos_image_1, orig_neg_image_1, output_dir, "interpolated_latent_gan_high_to_low")

    # draw_interpolated_latent_with_gan(gan_model, neg_to_pos_interpolated_latents, neg_to_pos_background_interpolated_latents, orig_neg_image_2, orig_pos_image_2, output_dir, "interpolated_latent_gan_low_to_high")

    # for i, interpolated_latent in enumerate(pos_to_neg_interpolated_latents):
    #     S_arr = np.vstack((S_arr, np.squeeze(interpolated_latent.cpu().numpy())))
    #     label_arr = np.hstack((label_arr, np.squeeze(np.array([3]))))

    # for i, interpolated_latent in enumerate(neg_to_pos_interpolated_latents):
    #     S_arr = np.vstack((S_arr, np.squeeze(interpolated_latent.cpu().numpy())))
    #     label_arr = np.hstack((label_arr, np.squeeze(np.array([4]))))

    # fig_mode = 'UMAP'

    # if fig_mode == 'TSNE':
    #     tsne_salient = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    #     tsne_salient_feature_space = tsne_salient.fit_transform(S_arr)
    #     tsne_salient_feature_space = np.array(tsne_salient_feature_space)

    #     fig_name = "salient_latent_space_high_til_vs_low_til"
    #     draw_latent_features_with_class(tsne_salient_feature_space, label_arr, output_dir, fig_name, dot_size = 50, pos_index_1 = pos_index_1, neg_index_1 = neg_index_1, pos_index_2 = pos_index_2, neg_index_2 = neg_index_2, arrow_head_width = 0.3, arrow_head_length = 0.3)
    # else:        
    #     umap_salient = UMAP(n_components=2, verbose=1, n_neighbors=15, min_dist=0.1, metric='euclidean')
    #     umap_salient_feature_space = umap_salient.fit_transform(S_arr)
    #     umap_salient_feature_space = np.array(umap_salient_feature_space)

    #     fig_name = "salient_latent_space_high_til_vs_low_til"
    #     draw_latent_features_with_class(umap_salient_feature_space, label_arr, output_dir, fig_name, dot_size = 50, pos_index_1 = pos_index_1, neg_index_1 = neg_index_1, pos_index_2 = pos_index_2, neg_index_2 = neg_index_2, arrow_head_width = 0.1, arrow_head_length = 0.1)

    # # np.save(f"{output_dir}/salient_latent_space.npy", filtered_salient_latent_space)
    # # np.save(f"{output_dir}/salient_label_arr.npy", filtered_salient_label_arr)

    # # filtered_salient_latent_space = np.load(f"{output_dir}/salient_latent_space.npy")
    # # filtered_salient_label_arr = np.load(f"{output_dir}/salient_label_arr.npy")