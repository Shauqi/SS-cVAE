import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from itertools import chain
from utils import mmd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from torch.utils.data import Dataset, DataLoader
from .Activation_Map import CAMS
import wandb

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, dec_channels, bias):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, dec_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm1 = nn.BatchNorm2d(dec_channels)
        self.conv2 = nn.Conv2d(dec_channels, dec_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm2 = nn.BatchNorm2d(dec_channels*2)
        self.conv3 = nn.Conv2d(dec_channels * 2, dec_channels * 4, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm3 = nn.BatchNorm2d(dec_channels * 4)
        self.conv4 = nn.Conv2d(dec_channels * 4, dec_channels * 8, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm4 = nn.BatchNorm2d(dec_channels * 8)
        self.gradients = None
    
    def forward(self, x):
        F = []
        x = self.conv1(x)
        F.append(x)
        x = self.relu1(x)
        F.append(x)
        x = self.batch_norm1(x)
        F.append(x)
        x = self.conv2(x)
        F.append(x)
        x = self.relu2(x)
        F.append(x)
        x = self.batch_norm2(x)
        F.append(x)
        x = self.conv3(x)
        F.append(x)
        x = self.relu3(x)
        F.append(x)
        x = self.batch_norm3(x)
        F.append(x)
        x = self.conv4(x)
        F.append(x)
        x = self.relu4(x)
        F.append(x)
        x = self.batch_norm4(x)
        F.append(x)
        return x, F

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, dec_channels, bias):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(dec_channels * 8, dec_channels * 4, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm1 = nn.BatchNorm2d(dec_channels * 4)
        self.conv2 = nn.ConvTranspose2d(dec_channels * 4, dec_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm2 = nn.BatchNorm2d(dec_channels * 2)
        self.conv3 = nn.ConvTranspose2d(dec_channels * 2, dec_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm3 = nn.BatchNorm2d(dec_channels)
        self.conv4 = nn.ConvTranspose2d(dec_channels, in_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batch_norm2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batch_norm3(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x

class Conv_MM_cVAE(pl.LightningModule):
    def __init__(self, salient_latent_size = None, background_latent_size = None, background_disentanglement_penalty = None, salient_disentanglement_penalty = None, save_path='', batch_test=None, save_img_epoch=100, train_ds = None, valid_ds = None, test_ds = None, train_batch_size = 1, valid_batch_size = 22, test_batch_size = 30):
        super(Conv_MM_cVAE, self).__init__()
        self.save_img_path = save_path + "imgs/"
        in_channels = 3
        dec_channels = 32
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.salient_latent_size = salient_latent_size
        self.background_latent_size = background_latent_size
        bias = False
        self.batch_test = batch_test
        self.save_img_epoch = save_img_epoch
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.CAMS = CAMS()

        self.z_convs = Encoder(self.in_channels, self.dec_channels, bias)
        self.s_convs = Encoder(self.in_channels, self.dec_channels, bias)
        self.z_mu = nn.Linear(dec_channels * 8 * 4 * 4, self.background_latent_size, bias=bias)
        self.z_var = nn.Linear(dec_channels * 8 * 4 * 4, self.background_latent_size, bias=bias)
        self.s_mu = nn.Linear(dec_channels * 8 * 4 * 4, self.salient_latent_size, bias=bias)
        self.s_var = nn.Linear(dec_channels * 8 * 4 * 4, self.salient_latent_size, bias=bias)

        self.decode_convs = Decoder(self.in_channels, self.dec_channels, bias)

        total_latent_size = self.salient_latent_size + self.background_latent_size

        self.background_disentanglement_penalty = background_disentanglement_penalty
        self.salient_disentanglement_penalty = salient_disentanglement_penalty

        self.d_fc_1 = nn.Linear(total_latent_size, dec_channels * 8 * 4 * 4)

    def reparameterize(self, mu, log_var):
        #:param mu: mean from the encoder's latent space
        #:param log_var: log variance from the encoder's latent space
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def encode(self, x):
        hz, Fz = self.z_convs(x)
        hs, Fs = self.s_convs(x)

        hz = hz.view(-1, self.dec_channels * 8 * 4 * 4)
        hs = hs.view(-1, self.dec_channels * 8 * 4 * 4)

        return self.z_mu(hz), self.z_var(hz), Fz, self.s_mu(hs), self.s_var(hs), Fs

    def decode(self, z):
        z = F.leaky_relu(self.d_fc_1(z), negative_slope=0.2)
        z = z.view(-1, self.dec_channels * 8, 4, 4)
        return self.decode_convs(z)

    def forward_target(self, x):
        mu_z, logvar_z, Fz, mu_s, logvar_s, Fs = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        return self.decode(torch.cat([z, s], dim=1)), mu_z, logvar_z, mu_s, logvar_s, z, s, Fz, Fs

    def forward_background(self, x):
        mu_z, logvar_z, Fz, mu_s, logvar_s, Fs = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        salient_var_vector = torch.zeros(x.shape[0], self.salient_latent_size).to(self.device)
        return self.decode(torch.cat([z, salient_var_vector], dim=1)), mu_z, logvar_z, mu_s, logvar_s, z, s, Fz, Fs

    def embed_shared(self, x):
        mu_z, _, _, _ = self.encode(x)
        return mu_z

    def embed_salient(self, x):
        _, _, mu_s, _ = self.encode(x)
        return mu_s

    def forward(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        return self.decode(torch.cat([z, s], dim=1)), mu_z, logvar_z, mu_s, logvar_s

    def save_swapped_image(self, batch_test):
        x, labels = batch_test
        background = x[labels == 0].to(self.device)
        targets = x[labels != 0].to(self.device)

        min_index_len = min(len(background), len(targets))
        background = background[:min_index_len]
        targets = targets[:min_index_len]

        mu_z_bg, _, _, mu_s_bg, _, _ = self.encode(background)
        mu_z_t, _, _, mu_s_t, _, _ = self.encode(targets)

        img_recon_bg = self.decode(torch.cat([mu_z_bg, mu_s_bg], dim=1))
        img_recon_t = self.decode(torch.cat([mu_z_t, mu_s_t], dim=1))

        salient_var_vector = torch.zeros_like(mu_s_bg)

        swap_img_zbg_st = self.decode(torch.cat([mu_z_bg, mu_s_t], dim=1))
        swap_img_zt_zeros = self.decode(torch.cat([mu_z_t, salient_var_vector], dim=1))

        img_name = self.save_img_path + 'val_epochs_' + str(self.current_epoch) + '_img_swap.png'

        reshape_background = background.permute(0,2,3,1).detach().cpu().numpy()
        reshape_targets = targets.permute(0,2,3,1).detach().cpu().numpy()
        reshape_img_recon_bg = img_recon_bg.permute(0,2,3,1).detach().cpu().numpy()
        reshape_img_recon_t = img_recon_t.permute(0,2,3,1).detach().cpu().numpy()
        reshape_swap_img_zbg_st = swap_img_zbg_st.permute(0,2,3,1).detach().cpu().numpy()
        reshape_swap_img_zt_zeros = swap_img_zt_zeros.permute(0,2,3,1).detach().cpu().numpy()

        img_to_save2 = np.zeros((64*6,64*min_index_len,self.in_channels))
        for i in range(min_index_len): 
            img_to_save2[0:64,64*i:64*(i+1),:] = reshape_background[i]
            img_to_save2[64:128,64*i:64*(i+1),:] = reshape_targets[i]
            img_to_save2[128:192,64*i:64*(i+1),:] = reshape_img_recon_bg[i]
            img_to_save2[192:256,64*i:64*(i+1),:] = reshape_img_recon_t[i]
            img_to_save2[256:320,64*i:64*(i+1),:] = reshape_swap_img_zbg_st[i]
            img_to_save2[320:384,64*i:64*(i+1),:] = reshape_swap_img_zt_zeros[i]

        plt.imsave(img_name, img_to_save2)

    def get_cam_heatmap(self, batch):
        with torch.set_grad_enabled(True):
            x, labels = batch
            background = x[labels == 0].to(self.device).requires_grad_()
            targets = x[labels != 0].to(self.device).requires_grad_()

            min_index_len = min(len(background), len(targets))
            background = background[:min_index_len]
            targets = targets[:min_index_len]

            mu_z_bg, logvar_z_bg, F_z_bg, mu_s_bg, logvar_s_bg, F_s_bg = self.encode(background)
            mu_z_tar, logvar_z_tar, F_z_tar, mu_s_tar, logvar_s_tar, F_s_tar = self.encode(targets)

            backgrounds = background.permute(0,2,3,1).detach().cpu().numpy()
            targets = targets.permute(0,2,3,1).detach().cpu().numpy()


            fig, axs = plt.subplots(min_index_len * 2, 2 * 4 + 2, figsize=(2 * 5 * 10, min_index_len * 10))
            fig.tight_layout(pad = 0.0)

            img_index = 0
            for row in range(0, min_index_len * 2, 2):
                axs[row, 0].imshow(backgrounds[img_index])
                axs[row, 0].axis('off')
                axs[row + 1, 0].imshow(np.ones_like(backgrounds[img_index]) * 255)
                axs[row + 1, 0].axis('off')
                axs[row, 5].imshow(targets[img_index])
                axs[row, 5].axis('off')
                axs[row + 1, 5].imshow(np.ones_like(targets[img_index]) * 255)
                axs[row + 1, 5].axis('off')
                img_index += 1

            col = 1

            for activation_index in [2, 5, 8, 9]:
                gcam_z_bg = self.CAMS.cam(F_z_bg[activation_index], normalization = 'sigm')
                gcam_z_bg = torch.nn.functional.interpolate(gcam_z_bg.unsqueeze(1), size=(64, 64), mode='bilinear', align_corners=True).squeeze()

                gcam_s_bg = self.CAMS.cam(F_s_bg[activation_index], normalization = 'sigm')
                gcam_s_bg = torch.nn.functional.interpolate(gcam_s_bg.unsqueeze(1), size=(64, 64), mode='bilinear', align_corners=True).squeeze()

                gcam_z_tar = self.CAMS.cam(F_z_tar[activation_index], normalization = 'sigm')
                gcam_z_tar = torch.nn.functional.interpolate(gcam_z_tar.unsqueeze(1), size=(64, 64), mode='bilinear', align_corners=True).squeeze()

                gcam_s_tar = self.CAMS.cam(F_s_tar[activation_index], normalization = 'sigm')
                gcam_s_tar = torch.nn.functional.interpolate(gcam_s_tar.unsqueeze(1), size=(64, 64), mode='bilinear', align_corners=True).squeeze()

                gcam_z_bg = gcam_z_bg.detach().cpu().numpy()
                gcam_s_bg = gcam_s_bg.detach().cpu().numpy()

                gcam_z_tar = gcam_z_tar.detach().cpu().numpy()
                gcam_s_tar = gcam_s_tar.detach().cpu().numpy()


                img_index = 0
                for row in range(0, min_index_len * 2, 2):
                    axs[row, col].imshow(gcam_z_bg[img_index], cmap = 'inferno')
                    axs[row, col].axis('off')

                    axs[row + 1, col].imshow(gcam_s_bg[img_index], cmap = 'inferno')
                    axs[row + 1, col].axis('off')
                    img_index += 1

                img_index = 0
                for row in range(0, min_index_len * 2, 2):
                    axs[row, col + 5].imshow(gcam_z_tar[img_index], cmap = 'inferno')
                    axs[row, col + 5].axis('off')

                    axs[row + 1, col + 5].imshow(gcam_s_tar[img_index], cmap = 'inferno')
                    axs[row + 1, col + 5].axis('off')
                    img_index += 1
                col += 1

            img_name = self.save_img_path + 'val_epochs_' + str(self.current_epoch) + '_grad_cam.png'
            fig.savefig(img_name)

    def training_step(self, batch, batch_idx):
        targets, background, mask = batch
        no_mask = torch.zeros_like(mask)
        recon_batch_bg, mu_z_bg, logvar_z_bg, mu_s_bg, logvar_s_bg, z_bg, s_bg, F_z_bg, F_s_bg = self.forward_background(background)
        recon_batch_tar, mu_z_tar, logvar_z_tar, mu_s_tar, logvar_s_tar, z_tar, s_tar, F_z_tar, F_s_tar = self.forward_target(targets)

        gcam_z_bg = self.CAMS.cam(F_z_bg[5], normalization = 'sigm')
        gcam_z_bg = torch.nn.functional.interpolate(gcam_z_bg.unsqueeze(1), size=(64, 64), mode='bilinear', align_corners=True).squeeze(dim = 0)

        gcam_s_bg = self.CAMS.cam(F_s_bg[5], normalization = 'sigm')
        gcam_s_bg = torch.nn.functional.interpolate(gcam_s_bg.unsqueeze(1), size=(64, 64), mode='bilinear', align_corners=True).squeeze(dim = 0)

        gcam_z_tar = self.CAMS.cam(F_z_tar[5], normalization = 'sigm')
        gcam_z_tar = torch.nn.functional.interpolate(gcam_z_tar.unsqueeze(1), size=(64, 64), mode='bilinear', align_corners=True).squeeze(dim = 0)

        gcam_s_tar = self.CAMS.cam(F_s_tar[5], normalization = 'sigm')
        gcam_s_tar = torch.nn.functional.interpolate(gcam_s_tar.unsqueeze(1), size=(64, 64), mode='bilinear', align_corners=True).squeeze(dim = 0)

        MSE_bg = F.mse_loss(recon_batch_bg, background, reduction='sum')
        MSE_tar = F.mse_loss(recon_batch_tar, targets, reduction='sum')

        KLD_z_bg = -0.5 * torch.sum(1 + logvar_z_bg - mu_z_bg.pow(2) - logvar_z_bg.exp())
        KLD_z_tar = -0.5 * torch.sum(1 + logvar_z_tar - mu_z_tar.pow(2) - logvar_z_tar.exp())
        KLD_s_tar = -0.5 * torch.sum(1 + logvar_s_tar - mu_s_tar.pow(2) - logvar_s_tar.exp())

        # MSE_gcam_s_tar = F.mse_loss(gcam_s_tar, mask, reduction='sum')
        # MSE_gcam_s_bg = F.mse_loss(gcam_s_bg, no_mask, reduction='sum')
        MSE_gcam_z_tar_gcam_z_bg = F.mse_loss(gcam_z_tar, gcam_z_bg, reduction='sum')
        MSE_gcam_s_tar = F.binary_cross_entropy(gcam_s_tar, mask, reduction='sum')
        MSE_gcam_s_bg = F.binary_cross_entropy(gcam_s_bg, no_mask, reduction='sum')

        loss = (MSE_bg + KLD_z_bg) + (MSE_tar + KLD_z_tar + KLD_s_tar)
        loss += MSE_gcam_s_tar + MSE_gcam_s_bg + MSE_gcam_z_tar_gcam_z_bg

        gammas = torch.FloatTensor([10 ** x for x in range(-6, 7, 1)])
        background_mmd_loss = self.background_disentanglement_penalty * mmd(z_bg, z_tar, gammas=gammas, device=self.device)
        salient_mmd_loss = self.salient_disentanglement_penalty * mmd(s_bg, torch.zeros_like(s_bg), gammas=gammas, device=self.device)
        loss += background_mmd_loss + salient_mmd_loss

        return {'loss': loss, 'mse_bg': MSE_bg, 'mse_tar': MSE_tar, 'kld_z_bg': KLD_z_bg, 'kld_z_tar': KLD_z_tar, 'kld_s_tar': KLD_s_tar, 'background_mmd_loss': background_mmd_loss, 'salient_mmd_loss': salient_mmd_loss}


    def training_epoch_end(self, outputs):
        train_loss = sum(output['loss'] for output in outputs) / (len(outputs) * self.train_batch_size)
        mse_bg = sum(output['mse_bg'] for output in outputs) / (len(outputs) * self.train_batch_size)
        mse_tar = sum(output['mse_tar'] for output in outputs) / (len(outputs) * self.train_batch_size)
        kld_z_bg = sum(output['kld_z_bg'] for output in outputs) / (len(outputs) * self.train_batch_size)
        kld_z_tar = sum(output['kld_z_tar'] for output in outputs) / (len(outputs) * self.train_batch_size)
        kld_s_tar = sum(output['kld_s_tar'] for output in outputs) / (len(outputs) * self.train_batch_size)
        salient_mmd_loss = sum(output['salient_mmd_loss'] for output in outputs) / (len(outputs) * self.train_batch_size)
        background_mmd_loss = sum(output['background_mmd_loss'] for output in outputs) / (len(outputs) * self.train_batch_size)
        self.log('train_loss', train_loss, prog_bar=True)
        self.log('mse_bg', mse_bg, prog_bar=True)
        self.log('mse_tar', mse_tar, prog_bar=True)
        self.log('kld_z_bg', kld_z_bg, prog_bar=True)
        self.log('kld_z_tar', kld_z_tar, prog_bar=True)
        self.log('kld_s_tar', kld_s_tar, prog_bar=True)
        self.log('salient_mmd_loss', salient_mmd_loss, prog_bar=True)
        self.log('background_mmd_loss', salient_mmd_loss, prog_bar=True)
        self.train_ds.shuffle()


    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            os.makedirs(self.save_img_path, exist_ok=True)
            self.save_swapped_image(batch)
            self.get_cam_heatmap(batch)
        x, labels = batch
        z_mu, _, _, s_mu, _, _ = self.encode(x)
        z_mu = z_mu.cpu().numpy()
        s_mu = s_mu.cpu().numpy()
        labels = labels.cpu().numpy()
        ss_z = silhouette_score(z_mu, labels)
        ss_s = silhouette_score(s_mu, labels)
        return {"val_ss_z": ss_z, "val_ss_s": ss_s}

    def validation_epoch_end(self, outputs):
        ss_z = sum(output['val_ss_z'] for output in outputs) / (len(outputs))
        ss_s = sum(output['val_ss_s'] for output in outputs) / (len(outputs))
        self.log("val_ss_z", ss_z, prog_bar=True)
        self.log("val_ss_s", ss_s, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, labels = batch
        z_mu, _, _, s_mu, _, _ = self.encode(x)
        z_mu = z_mu.cpu().numpy()
        s_mu = s_mu.cpu().numpy()
        labels = labels.cpu().numpy()
        ss_z = silhouette_score(z_mu, labels)
        ss_s = silhouette_score(s_mu, labels)
        return {"test_ss_z": ss_z, "test_ss_s": ss_s}

    def test_epoch_end(self, outputs):
        ss_z = sum(output['test_ss_z'] for output in outputs) / (len(outputs))
        ss_s = sum(output['test_ss_s'] for output in outputs) / (len(outputs))
        self.log("test_ss_z", ss_z, prog_bar=True)
        self.log("test_ss_s", ss_s, prog_bar=True)        

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.train_batch_size, shuffle=False, num_workers = 4)
    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size = self.valid_batch_size, shuffle=False, num_workers = 4)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size = self.test_batch_size, shuffle=False, num_workers = 4)

    def configure_optimizers(self):
        params = chain(
            self.z_convs.parameters(),
            self.s_convs.parameters(),
            self.z_mu.parameters(),
            self.z_var.parameters(),
            self.s_mu.parameters(),
            self.s_var.parameters(),
            self.d_fc_1.parameters(),
            self.decode_convs.parameters(),
        )

        opt = torch.optim.Adam(params)
        return opt