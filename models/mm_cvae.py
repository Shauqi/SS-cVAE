import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import mmd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from model_components.Encoder_Decoder import Encoder, Decoder

class MM_cVAE(pl.LightningModule):
    def __init__(self, config, train_ds = None, valid_ds = None, test_ds = None):
        super(MM_cVAE, self).__init__()
        self.save_img_path = f"{config['PROJECT_DIR']}{config['output_dir']}/{config['dataset']}/valid/v_{config['version_number']}"
        in_channels = 3
        dec_channels = 8
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.salient_latent_size = config['model_parameters']['salient_latent_size']
        self.background_latent_size = config['model_parameters']['background_latent_size']
        bias = False
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.train_batch_size = config['train_batch_size']
        self.valid_batch_size = config['val_batch_size']

        self.z_convs = Encoder(self.in_channels, self.dec_channels, bias)
        self.s_convs = Encoder(self.in_channels, self.dec_channels, bias)
        self.z_mu = nn.Linear(dec_channels * 8 * 8 * 8, self.background_latent_size, bias=bias)
        self.z_var = nn.Linear(dec_channels * 8 * 8 * 8, self.background_latent_size, bias=bias)
        self.s_mu = nn.Linear(dec_channels * 8 * 8 * 8, self.salient_latent_size, bias=bias)
        self.s_var = nn.Linear(dec_channels * 8 * 8 * 8, self.salient_latent_size, bias=bias)

        self.decode_convs = Decoder(self.in_channels, self.dec_channels, bias)

        total_latent_size = self.salient_latent_size + self.background_latent_size

        self.background_disentanglement_penalty = config['model_parameters']['background_disentanglement_penalty']
        self.salient_disentanglement_penalty = config['model_parameters']['salient_disentanglement_penalty']

        self.d_fc_1 = nn.Linear(total_latent_size, dec_channels * 8 * 8 * 8)

        self.validation_mu_s = []
        self.validation_mu_z = []
        self.validation_labels = []

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

        hz = hz.view(-1, self.dec_channels * 8 * 8 * 8)
        hs = hs.view(-1, self.dec_channels * 8 * 8 * 8)

        return self.z_mu(hz), self.z_var(hz), self.s_mu(hs), self.s_var(hs)
    
    def test_encode(self, batch, device):
        img, label, mask = batch
        img = img.to(device)
        label = label.to(device)
        mask = mask.unsqueeze(1).to(device)
        hz, Fz = self.z_convs(img)
        hs, Fs = self.s_convs(img)

        hz = hz.view(-1, self.dec_channels * 8 * 8 * 8)
        hs = hs.view(-1, self.dec_channels * 8 * 8 * 8)

        return self.z_mu(hz), self.z_var(hz), self.s_mu(hs), self.s_var(hs)

    def decode(self, z):
        z = F.leaky_relu(self.d_fc_1(z), negative_slope=0.2)
        z = z.view(-1, self.dec_channels * 8, 8, 8)
        return self.decode_convs(z)

    def decode_combined(self, z):
        z = F.leaky_relu(self.d_fc_1(z), negative_slope=0.2)
        z = z.view(-1, self.dec_channels * 8, 8, 8)
        return self.decode_convs(z)

    def forward_target(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        return self.decode(torch.cat([z, s], dim=1)), mu_z, logvar_z, mu_s, logvar_s, z, s

    def forward_background(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        salient_var_vector = torch.zeros(x.shape[0], self.salient_latent_size).to(self.device)
        return self.decode(torch.cat([z, salient_var_vector], dim=1)), mu_z, logvar_z, mu_s, logvar_s, z, s

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
        x, labels, mask = batch_test
        background = x[labels == 0].to(self.device)
        targets = x[labels != 0].to(self.device)

        min_index_len = min(len(background), len(targets))
        background = background[:min_index_len]
        targets = targets[:min_index_len]

        mu_z_bg, _, mu_s_bg, _ = self.encode(background)
        mu_z_t, _, mu_s_t, _ = self.encode(targets)

        img_recon_bg = self.decode(torch.cat([mu_z_bg, mu_s_bg], dim=1))
        img_recon_t = self.decode(torch.cat([mu_z_t, mu_s_t], dim=1))

        salient_var_vector = torch.zeros_like(mu_s_bg)

        swap_img_zbg_st = self.decode(torch.cat([mu_z_bg, mu_s_t], dim=1))
        swap_img_zt_zeros = self.decode(torch.cat([mu_z_t, salient_var_vector], dim=1))

        output = torch.cat((background, targets, img_recon_bg, img_recon_t, swap_img_zbg_st, swap_img_zt_zeros), 0)

        img_name = self.save_img_path + '/epochs_' + str(self.current_epoch) + '_img_swap.png'

        reshape_background = background.permute(0,2,3,1).detach().cpu().numpy()
        reshape_targets = targets.permute(0,2,3,1).detach().cpu().numpy()
        reshape_img_recon_bg = img_recon_bg.permute(0,2,3,1).detach().cpu().numpy()
        reshape_img_recon_t = img_recon_t.permute(0,2,3,1).detach().cpu().numpy()
        reshape_swap_img_zbg_st = swap_img_zbg_st.permute(0,2,3,1).detach().cpu().numpy()
        reshape_swap_img_zt_zeros = swap_img_zt_zeros.permute(0,2,3,1).detach().cpu().numpy()

        ax, fig = plt.subplots(6, min_index_len, figsize=(min_index_len, 6))
        for i in range(min_index_len):
            fig[0][i].imshow(reshape_background[i])
            fig[0][i].axis('off')
            fig[1][i].imshow(reshape_targets[i])
            fig[1][i].axis('off')
            fig[2][i].imshow(reshape_img_recon_bg[i])
            fig[2][i].axis('off')
            fig[3][i].imshow(reshape_img_recon_t[i])
            fig[3][i].axis('off')
            fig[4][i].imshow(reshape_swap_img_zbg_st[i])
            fig[4][i].axis('off')
            fig[5][i].imshow(reshape_swap_img_zt_zeros[i])
            fig[5][i].axis('off')

        plt.savefig(img_name)
        plt.close()

    def training_step(self, batch, batch_idx):
        x, labels, mask = batch
        background = x[labels == 0]
        targets = x[labels != 0]

        recon_batch_bg, mu_z_bg, logvar_z_bg, mu_s_bg, logvar_s_bg, z_bg, s_bg = self.forward_background(background)
        recon_batch_tar, mu_z_tar, logvar_z_tar, mu_s_tar, logvar_s_tar, z_tar, s_tar = self.forward_target(targets)

        MSE_bg = F.mse_loss(recon_batch_bg, background, reduction='sum')
        MSE_tar = F.mse_loss(recon_batch_tar, targets, reduction='sum')

        KLD_z_bg = -0.5 * torch.sum(1 + logvar_z_bg - mu_z_bg.pow(2) - logvar_z_bg.exp())
        KLD_z_tar = -0.5 * torch.sum(1 + logvar_z_tar - mu_z_tar.pow(2) - logvar_z_tar.exp())
        KLD_s_tar = -0.5 * torch.sum(1 + logvar_s_tar - mu_s_tar.pow(2) - logvar_s_tar.exp())

        loss = (MSE_bg + KLD_z_bg) + (MSE_tar + KLD_z_tar + KLD_s_tar)

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
        self.log('background_mmd_loss', background_mmd_loss, prog_bar=True)
        self.train_ds.shuffle()

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            os.makedirs(self.save_img_path, exist_ok=True)
            self.save_swapped_image(batch)
        x, labels, mask = batch
        mu_z, _, mu_s, _ = self.encode(x)
        self.validation_mu_z.append(mu_z.cpu().numpy()) 
        self.validation_mu_s.append(mu_s.cpu().numpy())
        self.validation_labels.append(labels.cpu().numpy())
        return

    def validation_epoch_end(self, outputs):
        self.validation_mu_z = np.concatenate(self.validation_mu_z, axis=0)
        self.validation_mu_s = np.concatenate(self.validation_mu_s, axis=0)
        self.validation_labels = np.concatenate(self.validation_labels, axis=0)
        ss_z = silhouette_score(self.validation_mu_z, self.validation_labels)
        ss_s = silhouette_score(self.validation_mu_s, self.validation_labels)
        self.log("val_ss_z", ss_z, prog_bar=True)
        self.log("val_ss_s", ss_s, prog_bar=True)
        self.validation_mu_z = []
        self.validation_mu_s = []
        self.validation_labels = []
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.train_batch_size, shuffle=False, num_workers = 4)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size = self.valid_batch_size, shuffle=False, num_workers = 4)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters())
        return opt