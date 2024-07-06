import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from itertools import chain
# from utils import mmd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
import yaml

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

class cVAE(pl.LightningModule):
    def __init__(self, config=None, train_ds = None, valid_ds = None, test_ds = None):
        super(cVAE, self).__init__()
        self.save_img_path = f"{config['PROJECT_DIR']}{config['output_dir']}/{config['dataset']}/valid/v_{config['version_number']}/"
        os.makedirs(self.save_img_path, exist_ok=True)
        in_channels = 3
        dec_channels = 8 #32
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

        self.z_convs = nn.Sequential(
            nn.Conv2d(in_channels, dec_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels),


            nn.Conv2d(dec_channels, dec_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels*2),


            nn.Conv2d(dec_channels * 2, dec_channels * 4, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 4),

            nn.Conv2d(dec_channels * 4, dec_channels * 8, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 8)
        )

        self.s_convs = nn.Sequential(
            nn.Conv2d(in_channels, dec_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels),

            nn.Conv2d(dec_channels, dec_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 2),

            nn.Conv2d(dec_channels * 2, dec_channels * 4, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 4),

            nn.Conv2d(dec_channels * 4, dec_channels * 8, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 8)
        )

        self.z_mu = nn.Linear(dec_channels * 8 * 8 * 8, self.background_latent_size, bias=bias)
        self.z_var = nn.Linear(dec_channels * 8 * 8 * 8, self.background_latent_size, bias=bias)

        self.s_mu = nn.Linear(dec_channels * 8 * 8 * 8, self.salient_latent_size, bias=bias)
        self.s_var = nn.Linear(dec_channels * 8 * 8 * 8, self.salient_latent_size, bias=bias)

        self.decode_convs = nn.Sequential(
            nn.ConvTranspose2d(dec_channels * 8, dec_channels * 4,
                               kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 4),

            nn.ConvTranspose2d(dec_channels * 4, dec_channels * 2,
                               kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 2),

            nn.ConvTranspose2d(dec_channels * 2, dec_channels,
                               kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels),

            nn.ConvTranspose2d(dec_channels, in_channels,
                               kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.Sigmoid()
        )

        total_latent_size = self.salient_latent_size + self.background_latent_size

        self.discriminator = nn.Linear(total_latent_size, 1)
        self.d_fc_1 = nn.Linear(total_latent_size, dec_channels * 8 * 8 * 8)
        # self.batch_test = batch_test
        # self.save_hyperparameters(ignore=["batch_test"])

        self.validation_mu_z = []
        self.validation_mu_s = []
        self.validation_labels = []

    def reparameterize(self, mu, log_var):
        #:param mu: mean from the encoder's latent space
        #:param log_var: log variance from the encoder's latent space
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def encode(self, x):
        hz = self.z_convs(x)
        hs = self.s_convs(x)

        hz = hz.view(-1, self.dec_channels * 8 * 8 * 8)
        hs = hs.view(-1, self.dec_channels * 8 * 8 * 8)

        return self.z_mu(hz), self.z_var(hz), self.s_mu(hs), self.s_var(hs)

    def decode(self, z):
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

        img_name = self.save_img_path + 'epochs_' + str(self.current_epoch) + '_img_swap.png'

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

    def training_step(self, batch, batch_idx):
        x, labels, mask = batch
        x = x.to(self.device)
        labels = labels.to(self.device)
        mask = mask.to(self.device)
        background = x[labels == 0]
        targets = x[labels != 0]

        # if self.current_epoch % self.save_img_epoch ==0 and batch_idx ==0:
        #     self.save_swaped_image(self.batch_test)

        recon_batch_bg, mu_z_bg, logvar_z_bg, mu_s_bg, logvar_s_bg, z_bg, s_bg = self.forward_background(background)
        recon_batch_tar, mu_z_tar, logvar_z_tar, mu_s_tar, logvar_s_tar, z_tar, s_tar = self.forward_target(targets)

        MSE_bg = F.mse_loss(recon_batch_bg, background, reduction='sum')
        MSE_tar = F.mse_loss(recon_batch_tar, targets, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_z_bg = -0.5 * torch.sum(1 + logvar_z_bg - mu_z_bg.pow(2) - logvar_z_bg.exp())
        KLD_z_tar = -0.5 * torch.sum(1 + logvar_z_tar - mu_z_tar.pow(2) - logvar_z_tar.exp())
        KLD_s_tar = -0.5 * torch.sum(1 + logvar_s_tar - mu_s_tar.pow(2) - logvar_s_tar.exp())

        loss = (MSE_bg + KLD_z_bg) + (MSE_tar + KLD_z_tar + KLD_s_tar)

        z1, z2 = z_tar[:int(len(z_tar) / 2)], z_tar[int(len(z_tar) / 2):]
        s1, s2 = s_tar[:int(len(s_tar) / 2)], s_tar[int(len(s_tar) // 2):]

        # In case we have an odd number of target samples
        size = min(len(z1), len(z2))
        z1, z2, s1, s2 = z1[:size], z2[:size], s1[:size], s2[:size]

        q = torch.cat([
            torch.cat([z1, s1], dim=1),
            torch.cat([z2, s2], dim=1)
        ])

        q_bar = torch.cat([
            torch.cat([z1, s2], dim=1),
            torch.cat([z2, s1], dim=1)
        ])

        q_bar_score = F.sigmoid(self.discriminator(q_bar))
        q_score = F.sigmoid(self.discriminator(q))

        eps = 1e-6
        q_score = q_score.clone().where(q_score == 0, torch.tensor(eps).to(self.device))
        q_score = q_score.clone().where(q_score == 1, torch.tensor(1 - eps).to(self.device))

        q_bar_score = q_bar_score.clone().where(q_bar_score == 0, torch.tensor(eps).to(self.device))
        q_bar_score = q_bar_score.clone().where(q_bar_score == 1, torch.tensor(1 - eps).to(self.device))

        tc_loss = torch.log(q_score / (1 - q_score)).sum()
        discriminator_loss = (- torch.log(q_score) - torch.log(1 - q_bar_score)).sum()

        loss += tc_loss + discriminator_loss

        # self.log('tc_loss', tc_loss, prog_bar=True)
        # self.log('discriminator_loss', discriminator_loss, prog_bar=True)

        # self.log('MSE_bg', MSE_bg, prog_bar=True)
        # self.log('MSE_tar', MSE_tar, prog_bar=True)
        # self.log('KLD_z_bg', KLD_z_bg, prog_bar=True)
        # self.log('KLD_z_tar', KLD_z_tar, prog_bar=True)
        # self.log('KLD_s_tar', KLD_s_tar, prog_bar=True)

        return {'loss': loss, 'MSE_bg': MSE_bg, 'MSE_tar': MSE_tar, 'KLD_z_bg': KLD_z_bg, 'KLD_z_tar': KLD_z_tar, 'KLD_s_tar': KLD_s_tar, 'tc_loss': tc_loss, 'discriminator_loss': discriminator_loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_MSE_bg = torch.stack([x['MSE_bg'] for x in outputs]).mean()
        avg_MSE_tar = torch.stack([x['MSE_tar'] for x in outputs]).mean()
        avg_KLD_z_bg = torch.stack([x['KLD_z_bg'] for x in outputs]).mean()
        avg_KLD_z_tar = torch.stack([x['KLD_z_tar'] for x in outputs]).mean()
        avg_KLD_s_tar = torch.stack([x['KLD_s_tar'] for x in outputs]).mean()
        avg_tc_loss = torch.stack([x['tc_loss'] for x in outputs]).mean()
        avg_discriminator_loss = torch.stack([x['discriminator_loss'] for x in outputs]).mean()

        self.log('avg_loss', avg_loss, prog_bar=True)
        self.log('avg_MSE_bg', avg_MSE_bg, prog_bar=True)
        self.log('avg_MSE_tar', avg_MSE_tar, prog_bar=True)
        self.log('avg_KLD_z_bg', avg_KLD_z_bg, prog_bar=True)
        self.log('avg_KLD_z_tar', avg_KLD_z_tar, prog_bar=True)
        self.log('avg_KLD_s_tar', avg_KLD_s_tar, prog_bar=True)
        self.log('avg_tc_loss', avg_tc_loss, prog_bar=True)
        self.log('avg_discriminator_loss', avg_discriminator_loss, prog_bar=True)
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

    def configure_optimizers(self):

        # params = chain(
        #     self.z_convs.parameters(),
        #     self.s_convs.parameters(),
        #     self.z_mu.parameters(),
        #     self.z_var.parameters(),
        #     self.s_mu.parameters(),
        #     self.s_var.parameters(),
        #     self.d_fc_1.parameters(),
        #     self.decode_convs.parameters(),
        #     self.discriminator.parameters()
        # )

        opt = torch.optim.Adam(self.parameters())
        return opt
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.train_batch_size, shuffle=False, num_workers = 4)
    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size = self.valid_batch_size, shuffle=False, num_workers = 4)


if __name__ == "__main__":
    config = read_yaml('./../configs/consep/cvae.yaml')

    model_name = config['CVAE_MODEL_TRAIN']['model_name']
    train_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['train_dir']}"
    val_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['val_dir']}"
    test_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['test_dir']}"

    from dataloader.brca_loader import BRCA_BIN_File_Loader
    if model_name == 'chc_vae' or model_name == 'ch_vae' or model_name == 'resnet_cvae':
        from dataloader.brca_loader import BRCA_BIN_Paired_File_Loader
        train_ds = BRCA_BIN_Paired_File_Loader(train_dir)
    else:
        train_ds = BRCA_BIN_File_Loader(train_dir)
    valid_ds = BRCA_BIN_File_Loader(val_dir)
    test_ds = BRCA_BIN_File_Loader(test_dir)

    model = cVAE(config, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds)
    device = torch.device(f"cuda:{config['GPU_LIST'][0]}" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # train_dataloader = model.train_dataloader()

    # for batch in train_dataloader:
    #     loss = model.training_step(batch, 0)
    #     print(loss)
    #     break

    valid_dataloader = model.val_dataloader()

    for batch in valid_dataloader:
        model.validation_step(batch, 0)
        break