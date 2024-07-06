import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)
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
from visualizer.visualize_segmentation import visualize_masks, visualize_recons
import yaml

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetEncoder(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetEncoder, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.mu_encoder = nn.Linear(512 * 4 * 4, 128)
        self.logvar_encoder = nn.Linear(512 * 4 * 4, 128)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        mu = self.mu_encoder(out)
        logvar = self.logvar_encoder(out)
        return mu, logvar
    
class ResNetDecoder(nn.Module):
    def __init__(self, z_dim=128, image_channels=3):
        super(ResNetDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4 * 4 * 2048),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (2048, 4, 4)),
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        recon_x = self.decoder(z)
        return recon_x

class SS_cVAE(pl.LightningModule):
    def __init__(self, config, train_ds = None, valid_ds = None, test_ds = None):
        super(SS_cVAE, self).__init__()
        self.save_img_path = f"{config['PROJECT_DIR']}{config['output_dir']}/valid/v_{config['version_number']}"
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.train_batch_size = config['train_batch_size']
        self.valid_batch_size = config['val_batch_size']
        self.test_batch_size = config['test_batch_size']
        self.num_classes = config['num_classes']

        block = BasicBlock
        num_blocks = [2, 2, 2, 2]

        self.z_convs = ResNetEncoder(block, num_blocks, num_classes=self.num_classes)
        self.s_convs = ResNetEncoder(block, num_blocks, num_classes=self.num_classes)

        self.combined_conv_decoder = ResNetDecoder(z_dim = 128 * 2)
        self.s_conv_decoder = ResNetDecoder(z_dim=128, image_channels=1)
        self.z_conv_decoder = ResNetDecoder(z_dim=128, image_channels=3)

        self.classifier = nn.Linear(128, self.num_classes)
        
        self.background_disentanglement_penalty = config['model_parameters']['background_disentanglement_penalty']
        self.salient_disentanglement_penalty = config['model_parameters']['salient_disentanglement_penalty']
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.classification_loss = nn.BCEWithLogitsLoss()
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
        z_mu, z_var = self.z_convs(x)
        s_mu, s_var = self.s_convs(x)
        return z_mu, z_var, s_mu, s_var
    
    def test_encode(self, batch, device):
        x, labels, _ = batch
        x = x.to(device)
        z_mu, z_var = self.z_convs(x)
        s_mu, s_var = self.s_convs(x)
        return z_mu, z_var, s_mu, s_var
    
    def gan_encode(self, x):
        z_mu, z_var = self.z_convs(x)
        z = self.reparameterize(z_mu, z_var)
        s_mu, s_var = self.s_convs(x)
        s = self.reparameterize(s_mu, s_var)
        return s, s_mu, s_var, z, z_mu, z_var
    
    def decode_combined(self, combined_vector):
        return self.combined_conv_decoder(combined_vector)
    
    def forward_target(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        combined_recon = self.combined_conv_decoder(torch.cat([z, s], dim=1))
        s_recon = self.s_conv_decoder(mu_s)
        z_recon = self.z_conv_decoder(mu_z)
        salient_class = self.classifier(mu_s)
        return combined_recon, salient_class, s_recon, z_recon, mu_z, logvar_z, mu_s, logvar_s, z, s

    def forward_background(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        salient_var_vector = torch.zeros_like(s)
        combined_recon = self.combined_conv_decoder(torch.cat([z, salient_var_vector], dim=1))
        s_recon = self.s_conv_decoder(mu_s)
        z_recon = self.z_conv_decoder(mu_z)
        salient_class = self.classifier(mu_s)
        return combined_recon, salient_class, s_recon, z_recon, mu_z, logvar_z, mu_s, logvar_s, z, s

    def save_swapped_image(self, batch_test):
        x, labels, _ = batch_test
        background = x[labels == 0].to(self.device)
        targets = x[labels != 0].to(self.device)

        min_index_len = min(len(background), len(targets))
        if min_index_len > 4:
            min_index_len = 4
        background = background[:min_index_len]
        targets = targets[:min_index_len]

        mu_z_bg, logvar_z_bg, mu_s_bg, logvar_s_bg = self.encode(background)
        mu_z_t, logvar_z_t, mu_s_t, logvar_s_t = self.encode(targets)

        img_recon_bg = self.combined_conv_decoder(torch.cat([mu_z_bg, mu_s_bg], dim=1))
        img_recon_t = self.combined_conv_decoder(torch.cat([mu_z_t, mu_s_t], dim=1))

        salient_var_vector = torch.zeros_like(mu_s_bg)

        swap_img_zbg_st = self.combined_conv_decoder(torch.cat([mu_z_bg, mu_s_t], dim=1))
        swap_img_zt_zeros = self.combined_conv_decoder(torch.cat([mu_z_t, salient_var_vector], dim=1))

        img_name = f"{self.save_img_path}/val_epochs_{self.current_epoch}_img_swap.png" 

        reshape_background = background.permute(0,2,3,1).detach().cpu().numpy()
        reshape_targets = targets.permute(0,2,3,1).detach().cpu().numpy()
        reshape_img_recon_bg = img_recon_bg.permute(0,2,3,1).detach().cpu().numpy()
        reshape_img_recon_t = img_recon_t.permute(0,2,3,1).detach().cpu().numpy()
        reshape_swap_img_zbg_st = swap_img_zbg_st.permute(0,2,3,1).detach().cpu().numpy()
        reshape_swap_img_zt_zeros = swap_img_zt_zeros.permute(0,2,3,1).detach().cpu().numpy()

        fig, ax = plt.subplots(4, 6, figsize=(10, 10))

        for i in range(min_index_len):
            ax[i, 0].imshow(reshape_background[i])
            ax[i, 0].axis('off')
            ax[i, 1].imshow(reshape_img_recon_bg[i])
            ax[i, 1].axis('off')
            ax[i, 2].imshow(reshape_swap_img_zbg_st[i])
            ax[i, 2].axis('off')
            ax[i, 3].imshow(reshape_targets[i])
            ax[i, 3].axis('off')
            ax[i, 4].imshow(reshape_img_recon_t[i])
            ax[i, 4].axis('off')
            ax[i, 5].imshow(reshape_swap_img_zt_zeros[i])
            ax[i, 5].axis('off')

        fig.savefig(img_name)
        plt.close(fig)

    def swap_salient_features(self, batch_test, output_dir):
        x, labels, _ = batch_test
        background = x[labels == 0].to(self.device)
        targets = x[labels != 0].to(self.device)

        min_index_len = min(len(background), len(targets))
        if min_index_len > 4:
            min_index_len = 4
        background = background[:min_index_len]
        targets = targets[:min_index_len]

        mu_z_bg, logvar_z_bg, Fz_bg, mu_s_bg, logvar_s_bg, Fs_bg = self.encode(background)
        mu_z_t, logvar_z_t, Fz_t, mu_s_t, logvar_s_t, Fs_t = self.encode(targets)

        img_recon_bg = self.decode_combined(torch.cat([mu_z_bg, mu_s_bg], dim=1))
        img_recon_t = self.decode_combined(torch.cat([mu_z_t, mu_s_t], dim=1))

        salient_var_vector = torch.zeros_like(mu_s_bg)

        swap_img_zbg_st = self.decode_combined(torch.cat([mu_z_bg, mu_s_t], dim=1))
        swap_img_zt_zeros = self.decode_combined(torch.cat([mu_z_t, salient_var_vector], dim=1))

        img_name = f'{output_dir}/swap_salient_features.png'

        reshape_background = background.permute(0,2,3,1).detach().cpu().numpy()
        reshape_targets = targets.permute(0,2,3,1).detach().cpu().numpy()
        reshape_img_recon_bg = img_recon_bg.permute(0,2,3,1).detach().cpu().numpy()
        reshape_img_recon_t = img_recon_t.permute(0,2,3,1).detach().cpu().numpy()
        reshape_swap_img_zbg_st = swap_img_zbg_st.permute(0,2,3,1).detach().cpu().numpy()
        reshape_swap_img_zt_zeros = swap_img_zt_zeros.permute(0,2,3,1).detach().cpu().numpy()

        fig, ax = plt.subplots(4, 6, figsize=(10, 10))

        for i in range(min_index_len):
            ax[i, 0].imshow(reshape_background[i])
            ax[i, 0].axis('off')
            ax[i, 1].imshow(reshape_img_recon_bg[i])
            ax[i, 1].axis('off')
            ax[i, 2].imshow(reshape_swap_img_zbg_st[i])
            ax[i, 2].axis('off')
            ax[i, 3].imshow(reshape_targets[i])
            ax[i, 3].axis('off')
            ax[i, 4].imshow(reshape_img_recon_t[i])
            ax[i, 4].axis('off')
            ax[i, 5].imshow(reshape_swap_img_zt_zeros[i])
            ax[i, 5].axis('off')

        fig.savefig(img_name)
        plt.close(fig)        

    def training_step(self, batch, batch_idx):
        x, labels, mask = batch
        background = x[(labels == 0) | (labels == 2)]
        no_mask = mask[(labels == 0) | (labels == 2)]
        targets = x[labels == 1]
        mask = mask[labels == 1]

        if background.shape[0] == 0 or targets.shape[0] == 0:
            # Skip the training step if either 'background' or 'targets' is empty
            print("Skipping training step due to empty background or targets.")
            loss = torch.tensor(0.0).cuda().requires_grad_(True)
            MSE_bg = torch.tensor(0.0).cuda().requires_grad_(True)
            MSE_tar = torch.tensor(0.0).cuda().requires_grad_(True)
            KLD_z_bg = torch.tensor(0.0).cuda().requires_grad_(True)
            KLD_z_tar = torch.tensor(0.0).cuda().requires_grad_(True)
            KLD_s_tar = torch.tensor(0.0).cuda().requires_grad_(True)
            background_mmd_loss = torch.tensor(0.0).cuda().requires_grad_(True)
            salient_mmd_loss = torch.tensor(0.0).cuda().requires_grad_(True)
        else:
            background = background.to(self.device)
            no_mask = no_mask.to(self.device)
            targets = targets.to(self.device)
            mask = mask.to(self.device)
            recon_combined_bg, salient_class_bg, s_recon_bg, z_recon_bg, mu_z_bg, logvar_z_bg, mu_s_bg, logvar_s_bg, z_bg, s_bg = self.forward_background(background)
            recon_combined_tar, salient_class_tar, s_recon_tar, z_recon_tar, mu_z_tar, logvar_z_tar, mu_s_tar, logvar_s_tar, z_tar, s_tar = self.forward_target(targets)

            MSE_bg = F.mse_loss(recon_combined_bg, background, reduction='sum')
            MSE_tar = F.mse_loss(recon_combined_tar, targets, reduction='sum')

            s_recon_loss_bg = self.bce_loss(torch.squeeze(s_recon_bg, 1), no_mask)
            s_recon_loss_tar = self.bce_loss(torch.squeeze(s_recon_tar, 1), mask)
            
            # z_recon_loss_bg = F.mse_loss(z_recon_bg, background, reduction='sum')
            # z_recon_loss_tar = F.mse_loss(z_recon_tar, background, reduction='sum')

            KLD_z_bg = -0.5 * torch.sum(1 + logvar_z_bg - mu_z_bg.pow(2) - logvar_z_bg.exp())
            KLD_z_tar = -0.5 * torch.sum(1 + logvar_z_tar - mu_z_tar.pow(2) - logvar_z_tar.exp())
            KLD_s_tar = -0.5 * torch.sum(1 + logvar_s_tar - mu_s_tar.pow(2) - logvar_s_tar.exp())

            ground_truth_bg = torch.zeros_like(salient_class_bg)
            ground_truth_tar = torch.ones_like(salient_class_tar)
            s_class_loss_bg = self.classification_loss(salient_class_bg, ground_truth_bg)
            s_class_loss_tar = self.classification_loss(salient_class_tar, ground_truth_tar)

            loss = (MSE_bg + KLD_z_bg) + (MSE_tar + KLD_z_tar + KLD_s_tar)
            loss += s_class_loss_bg + s_class_loss_tar
            loss += s_recon_loss_bg + s_recon_loss_tar
            # loss += z_recon_loss_bg + z_recon_loss_tar

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
        x, labels, _ = batch
        x = x.to(self.device)
        labels = labels.to(self.device)
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        if batch_idx == 0:
            os.makedirs(self.save_img_path, exist_ok=True)
            self.save_swapped_image(batch)
            s_recon = self.s_conv_decoder(mu_s)
            z_recon = self.z_conv_decoder(mu_z)
            # visualize_masks(batch, s_recon, self.save_img_path, 'val_epochs_' + str(self.current_epoch) + '_segmentation_mask.png')
            # visualize_recons(batch, z_recon, self.save_img_path, 'val_epochs_' + str(self.current_epoch) + '_z_recon.png')
        predicted_labels = self.classifier(mu_s).argmax(dim=1)
        val_acc_s = (predicted_labels == labels).float().mean()
        self.validation_mu_z.append(mu_z.cpu().numpy()) 
        self.validation_mu_s.append(mu_s.cpu().numpy())
        self.validation_labels.append(labels.cpu().numpy())
        return {"val_acc_s": val_acc_s}

    def validation_epoch_end(self, outputs):
        val_acc_s = sum(output['val_acc_s'] for output in outputs) / (len(outputs))
        mu_z = np.concatenate(self.validation_mu_z, axis = 0)
        mu_s = np.concatenate(self.validation_mu_s, axis = 0)
        labels = np.concatenate(self.validation_labels, axis = 0)
        ss_z = silhouette_score(mu_z, labels)
        ss_s = silhouette_score(mu_s, labels)
        self.log("val_acc_s", val_acc_s, prog_bar=True)
        self.log("val_ss_z", ss_z, prog_bar=True)
        self.log("val_ss_s", ss_s, prog_bar=True)
        self.validation_mu_z = []
        self.validation_mu_s = []
        self.validation_labels = []

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.train_batch_size, shuffle=False, num_workers = 4)
    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size = self.valid_batch_size, shuffle=False, num_workers = 4)
    # def test_dataloader(self):
    #     return DataLoader(self.test_ds, batch_size = self.test_batch_size, shuffle=False, num_workers = 4)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters())
        return opt


if __name__ == "__main__":
    config = read_yaml('./configs/brca/ss_cvae_one_stage_ablation.yaml')

    model_name = config['CVAE_MODEL_TRAIN']['model_name']
    train_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['train_dir']}"
    val_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['val_dir']}"
    test_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['test_dir']}"

    from dataloader.brca_loader import BRCA_BIN_File_Loader
    if model_name == 'chc_vae' or model_name == 'ch_vae' or model_name == 'mtl_cvae' or model_name == 'resnet_cvae':
        from dataloader.brca_loader import BRCA_BIN_Paired_File_Loader
        train_ds = BRCA_BIN_Paired_File_Loader(train_dir)
        valid_ds = BRCA_BIN_File_Loader(val_dir)
        test_ds = BRCA_BIN_File_Loader(test_dir)
    elif model_name == 'ss_cvae_one_stage_ablation':
        from dataloader.brca_loader import BRCA_MTL_File_Loader
        train_ds = BRCA_MTL_File_Loader(train_dir)
        valid_ds = BRCA_MTL_File_Loader(val_dir)
        test_ds = BRCA_BIN_File_Loader(test_dir)
    else:
        train_ds = BRCA_BIN_File_Loader(train_dir)
        valid_ds = BRCA_BIN_File_Loader(val_dir)
        test_ds = BRCA_BIN_File_Loader(test_dir)

    model = SS_cVAE(config['CVAE_MODEL_TRAIN'], train_ds, valid_ds, test_ds)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # train_dataloader = model.train_dataloader()

    # for batch in train_dataloader:
    #     loss = model.training_step(batch, 0)
    #     print(loss)
    #     break

    valid_dataloader = model.val_dataloader()

    with torch.no_grad():
        for batch in valid_dataloader:
            loss = model.validation_step(batch, 0)
            print(loss)
            break