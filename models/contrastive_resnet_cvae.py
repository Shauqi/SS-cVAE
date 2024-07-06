import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os
import yaml
from torch.utils.data import DataLoader
from evaluation.info_nce_score import InfoNCE
import numpy as np
from sklearn.metrics import silhouette_score
from dataloader import get_til_vs_other_datasets

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

class ContrastiveResNetCVAE(pl.LightningModule):
    def __init__(self, config, train_ds = None, valid_ds = None):
        super(ContrastiveResNetCVAE, self).__init__()
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.train_batch_size = config['CONTRASTIVE_MODEL_TRAIN']['train_batch_size']
        self.valid_batch_size = config['CONTRASTIVE_MODEL_TRAIN']['val_batch_size']

        self.validation_mu = []
        self.validation_labels = []
        
        version_number = config['CVAE_MODEL_TRAIN']['version_number']
        version_name = f'v_{version_number}'
        chkpt_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['chkpt_dir']}/{version_name}/epoch={config['CVAE_MODEL_TRAIN']['epoch_number']}.ckpt"

        from models.resnet_cvae import ResNet_cVAE
        self.model = ResNet_cVAE.load_from_checkpoint(chkpt_dir, config = config['CVAE_MODEL_TRAIN'])
        self.nce_loss = InfoNCE()

    def forward(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.model.encode(x)
        s = self.model.reparameterize(mu_s, logvar_s)
        return mu_s, logvar_s, s
    
    def encode(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.model.encode(x)
        s = self.model.reparameterize(mu_s, logvar_s)
        return mu_s,logvar_s, s
    
    def test_encode(self, batch, device):
        x, labels, mask = batch
        x = x.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask, 1)
        mu_z, logvar_z, mu_s, logvar_s = self.model.encode(x)
        return mu_z, logvar_z, mu_s, logvar_s

    def training_step(self, batch, batch_idx):
        x, labels, mask = batch
        x = x.to(self.device)
        labels = labels.to(self.device)
        mask = mask.to(self.device)
        neg_x = x[labels == 0]
        pos_x = x[labels == 1]
        neg_mask = mask[labels == 0].unsqueeze(1)
        pos_mask = mask[labels == 1].unsqueeze(1)

        if len(neg_mask) == 0 or len(pos_mask) == 0:
            nce_loss_batch = torch.tensor(0.0).cuda().requires_grad_(True)
        else:
            neg_mu, neg_log_var, neg_sample = self.forward(neg_x)
            pos_mu, pos_log_var, pos_sample = self.forward(pos_x)

            neg_kld_loss = -0.5 * torch.sum(1 + neg_log_var - neg_mu.pow(2) - neg_log_var.exp())
            pos_kld_loss = -0.5 * torch.sum(1 + pos_log_var - pos_mu.pow(2) - pos_log_var.exp())

            nce_loss_pos = self.nce_loss(pos_sample, pos_sample, neg_sample)
            nce_loss_neg = self.nce_loss(neg_sample, neg_sample, pos_sample)
            nce_loss_pos_mu = self.nce_loss(pos_mu, pos_mu, neg_mu)
            nce_loss_neg_mu = self.nce_loss(neg_mu, neg_mu, pos_mu)
            nce_loss_batch = nce_loss_pos + nce_loss_neg + neg_kld_loss + pos_kld_loss + nce_loss_pos_mu + nce_loss_neg_mu

        return {'loss': nce_loss_batch}
    
    def training_epoch_end(self, outputs):
        avg_nce_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('loss', avg_nce_loss)

    def validation_step(self, batch, batch_idx):
        x, labels, mask = batch
        x = x.to(self.device)
        labels = labels.to(self.device)
        mask = mask.to(self.device)
        mask = torch.unsqueeze(mask, 1)
        mu, logvar, sample = self.encode(x)

        self.validation_mu.append(mu.detach().cpu().numpy())
        self.validation_labels.append(labels.detach().cpu().numpy())

    def validation_epoch_end(self, outputs):
        self.validation_mu = np.concatenate(self.validation_mu, axis=0)
        self.validation_labels = np.concatenate(self.validation_labels, axis=0)

        ss = silhouette_score(self.validation_mu, self.validation_labels)
        self.log('Validation Silhouette Score', ss)

        self.validation_mu = []
        self.validation_labels = []
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.train_batch_size, shuffle=True, num_workers = 4)
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size = self.valid_batch_size, shuffle=False, num_workers = 4)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return opt

if __name__ == '__main__':
    config = read_yaml('./../configs/config_brca_resnet_cvae.yaml')
    train_ds, valid_ds, test_ds = get_til_vs_other_datasets(config)
    model = ContrastiveResNetCVAE(config, train_ds, valid_ds)

    # train_loader = model.train_dataloader()
    # for batch in train_loader:
    #     print(model.training_step(batch, 0))
    #     break

    valid_loader = model.val_dataloader()
    for batch in valid_loader:
        print(model.validation_step(batch, 0))
        break
    