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
from models.phi_seg import PhiSeg
import yaml
from torch.utils.data import DataLoader
from evaluation.info_nce_score import InfoNCE
import numpy as np
from sklearn.metrics import silhouette_score
from dataloader import get_datasets

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

class ContrastivePhiSeg(pl.LightningModule):
    def __init__(self, config, train_ds = None, valid_ds = None):
        super(ContrastivePhiSeg, self).__init__()
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.train_batch_size = config['CONTRASTIVE_MODEL_TRAIN']['train_batch_size']
        self.valid_batch_size = config['CONTRASTIVE_MODEL_TRAIN']['val_batch_size']

        self.validation_mu = []
        self.validation_labels = []
        version_name = f'v_{config["CONTRASTIVE_MODEL_TRAIN"]["version_number"]}'
        self.valid_dir = os.path.join(config['CONTRASTIVE_MODEL_TRAIN']['output_dir'],config['CONTRASTIVE_MODEL_TRAIN']['dataset'],'valid',version_name)
        os.makedirs(self.valid_dir, exist_ok=True)
        self.test_dir = os.path.join(config['CONTRASTIVE_MODEL_TRAIN']['output_dir'],config['CONTRASTIVE_MODEL_TRAIN']['dataset'],'test',version_name)
        os.makedirs(self.test_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(config['SEGMENTATION_MODEL_TRAIN']['output_dir'], config['SEGMENTATION_MODEL_TRAIN']['dataset'], 'checkpoints', 'v_' + str(config['SEGMENTATION_MODEL_TRAIN']['version_number']), 'epoch='+ str(config['SEGMENTATION_MODEL_TRAIN']['epoch_number']) +'.ckpt')
        self.segmentor = PhiSeg.load_from_checkpoint(checkpoint_path, config=config)
        self.nce_loss = InfoNCE()

    def forward(self, x, mask):
        _, mu_post, sig_post, mu_prior, sig_prior = self.segmentor(x, mask)
        sample_post = self.segmentor.sample_posterior(mu_post, sig_post)
        sample_post_flattened = []
        for i in range(len(sample_post)):
            sample_post_flattened.append(sample_post[i].flatten(start_dim=1))

        sample_post_flattened = torch.cat(sample_post_flattened, dim=1)
        return sample_post_flattened
    
    def encode(self, x, mask):
        _, mu_post, sig_post, mu_prior, sig_prior = self.segmentor(x, mask)
        mu_post_flattened = []
        for i in range(len(mu_post)):
            mu_post_flattened.append(mu_post[i].flatten(start_dim=1))
        mu_post_flattened = torch.cat(mu_post_flattened, dim=1)
        return mu_post_flattened

    def training_step(self, batch, batch_idx):
        x, labels, mask = batch
        neg_x = x[labels == 0]
        pos_x = x[labels == 1]
        neg_mask = mask[labels == 0].unsqueeze(1)
        pos_mask = mask[labels == 1].unsqueeze(1)

        if len(neg_mask) == 0 or len(pos_mask) == 0:
            nce_loss_batch = torch.tensor(0.0).cuda().requires_grad_(True)
        else:
            sample_post_neg_flattened = self.forward(neg_x, neg_mask)
            sample_post_pos_flattened = self.forward(pos_x, pos_mask)

            nce_loss_pos = self.nce_loss(sample_post_pos_flattened, sample_post_pos_flattened, sample_post_neg_flattened)
            nce_loss_neg = self.nce_loss(sample_post_neg_flattened, sample_post_neg_flattened, sample_post_pos_flattened)
            nce_loss_batch = nce_loss_pos + nce_loss_neg

        return {'loss': nce_loss_batch}
    
    def training_epoch_end(self, outputs):
        avg_nce_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('loss', avg_nce_loss)

    def validation_step(self, batch, batch_idx):
        x, labels, mask = batch
        mask = torch.unsqueeze(mask, 1)

        mu_post_flattened = self.forward(x, mask)

        self.validation_mu.append(mu_post_flattened.detach().cpu().numpy())
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
    config = read_yaml('./../configs/config_brca.yaml')
    contrastive_phi_seg = ContrastivePhiSeg(config)
    
    