import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer.phiseg_trainer import PhiSegTrainer
from models.phi_seg_components import Encoder, Decoder
import yaml

def load_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

class CHCVAE(nn.Module):
    def __init__(self, config):
        super(CHCVAE, self).__init__()
        self.config = config
        checkpoint_path = os.path.join(config['SEGMENTATION_MODEL_TRAIN']['output_dir'], config['SEGMENTATION_MODEL_TRAIN']['dataset'], 'checkpoints', 'v_' + str(config['SEGMENTATION_MODEL_TRAIN']['version_number']), 'epoch='+ str(config['SEGMENTATION_MODEL_TRAIN']['epoch_number']) +'.ckpt')
        self.s_encoder = PhiSegTrainer.load_from_checkpoint(checkpoint_path, config=config['SEGMENTATION_MODEL_TRAIN'])
        self.z_encoder = Encoder(in_channels=3, dec_channels=8, latent_size=50)
        self.decoder = Decoder(in_channels=3, dec_channels=8, latent_size=2778)
        self.z_decoder = Decoder(in_channels=3, dec_channels=8, latent_size=50)

    def encode(self, x, mask):
        _, mu_s, logvar_s, _, _ = self.s_encoder.model.forward_s(x, mask)
        mu_z, logvar_z = self.z_encoder(x)
        mu_s_flattened = []
        for i in range(len(mu_s)):
            mu_s_flattened.append(mu_s[i].view(mu_s[i].size(0), -1))
        mu_s_flattened = torch.cat(mu_s_flattened, dim = 1)        
        return mu_z, logvar_z, mu_s_flattened, logvar_s

    def decode_combined(self, combined_vector):
        combined_recon = self.decoder(combined_vector)
        return combined_recon
    
    def forward_target(self, x, mask):
        _, mu_s, logvar_s, _, _ = self.s_encoder.model.forward_s(x, mask)
        s = self.s_encoder.model.sample_posterior(mu_s, logvar_s)
        mu_z, logvar_z = self.z_encoder(x)
        z = self.z_encoder.reparameterize(mu_z, logvar_z)
        s_flattened = []
        for i in range(len(s)):
            s_flattened.append(s[i].view(s[i].size(0), -1))

        s_flattened = torch.cat(s_flattened, dim = 1)
        combined_vector = torch.cat([z, s_flattened], dim=1)
        combined_recon = self.decoder(combined_vector)
        z_recon  = self.z_decoder(z)
        return combined_recon, z_recon, mu_z, logvar_z

    def forward_background(self, x, mask):
        mu_z, logvar_z = self.z_encoder(x)
        z = self.z_encoder.reparameterize(mu_z, logvar_z)
        zero_vector = torch.zeros(x.shape[0], 2778 - 50).cuda()
        combined_vector = torch.cat([z, zero_vector], dim=1)
        combined_recon = self.decoder(combined_vector)
        z_recon  = self.z_decoder(z)
        return combined_recon, z_recon, mu_z, logvar_z
    
    def forward(self, x, mask):
        _, mu_s, logvar_s, _, _ = self.s_encoder.model.forward_s(x, mask)
        s = self.s_encoder.model.sample_posterior(mu_s, logvar_s)
        mu_z, logvar_z = self.z_encoder(x)
        z = self.z_encoder.reparameterize(mu_z, logvar_z)
        s_flattened = []
        for i in range(len(s)):
            s_flattened.append(s[i].view(s[i].size(0), -1))

        s_flattened = torch.cat(s_flattened, dim = 1)
        combined_vector = torch.cat([z, s_flattened], dim=1)
        combined_recon = self.decoder(combined_vector)
        return combined_recon, z, mu_z, logvar_z, s, mu_s, logvar_s
        
if __name__ == '__main__':
    config = load_config('./../configs/config_brca_v2.yaml')
    model = CHCVAE(config)
    model = model.cuda()
    x = torch.randn(64, 3, 128, 128)
    x = x.cuda()
    mask = torch.randn(64, 1, 128, 128)
    mask = mask.cuda()
    recon, _, _, _ = model.forward_target(x, mask)
    print(recon.shape)