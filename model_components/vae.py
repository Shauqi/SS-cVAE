import torch
import torch.nn as nn
import torch.nn.functional as F
from .Encoder_Decoder import MTL_Encoder, Decoder

class VAE(nn.Module):
    def __init__(self, latent_size, num_classes, in_channels = 3, dec_channels = 32, bias = False):
        super(VAE, self).__init__()
        self.dec_channels = dec_channels
        self.conv_encoder = MTL_Encoder(in_channels, dec_channels, bias)
        self.mu_encoder = nn.Linear(dec_channels * 8 * 4 * 4, latent_size, bias=bias)
        self.var_encoder = nn.Linear(dec_channels * 8 * 4 * 4, latent_size, bias=bias)
        self.linear_decoder = nn.Linear(latent_size, dec_channels * 8 * 4 * 4, bias=bias)
        self.conv_decoder = Decoder(in_channels, dec_channels, bias)
        self.classifier = nn.Sequential(nn.Linear(latent_size, num_classes))

    def reparameterize(self, mu, log_var):
        #:param mu: mean from the encoder's latent space
        #:param log_var: log variance from the encoder's latent space
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def encode(self, x):
        hz = self.conv_encoder(x)
        hz = hz.view(-1, self.dec_channels * 8 * 4 * 4)
        mu = self.mu_encoder(hz)
        log_var = self.var_encoder(hz)
        return mu, log_var

    def decode(self, z):
        z = F.leaky_relu(self.linear_decoder(z), negative_slope=0.2)
        z = z.view(-1, self.dec_channels * 8, 4, 4)
        return self.conv_decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        class_logits = self.classifier(z)
        return mu, log_var, recon, class_logits