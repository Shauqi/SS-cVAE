import torch
import torch.nn as nn
import torch.nn.functional as F
from .Encoder_Decoder import MTL_Encoder, Decoder

class MM_cVAE(nn.Module):
    def __init__(self, salient_latent_size, background_latent_size, in_channels = 3, dec_channels = 32, bias = False):
        super(MM_cVAE, self).__init__()
        self.salient_latent_size = salient_latent_size
        self.background_latent_size = background_latent_size
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.z_convs = MTL_Encoder(self.in_channels, self.dec_channels, bias)
        self.s_convs = MTL_Encoder(self.in_channels, self.dec_channels, bias)
        self.z_mu = nn.Linear(dec_channels * 8 * 4 * 4, self.background_latent_size, bias=bias)
        self.z_var = nn.Linear(dec_channels * 8 * 4 * 4, self.background_latent_size, bias=bias)
        self.s_mu = nn.Linear(dec_channels * 8 * 4 * 4, self.salient_latent_size, bias=bias)
        self.s_var = nn.Linear(dec_channels * 8 * 4 * 4, self.salient_latent_size, bias=bias)
        self.decode_convs = Decoder(self.in_channels, self.dec_channels, bias)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        total_latent_size = self.salient_latent_size + self.background_latent_size
        self.d_fc_1 = nn.Linear(total_latent_size, dec_channels * 8 * 4 * 4)


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

        hz = hz.view(-1, self.dec_channels * 8 * 4 * 4)
        hs = hs.view(-1, self.dec_channels * 8 * 4 * 4)

        return self.z_mu(hz), self.z_var(hz), self.s_mu(hs), self.s_var(hs)

    def decode(self, z):
        z = F.leaky_relu(self.d_fc_1(z), negative_slope=0.2)
        z = z.view(-1, self.dec_channels * 8, 4, 4)
        return self.decode_convs(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        class_logits = self.classifier(z)
        return mu, log_var, recon, class_logits
    
    def decode_combined(self, z):
        z = F.leaky_relu(self.d_fc_1(z), negative_slope=0.2)
        z = z.view(-1, self.dec_channels * 8, 4, 4)
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