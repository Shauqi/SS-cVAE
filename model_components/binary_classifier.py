import torch
import torch.nn as nn
from .Encoder_Decoder import MTL_Encoder

class BinaryClassifier(nn.Module):
    def __init__(self, latent_size, num_classes, in_channels = 3, dec_channels = 32, bias = False):
        super(BinaryClassifier, self).__init__()
        self.conv_encoder = MTL_Encoder(in_channels, dec_channels, bias)
        self.linear_encoder = nn.Linear(dec_channels * 8 * 4 * 4, latent_size, bias=bias)
        self.classifier = nn.Sequential(nn.Linear(latent_size, num_classes))

    def forward(self, x):
        x = self.conv_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear_encoder(x)
        x = self.classifier(x)
        return x