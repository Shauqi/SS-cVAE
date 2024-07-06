import torch
import torch.nn as nn
import torch.nn.functional as F

# Define ResNet block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection if input and output dimensions don't match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

# Encoder class
class ResNetEncoder(nn.Module):
    def __init__(self, image_channels=3, z_dim=128):
        super(ResNetEncoder, self).__init__()

        self.blocks = nn.Sequential(
            ResNetBlock(image_channels, 64, stride=2),
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 512, stride=2),
            ResNetBlock(512, 1024, stride=2),
            ResNetBlock(1024, 2048, stride=2),
            ResNetBlock(2048, 2048, stride=2),
        )

        self.fc_mu = nn.Linear(2048, z_dim)
        self.fc_logvar = nn.Linear(2048, z_dim)

    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Decoder class
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
    
class BinaryClassifier(nn.Module):
    def __init__(self, z_dim=128):
        super(BinaryClassifier, self).__init__()

        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = torch.sigmoid(self.fc3(z))
        return z