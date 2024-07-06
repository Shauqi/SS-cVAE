import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Define ResNet18 architecture
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
    def __init__(self, block, num_blocks, latent_dim=1024):
        super(ResNetEncoder, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Linear layers for mean and log variance
        self.fc_mu = nn.Linear(512 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16 * 16, latent_dim)

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
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar

class ResNetDecoder(nn.Module):
    def __init__(self, block, num_blocks, latent_dim=1024):
        super(ResNetDecoder, self).__init__()
        self.in_planes = 512 * block.expansion
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, self.in_planes * 16 * 16)

        self.layer4 = self._make_transpose_layer(block, 256, num_blocks[3], stride=2)
        self.layer3 = self._make_transpose_layer(block, 128, num_blocks[2], stride=2)
        self.layer2 = self._make_transpose_layer(block, 64, num_blocks[1], stride=2)
        self.layer1 = self._make_transpose_layer(block, 64, num_blocks[0], stride=1)

        self.conv_out = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False)

    def _make_transpose_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers.append(nn.ConvTranspose2d(self.in_planes, planes, kernel_size=4, stride=stride, padding=1, output_padding=0, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU())
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.fc(x)
        out = out.view(out.size(0), 512, 16, 16)
        out = F.relu(self.layer4(out))
        out = F.relu(self.layer3(out))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer1(out))

        out = torch.tanh(self.conv_out(out))
        out = nn.functional.interpolate(out, size=(128, 128), mode='bilinear', align_corners=False)
        return out
    

class SegmentationDecoder(nn.Module):
    def __init__(self, block, num_blocks, latent_dim=1024, num_classes=2):
        super(SegmentationDecoder, self).__init__()
        self.in_planes = 512 * block.expansion
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 512 * 16 * 16)

        self.layer4 = self._make_transpose_layer(block, 256, num_blocks[3], stride=2)
        self.layer3 = self._make_transpose_layer(block, 128, num_blocks[2], stride=2)
        self.layer2 = self._make_transpose_layer(block, 64, num_blocks[1], stride=2)
        self.layer1 = self._make_transpose_layer(block, 64, num_blocks[0], stride=1)

        self.conv_out = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False)

    def _make_transpose_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers.append(nn.ConvTranspose2d(self.in_planes, planes, kernel_size=4, stride=stride, padding=1, output_padding=0, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU())
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.fc(x)
        out = out.view(out.size(0), 512, 16, 16)
        out = F.relu(self.layer4(out))
        out = F.relu(self.layer3(out))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer1(out))
        out = torch.tanh(self.conv_out(out))
        out = nn.functional.interpolate(out, size=(128, 128), mode='bilinear', align_corners=False)
        out = torch.sigmoid(out)
        return out
    

class BinaryClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features=1):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.sigmoid(self.fc2(x))
        return x