import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, dec_channels, bias):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, dec_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm1 = nn.BatchNorm2d(dec_channels)
        self.conv2 = nn.Conv2d(dec_channels, dec_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm2 = nn.BatchNorm2d(dec_channels*2)
        self.conv3 = nn.Conv2d(dec_channels * 2, dec_channels * 4, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm3 = nn.BatchNorm2d(dec_channels * 4)
        self.conv4 = nn.Conv2d(dec_channels * 4, dec_channels * 8, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm4 = nn.BatchNorm2d(dec_channels * 8)
        self.gradients = None
    
    def forward(self, x):
        F = []
        x = self.conv1(x)
        F.append(x)
        x = self.relu1(x)
        F.append(x)
        x = self.batch_norm1(x)
        F.append(x)
        x = self.conv2(x)
        F.append(x)
        x = self.relu2(x)
        F.append(x)
        x = self.batch_norm2(x)
        F.append(x)
        x = self.conv3(x)
        F.append(x)
        x = self.relu3(x)
        F.append(x)
        x = self.batch_norm3(x)
        F.append(x)
        x = self.conv4(x)
        F.append(x)
        x = self.relu4(x)
        F.append(x)
        x = self.batch_norm4(x)
        F.append(x)
        return x, F

class MTL_Encoder(nn.Module):
    def __init__(self, in_channels, dec_channels, bias):
        super(MTL_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, dec_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm1 = nn.BatchNorm2d(dec_channels)
        self.conv2 = nn.Conv2d(dec_channels, dec_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm2 = nn.BatchNorm2d(dec_channels*2)
        self.conv3 = nn.Conv2d(dec_channels * 2, dec_channels * 4, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm3 = nn.BatchNorm2d(dec_channels * 4)
        self.conv4 = nn.Conv2d(dec_channels * 4, dec_channels * 8, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm4 = nn.BatchNorm2d(dec_channels * 8)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batch_norm2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batch_norm3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.batch_norm4(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, dec_channels, bias):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(dec_channels * 8, dec_channels * 4, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm1 = nn.BatchNorm2d(dec_channels * 4)
        self.conv2 = nn.ConvTranspose2d(dec_channels * 4, dec_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm2 = nn.BatchNorm2d(dec_channels * 2)
        self.conv3 = nn.ConvTranspose2d(dec_channels * 2, dec_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm3 = nn.BatchNorm2d(dec_channels)
        self.conv4 = nn.ConvTranspose2d(dec_channels, in_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batch_norm2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batch_norm3(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x


class MTL_Classifier(torch.nn.Module):
    def __init__(self, latent_size, num_classes):
        super(MTL_Classifier, self).__init__()
        self.fc1 = nn.Linear(latent_size, 128)  # Add a fully connected layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # Flatten the encoder's output
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # Output class scores
        x = nn.functional.softmax(x, dim=1)
        return x


class Segmentation_Decoder(torch.nn.Module):
    def __init__(self, in_channels, dec_channels, bias):
        super(Segmentation_Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(dec_channels * 8, dec_channels * 4, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm1 = nn.BatchNorm2d(dec_channels * 4)
        self.conv2 = nn.ConvTranspose2d(dec_channels * 4, dec_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm2 = nn.BatchNorm2d(dec_channels * 2)
        self.conv3 = nn.ConvTranspose2d(dec_channels * 2, dec_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm3 = nn.BatchNorm2d(dec_channels)
        self.conv4 = nn.ConvTranspose2d(dec_channels, 1, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batch_norm2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batch_norm3(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    z_encoder = Encoder(3, 128, True)
    z_decoder = Decoder(3, 128, True)
    z_encoder = z_encoder.to('cuda')
    z_decoder = z_decoder.to('cuda')
    x = torch.randn(1, 3, 128, 128).to('cuda')
    z, _ = z_encoder(x)
    print(z.shape)
    z = z.view(-1, 128 * 8 * 4 * 4)
    print(z.shape)