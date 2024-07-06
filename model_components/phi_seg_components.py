import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import utils
from model_components.torchlayers import Conv2D, Conv2DSequence, ReversibleSequence

class DownConvolutionalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, initializers, depth=3, padding=True, pool=True, reversible=False):
        super(DownConvolutionalBlock, self).__init__()

        if depth < 1:
            raise ValueError

        layers = []
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        if reversible:
            layers.append(ReversibleSequence(input_dim, output_dim, reversible_depth=3))
        else:
            layers.append(Conv2D(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))

            if depth > 1:
                for i in range(depth-1):
                    layers.append(Conv2D(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))

        self.layers = nn.Sequential(*layers)

        #self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

class UpConvolutionalBlock(nn.Module):
    """
        A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
        If bilinear is set to false, we do a transposed convolution instead of upsampling
        """

    def __init__(self, input_dim, output_dim, initializers, padding, bilinear=True, reversible=False):
        super(UpConvolutionalBlock, self).__init__()
        self.bilinear = bilinear

        if self.bilinear:
            if reversible:
                self.upconv_layer = ReversibleSequence(input_dim, output_dim, reversible_depth=2)
            else:
                self.upconv_layer = nn.Sequential(
                    Conv2D(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
                    Conv2D(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
                    )

        else:
            raise NotImplementedError

    def forward(self, x, bridge):
        if self.bilinear:
            x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
            x = self.upconv_layer(x)

        assert x.shape[3] == bridge.shape[3]
        assert x.shape[2] == bridge.shape[2]
        out = torch.cat([x, bridge], dim=1)

        return out

class SampleZBlock(nn.Module):
    """
    Performs 2 3X3 convolutions and a 1x1 convolution to mu and sigma which are used as parameters for a Gaussian
    for generating z
    """
    def __init__(self, input_dim, z_dim0=2, depth=2, reversible=False):
        super(SampleZBlock, self).__init__()
        self.input_dim = input_dim

        layers = []

        if reversible:
            layers.append(ReversibleSequence(input_dim, input_dim, reversible_depth=3))
        else:
            for i in range(depth):
                layers.append(Conv2D(input_dim, input_dim, kernel_size=3, padding=1))

        self.conv = nn.Sequential(*layers)

        self.mu_conv = nn.Sequential(nn.Conv2d(input_dim, z_dim0, kernel_size=1))
        self.sigma_conv = nn.Sequential(nn.Conv2d(input_dim, z_dim0, kernel_size=1),
                                        nn.Softplus())

    def forward(self, pre_z):
        pre_z = self.conv(pre_z)
        mu = self.mu_conv(pre_z)
        sigma = self.sigma_conv(pre_z)

        z = mu + sigma * torch.randn_like(sigma, dtype=torch.float32)

        return mu, sigma, z

def increase_resolution(times, input_dim, output_dim):
    """ Increase the resolution by n time for the beginning of the likelihood path"""
    module_list = []
    for i in range(times):
        module_list.append(nn.Upsample(
                    mode='bilinear',
                    scale_factor=2,
                    align_corners=True))
        if i != 0:
            input_dim = output_dim
        module_list.append(Conv2DSequence(input_dim=input_dim, output_dim=output_dim, depth=1))

    return nn.Sequential(*module_list)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, dec_channels=64, latent_size = 50, bias = False):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.latent_size = latent_size
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
        self.mu = nn.Linear(dec_channels * 8 * 8 * 8, latent_size, bias=bias)
        self.var = nn.Linear(dec_channels * 8 * 8 * 8, latent_size, bias=bias)

    def reparameterize(self, mu, log_var):
        #:param mu: mean from the encoder's latent space
        #:param log_var: log variance from the encoder's latent space
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample
    
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
        x = x.view(-1, self.dec_channels * 8 * 8 * 8)
        x_mu = self.mu(x)
        x_var = self.var(x)
        return x_mu, x_var
    
class Decoder(torch.nn.Module):
    def __init__(self, in_channels = 3, dec_channels = 64, latent_size = 50, bias = False):
        super(Decoder, self).__init__()
        self.dec_channels = dec_channels
        self.relu_linear = nn.LeakyReLU(negative_slope=0.2)
        self.linear_decoder = nn.Linear(latent_size, dec_channels * 8 * 8 * 8)
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
        x = self.relu_linear(self.linear_decoder(x))
        x = x.view(-1, self.dec_channels * 8, 8, 8)
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

class Posterior(nn.Module):
    """
    Posterior network of the PHiSeg Module
    For each latent level a sample of the distribution of the latent level is returned

    Parameters
    ----------
    input_channels : Number of input channels, 1 for greyscale,
    is_posterior: if True, the mask is concatenated to the input of the encoder, causing it to be a ConditionalVAE
    """
    def __init__(self, input_channels, num_classes, num_filters, initializers=None, padding=True, is_posterior=True, reversible=False):
        super(Posterior, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.latent_levels = 5
        self.resolution_levels = 7
        self.lvl_diff = self.resolution_levels - self.latent_levels

        self.padding = padding
        self.activation_maps = []

        if is_posterior:
            # increase input channel by two to accomodate place for mask in one hot encoding
            self.input_channels += num_classes

        self.contracting_path = nn.ModuleList()

        for i in range(self.resolution_levels):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            pool = False if i == 0 else True
            self.contracting_path.append(DownConvolutionalBlock(input, output, initializers, depth=3, padding=padding, pool=pool, reversible=reversible))

        self.upsampling_path = nn.ModuleList()

        for i in reversed(range(self.latent_levels)):  # iterates from [latent_levels -1, ... ,0]
            input = 2
            output = self.num_filters[0]*2
            self.upsampling_path.append(UpConvolutionalBlock(input, output, initializers, padding, reversible=reversible))

        self.sample_z_path = nn.ModuleList()
        for i in reversed(range(self.latent_levels)):
            input = 2*self.num_filters[0] + self.num_filters[i + self.lvl_diff]
            if i == self.latent_levels - 1:
                input = self.num_filters[i + self.lvl_diff]
                self.sample_z_path.append(SampleZBlock(input, depth=2, reversible=reversible))
            else:
                self.sample_z_path.append(SampleZBlock(input, depth=2, reversible=reversible))

    def forward(self, patch, segm=None, training_prior=False, z_list=None):
        if segm is not None:
            with torch.no_grad():
                segm_one_hot = utils.convert_batch_to_onehot(segm, nlabels= self.num_classes).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                segm_one_hot = segm_one_hot.float()
            patch = torch.cat([patch, torch.add(segm_one_hot, -0.5)], dim=1)

        blocks = []
        z = [None] * self.latent_levels # contains all hidden z
        sigma = [None] * self.latent_levels
        mu = [None] * self.latent_levels

        x = patch
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        pre_conv = x
        for i, sample_z in enumerate(self.sample_z_path):
            if i != 0:
                pre_conv = self.upsampling_path[i-1](z[-i], blocks[-i])
            mu[-i-1], sigma[-i-1], z[-i-1] = self.sample_z_path[i](pre_conv)
            if training_prior:
                z[-i-1] = z_list[-i-1]

        del blocks

        return z, mu, sigma

class Likelihood(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters, latent_levels=5,
                 resolution_levels=7, image_size=(128,128,1), reversible=False,
                 initializers=None, apply_last_layer=True, padding=True):
        super(Likelihood, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.latent_levels = latent_levels
        self.resolution_levels = resolution_levels
        self.lvl_diff = resolution_levels - latent_levels

        self.image_size = image_size
        self.reversible= reversible

        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        # LIKELIHOOD
        self.likelihood_ups_path = nn.ModuleList()
        self.likelihood_post_ups_path = nn.ModuleList()

        # path for upsampling
        for i in reversed(range(self.latent_levels)):
            input = self.num_filters[i]
            output = self.num_filters[i]
            if reversible:
                self.likelihood_ups_path.append(ReversibleSequence(input_dim=2, output_dim=input, reversible_depth=2))
            else:
                self.likelihood_ups_path.append(Conv2DSequence(input_dim=2, output_dim=input, depth=2))

            self.likelihood_post_ups_path.append(increase_resolution(times=self.lvl_diff, input_dim=input, output_dim=input))

        # path after concatenation
        self.likelihood_post_c_path = nn.ModuleList()
        for i in range(latent_levels - 1):
            input = self.num_filters[i] + self.num_filters[i + 1 + self.lvl_diff]
            output = self.num_filters[i + self.lvl_diff]

            if reversible:
                self.likelihood_post_c_path.append(ReversibleSequence(input_dim=input, output_dim=output, reversible_depth=2))
            else:
                self.likelihood_post_c_path.append(Conv2DSequence(input_dim=input, output_dim=output, depth=2))

        self.s_layer = nn.ModuleList()
        output = self.num_classes
        for i in reversed(range(self.latent_levels)):
            input = self.num_filters[i + self.lvl_diff]
            self.s_layer.append(Conv2DSequence(
                input_dim=input, output_dim=output, depth=1, kernel=1, activation=torch.nn.Identity, norm=torch.nn.Identity))

    def forward(self, z):
        """Likelihood network which takes list of latent variables z with dimension latent_levels"""
        s = [None] * self.latent_levels
        post_z = [None] * self.latent_levels
        post_c = [None] * self.latent_levels

        # start from the downmost layer and the last filter
        for i in range(self.latent_levels):
            assert z[-i-1].shape[1] == 2
            assert z[-i-1].shape[2] == self.image_size[1] * 2**(-self.resolution_levels + 1 + i)
            post_z[-i - 1] = self.likelihood_ups_path[i](z[-i - 1])
            post_z[-i - 1] = self.likelihood_post_ups_path[i](post_z[-i - 1])
            assert post_z[-i - 1].shape[2] == self.image_size[1] * 2 ** (-self.latent_levels + i + 1)
            assert post_z[-i-1].shape[1] == self.num_filters[-i-1 - self.lvl_diff], '{} != {}'.format(post_z[-i-1].shape[1],self.num_filters[-i-1])

        post_c[self.latent_levels - 1] = post_z[self.latent_levels - 1]

        for i in reversed(range(self.latent_levels - 1)):
            ups_below = nn.functional.interpolate(post_c[i+1], mode='bilinear', scale_factor=2, align_corners=True)

            assert post_z[i].shape[3] == ups_below.shape[3]
            assert post_z[i].shape[2] == ups_below.shape[2]

            # Reminder: Pytorch standard is NCHW, TF NHWC
            concat = torch.cat([post_z[i], ups_below], dim=1)

            post_c[i] = self.likelihood_post_c_path[i](concat)

        for i, block in enumerate(self.s_layer):
            s_in = block(post_c[-i-1]) # no activation in the last layer
            s[-i-1] = torch.nn.functional.interpolate(s_in, size=[self.image_size[1], self.image_size[2]], mode='nearest')

        return s



if __name__ == '__main__':
    # z_posterior = Recon_Posterior(input_channels=3, num_classes=4, num_filters=[32,64,128,192,192,192,192], initializers=None, padding=True, is_posterior=True)
    # z_likelihood = Recon_Likelihood(input_channels=3, num_classes=4, num_filters=[32,64,128,192,192,192,192], image_size=(3,128,128), initializers=None, padding=True)
    # z_posterior = z_posterior.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # z_likelihood = z_likelihood.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # x = torch.rand(1, 3, 128, 128).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # mask = torch.rand(1, 1, 128, 128).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # z, mu, sigma = z_posterior(x)
    # s = z_likelihood(z)

    z_conv_encoder = Encoder(in_channels=3, dec_channels=32, latent_size = 50, bias = False)
    z_conv_decoder = Decoder(in_channels = 3, dec_channels = 32, latent_size = 50, bias = False)
    z_conv_encoder = z_conv_encoder.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    z_conv_decoder = z_conv_decoder.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    x = torch.rand(1, 3, 128, 128).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    z_mu, z_sigma = z_conv_encoder(x)
    z = z_conv_encoder.reparameterize(z_mu, z_sigma)
    x_hat = z_conv_decoder(z)
    print(x_hat.shape)