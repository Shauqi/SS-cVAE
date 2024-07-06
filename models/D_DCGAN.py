import torch.nn as nn
import torch
import functools

class GaussianNoise(nn.Module):                         # Try noise just for real or just for fake images.
    def __init__(self, std=0.1, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate
        print('std : ', self.std)

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.std ==0 or not self.training: 
            return x 
        else: 
            return x + torch.empty_like(x).normal_(std=self.std)


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        #nc = 3
        ndf = 64

        self.GaussNoise = GaussianNoise(std=args['discriminator']['std'])
        self.conv1 = nn.Conv2d(self.args['in_channels'], ndf, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(ndf * 16)
        self.conv6 = nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.class_conv = nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)
        self.z_conv = nn.Conv2d(ndf * 16, args['background_latent_size'], 4, 1, 0, bias=False)
        self.s_conv = nn.Conv2d(ndf * 16, args['salient_latent_size'], 4, 1, 0, bias=False)

        # if self.args["latent_distribution"]=="uniform":
        #     self.act_fn = nn.Sigmoid()
        # elif self.args["latent_distribution"]=="normal":
        #     self.act_fn = nn.Tanh()
        # elif self.args["latent_distribution"]=="log_normal":
        #     self.act_fn = nn.ReLU()
        # else:
        #     raise NotImplementedError('latent distibution [%s] is not found' % args['latent_distribution'])
   
    def forward(self, input):
        out = self.lrelu(self.conv1(self.GaussNoise(input)))
        out = self.lrelu(self.bn2(self.conv2(self.GaussNoise(out))))
        out = self.lrelu(self.bn3(self.conv3(self.GaussNoise(out))))
        out = self.lrelu(self.bn4(self.conv4(self.GaussNoise(out))))
        out = self.lrelu(self.bn5(self.conv5(self.GaussNoise(out))))
        adv = self.conv6(self.GaussNoise(out))
        adv = adv.view(adv.shape[0], -1)
        validity = self.sigmoid(adv)

        classe = self.class_conv(self.GaussNoise(out))
        classe = classe.view(classe.shape[0], -1)
        classe = self.sigmoid(classe)

        z_pred = self.z_conv(self.GaussNoise(out))
        z_pred = z_pred.view(z_pred.shape[0], -1)
        #z_pred = self.act_fn(z_pred)

        s_pred = self.s_conv(self.GaussNoise(out))
        s_pred = s_pred.view(s_pred.shape[0], -1)
        #s_pred = self.act_fn(s_pred)

        return validity, classe, z_pred, s_pred

