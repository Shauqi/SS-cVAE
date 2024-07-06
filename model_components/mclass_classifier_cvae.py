import torch
import torch.nn as nn


class MClassClassifierCVAE(nn.Module):
    def __init__(self, latent_size, num_classes, cvae_hparams, in_channels = 3, dec_channels = 32, bias = False):
        super(MClassClassifierCVAE, self).__init__()
        if cvae_hparams['model_name'] == 'mmcvae_original':
            self.cvae_model_name = 'mmcvae_original'
            from models.MM_cVAE import Conv_MM_cVAE
            self.cvae = Conv_MM_cVAE.load_from_checkpoint(cvae_hparams['chkpt_path'], salient_latent_size = cvae_hparams['model_parameters']['salient_latent_size'], background_latent_size = cvae_hparams['model_parameters']['background_latent_size'], salient_disentanglement_penalty = cvae_hparams['model_parameters']['salient_disentanglement_penalty'], background_disentanglement_penalty = cvae_hparams['model_parameters']['background_disentanglement_penalty'])
        elif cvae_hparams['model_name'] == 'mtl_cvae':
            self.cvae_model_name = 'mtl_cvae'
            from trainer.mtl_cvae_trainer import MTL_cVAE
            self.cvae = MTL_cVAE.load_from_checkpoint(cvae_hparams['chkpt_path'], salient_latent_size = cvae_hparams['model_parameters']['salient_latent_size'], background_latent_size = cvae_hparams['model_parameters']['background_latent_size'], salient_disentanglement_penalty = cvae_hparams['model_parameters']['salient_disentanglement_penalty'], background_disentanglement_penalty = cvae_hparams['model_parameters']['background_disentanglement_penalty'])
        elif cvae_hparams['model_name'] == 'binary_classifier':
            self.cvae_model_name = 'binary_classifier'
            from trainer.binary_classifier_trainer import BinaryClassifierTrainer
            self.cvae = BinaryClassifierTrainer.load_from_checkpoint(cvae_hparams['chkpt_path'], hparams = cvae_hparams)
        elif cvae_hparams['model_name'] == 'vae':
            self.cvae_model_name = 'vae'
            from trainer.vae_trainer import VAETrainer
            self.cvae = VAETrainer.load_from_checkpoint(cvae_hparams['chkpt_path'], hparams = cvae_hparams)
        self.fc6 = nn.Linear(latent_size, 128)  # Add a fully connected layer
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(128, num_classes)
        self.softmax8 = nn.Softmax(dim=1)

    def encode(self, x):
        if self.cvae_model_name == 'mmcvae_original':
            _, _, x, _ = self.cvae.encode(x)
        elif self.cvae_model_name == 'mtl_cvae':
            _, _, _, x, _, _ = self.cvae.encode(x)
        elif self.cvae_model_name == 'binary_classifier':
            x = self.cvae.encode(x)
        elif self.cvae_model_name == 'vae':
            x, _ = self.cvae.model.encode(x)
        x = self.fc6(x)
        x = self.relu6(x)
        return x

    def forward(self, x):
        if self.cvae_model_name == 'mmcvae_original':
            _, _, x, _ = self.cvae.encode(x)
        elif self.cvae_model_name == 'mtl_cvae':
            _, _, _, x, _, _ = self.cvae.encode(x)
        elif self.cvae_model_name == 'binary_classifier':
            x = self.cvae.encode(x)
        elif self.cvae_model_name == 'vae':
            x, _ = self.cvae.model.encode(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.fc7(x)
        x = self.softmax8(x)
        return x