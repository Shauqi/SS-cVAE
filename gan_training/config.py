import yaml
from torch import optim
from gan_training.models import generator_dict, discriminator_dict
from gan_training.train import toogle_grad

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_models(config):
    # Get classes
    Generator = generator_dict[config['GAN_TRAIN']['generator']['name']]
    Discriminator = discriminator_dict[config['GAN_TRAIN']['discriminator']['name']]
    dvae_model_name = config['CVAE_MODEL_TRAIN']['model_name']
    dVAE_version_number = config['CVAE_MODEL_TRAIN']['version_number']
    dVAE_version_name = f'v_{dVAE_version_number}'
    dVAE_epoch_number = config['CVAE_MODEL_TRAIN']['epoch_number']
    dvae_ckpt_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['chkpt_dir']}/{dVAE_version_name}/epoch={dVAE_epoch_number}.ckpt"
    background_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['background_disentanglement_penalty']
    salient_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_disentanglement_penalty']
    salient_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size']
    background_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['background_latent_size']

    if dvae_model_name == 'mm_cvae':
        from models.mm_cvae import MM_cVAE
        dvae = MM_cVAE.load_from_checkpoint(dvae_ckpt_dir, config = config)
    elif dvae_model_name == 'ch_vae':
        from models.ch_vae import CH_VAE
        dvae = CH_VAE.load_from_checkpoint(dvae_ckpt_dir, config = config)
    elif dvae_model_name == 'resnet_cvae':
        from models.resnet_cvae import ResNet_cVAE
        dvae = ResNet_cVAE.load_from_checkpoint(dvae_ckpt_dir, config = config['CVAE_MODEL_TRAIN'])
    
    generator = Generator(
        z_dim = config['CVAE_MODEL_TRAIN']['model_parameters']['background_latent_size']+config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size'],
        size=config['GAN_TRAIN']['img_size'],
        **config['GAN_TRAIN']['generator']['kwargs']
    )

    discriminator = Discriminator(
        config['GAN_TRAIN']['discriminator']['name'], size=config['GAN_TRAIN']['img_size'],
        **config['GAN_TRAIN']['discriminator']['kwargs']
    )

    return dvae, generator, discriminator


def build_optimizers(generator, discriminator, dvae, config):
    optimizer = config['GAN_TRAIN']['optimizer']
    lr_g = config['GAN_TRAIN']['lr_g']
    lr_d = config['GAN_TRAIN']['lr_d']
    equalize_lr = config['GAN_TRAIN']['equalize_lr']

    toogle_grad(generator, True)
    toogle_grad(discriminator, True)

    if equalize_lr:
        g_gradient_scales = getattr(generator, 'gradient_scales', dict())
        d_gradient_scales = getattr(discriminator, 'gradient_scales', dict())

        g_params = get_parameter_groups(generator.parameters(), g_gradient_scales, base_lr=lr_g)
        d_params = get_parameter_groups(discriminator.parameters(), d_gradient_scales, base_lr=lr_d)
    else:
        g_params = generator.parameters()
        d_params = discriminator.parameters()

    # Optimizers
    if optimizer == 'rmsprop':
        g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
        d_optimizer = optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)
    elif optimizer == 'adam':
        g_optimizer = optim.Adam(g_params, lr=lr_g, betas=(0., 0.99), eps=1e-8)
        d_optimizer = optim.Adam(d_params, lr=lr_d, betas=(0., 0.99), eps=1e-8)
    elif optimizer == 'sgd':
        g_optimizer = optim.SGD(g_params, lr=lr_g, momentum=0.)
        d_optimizer = optim.SGD(d_params, lr=lr_d, momentum=0.)

    return g_optimizer, d_optimizer


def build_lr_scheduler(optimizer, config, last_epoch=-1):
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['GAN_TRAIN']['lr_anneal_every'],
        gamma=config['GAN_TRAIN']['lr_anneal'],
        last_epoch=last_epoch
    )
    return lr_scheduler


# Some utility functions
def get_parameter_groups(parameters, gradient_scales, base_lr):
    param_groups = []
    for p in parameters:
        c = gradient_scales.get(p, 1.)
        param_groups.append({
            'params': [p],
            'lr': c * base_lr
        })
    return param_groups
