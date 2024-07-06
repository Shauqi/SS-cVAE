import torch
from gan_training.checkpoints import CheckpointIO
from gan_training.config import (load_config, build_models)
import copy
import glob

def get_model_from_checkpoints(config):
    model_name = config['model_name']
    version_number = config['version_number']
    version_name = f'v_{version_number}'
    chkpt_dir = f"{config['PROJECT_DIR']}{config['chkpt_dir']}/{version_name}/epoch={config['epoch_number']}.ckpt"
    if model_name != 'double_infogan':
        background_disentanglement_penalty = config['model_parameters']['background_disentanglement_penalty']
        salient_disentanglement_penalty = config['model_parameters']['salient_disentanglement_penalty']
        salient_latent_size = config['model_parameters']['salient_latent_size']
        background_latent_size = config['model_parameters']['background_latent_size']
    gpu_list = config['GPU_LIST']

    if model_name == 'chc_vae':
        from models.chc_vae import CHC_VAE
        model = CHC_VAE.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'ch_vae':
        from models.ch_vae import CH_VAE
        model = CH_VAE.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'mm_cvae':
        from models.mm_cvae import MM_cVAE
        model = MM_cVAE.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'ss_cvae':
        from models.ss_cvae import SS_cVAE
        model = SS_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, background_disentanglement_penalty=background_disentanglement_penalty, salient_disentanglement_penalty=salient_disentanglement_penalty)
    elif model_name == 'ss_cvae_ablation':
        from models.ss_cvae_ablation import SS_cVAE
        model = SS_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, background_disentanglement_penalty=background_disentanglement_penalty, salient_disentanglement_penalty=salient_disentanglement_penalty)
    elif model_name == 'resnet_cvae':
        from models.resnet_cvae import ResNet_cVAE
        model = ResNet_cVAE.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'double_infogan':
        from models.double_InfoGAN import Double_InfoGAN
        model = Double_InfoGAN.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'cvae':
        from models.cvae import cVAE
        model = cVAE.load_from_checkpoint(chkpt_dir, config = config)

    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

def get_contrastive_model_from_checkpoints(config):
    model_name = config['CONTRASTIVE_MODEL_TRAIN']['model_name']
    dataset_name = config['CONTRASTIVE_MODEL_TRAIN']['dataset']
    version_number = config['CONTRASTIVE_MODEL_TRAIN']['version_number']
    version_name = f'v_{version_number}'
    gpu_list = config['GPU_LIST']
    chkpt_dir = f"{config['PROJECT_DIR']}{config['CONTRASTIVE_MODEL_TRAIN']['chkpt_dir']}/{dataset_name}/checkpoints/{version_name}/epoch={config['CONTRASTIVE_MODEL_TRAIN']['epoch_number']}.ckpt"
    if model_name == 'contrastive_resnet_cvae':
        from models.contrastive_resnet_cvae import ContrastiveResNetCVAE
        model = ContrastiveResNetCVAE.load_from_checkpoint(chkpt_dir, config = config)

    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

def get_gan_model_from_checkpoints(config):
    checkpoint_dir = f"{config['PROJECT_DIR']}{config['GAN_TRAIN']['CHECKPOINT_DIR']}"
    gpu_list = config['GPU_LIST']
    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
    dvae, generator, discriminator = build_models(config)
    dvae = dvae.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
    checkpoint_io.register_modules(generator=generator, discriminator=discriminator,)

    if config['GAN_TRAIN']['use_model_average']:
        generator_test = copy.deepcopy(generator)
        checkpoint_io.register_modules(generator_test=generator_test)
    else:
        generator_test = generator

    checkpoint_files = glob.glob(f"{checkpoint_dir}/*.pt")
    if len(checkpoint_files) > 0:
        it = checkpoint_io.load('model.pt')

    return generator_test