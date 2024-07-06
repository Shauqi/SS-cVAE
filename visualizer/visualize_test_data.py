import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import yaml
import torch
from torch.utils.data import DataLoader
from dataloader import BRCA_MTL2BIN_File_Loader
from visualizer.visualize_segmentation import visualize_masks, visualize_recons

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_dataloader(config):
    dataset = config['CVAE_MODEL_TRAIN']['dataset']
    test_dir = config['CVAE_MODEL_TRAIN']['in_distribution']['test_dir']
    test_batch_size = config['CVAE_MODEL_TRAIN']['test_batch_size']

    if dataset == 'BRCA_Synth':
        test_ds = BRCA_MTL2BIN_File_Loader(test_dir)

    test_loader = DataLoader(test_ds, batch_size=64, shuffle = False, num_workers = 4)
    return test_loader

def get_model(config):
    model_name = config['CVAE_MODEL_TRAIN']['model_name']
    version_number = config['CVAE_MODEL_TRAIN']['version_number']
    version_name = f'v_{version_number}'
    chkpt_dir = f"{config['CVAE_MODEL_TRAIN']['chkpt_dir']}/{version_name}/epoch={config['CVAE_MODEL_TRAIN']['epoch_number']}.ckpt"
    output_dir = config['CVAE_MODEL_TRAIN']['output_dir']
    background_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['background_disentanglement_penalty']
    salient_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_disentanglement_penalty']
    salient_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size']
    background_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['background_latent_size']
    train_batch_size = config['CVAE_MODEL_TRAIN']['train_batch_size']
    valid_batch_size = config['CVAE_MODEL_TRAIN']['val_batch_size']
    test_batch_size = config['CVAE_MODEL_TRAIN']['test_batch_size']
    num_classes = config['CVAE_MODEL_TRAIN']['num_classes']

    if model_name == 'guided_mmcvae':
        from models.Guided_MM_cVAE import Conv_MM_cVAE
        model = Conv_MM_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, salient_disentanglement_penalty = salient_disentanglement_penalty, background_disentanglement_penalty = background_disentanglement_penalty)
    elif model_name == 'mmcvae_original':
        from models.MM_cVAE import Conv_MM_cVAE
        model = Conv_MM_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, salient_disentanglement_penalty = salient_disentanglement_penalty, background_disentanglement_penalty = background_disentanglement_penalty)
    elif model_name == 'mmcvae_paired':
        from models.MM_cVAE_Paired import Conv_MM_cVAE
        model = Conv_MM_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, salient_disentanglement_penalty = salient_disentanglement_penalty, background_disentanglement_penalty = background_disentanglement_penalty)
    elif model_name == 'mtl_cvae':
        from trainer.mtl_cvae_trainer import MTL_cVAE
        model = MTL_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, background_disentanglement_penalty = background_disentanglement_penalty, salient_disentanglement_penalty = salient_disentanglement_penalty)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

if __name__ == '__main__':
    config = read_yaml('./../configs/config_brca.yaml')
    output_dir = config['CVAE_MODEL_TRAIN']['output_dir']
    test_loader = get_dataloader(config)
    model, device = get_model(config)

    with torch.no_grad():
        for batch_test in test_loader:
            img, labels, _ = batch_test
            img = img.to(device)
            z_mu, logvar_z, F_z, s_mu, logvar_s, F_s = model.encode(img)
            s_recon = model.decode_s(s_mu)
            z_recon = model.decode_z(z_mu)
            visualize_masks(batch_test, s_recon, output_dir, 'segmentation_mask.png')
            # visualize_recons(batch_test, z_recon, output_dir, 'z_recon.png')
            break