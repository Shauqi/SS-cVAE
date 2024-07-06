import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from sklearn.metrics import silhouette_score
from tqdm import tqdm

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_dataloader(config):
    test_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['test_dir']}"
    test_batch_size = config['CVAE_MODEL_TRAIN']['test_batch_size']

    from dataloader.brca_loader import BRCA_BIN_File_Loader
    test_ds = BRCA_BIN_File_Loader(test_dir)

    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle = False, num_workers = 4)
    return test_loader

def get_model(config):
    model_name = config['CVAE_MODEL_TRAIN']['model_name']
    version_number = config['CVAE_MODEL_TRAIN']['version_number']
    version_name = f'v_{version_number}'
    chkpt_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['chkpt_dir']}/{version_name}/epoch={config['CVAE_MODEL_TRAIN']['epoch_number']}.ckpt"

    if model_name != 'double_infogan':        
        background_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['background_disentanglement_penalty']
        salient_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_disentanglement_penalty']
        salient_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size']
        background_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['background_latent_size']

    if model_name == 'mm_cvae':
        from models.mm_cvae import MM_cVAE
        model = MM_cVAE.load_from_checkpoint(chkpt_dir, config = config['CVAE_MODEL_TRAIN'])
    elif model_name == 'resnet_cvae':
        from models.resnet_cvae import ResNet_cVAE
        model = ResNet_cVAE.load_from_checkpoint(chkpt_dir, config = config['CVAE_MODEL_TRAIN'])
    elif model_name == 'cvae':
        from models.cvae import cVAE
        model = cVAE.load_from_checkpoint(chkpt_dir, config = config['CVAE_MODEL_TRAIN'])
    elif model_name == 'double_infogan':
        from models.double_InfoGAN import Double_InfoGAN
        model = Double_InfoGAN.load_from_checkpoint(chkpt_dir, config = config['CVAE_MODEL_TRAIN'])
    elif model_name == 'ss_cvae_one_stage_ablation':
        from models.ss_cvae_one_stage_ablation import SS_cVAE
        model = SS_cVAE.load_from_checkpoint(chkpt_dir, config = config['CVAE_MODEL_TRAIN'])
    elif model_name == 'ss_cvae_ablation':
        from models.ss_cvae_ablation import SS_cVAE
        model = SS_cVAE.load_from_checkpoint(chkpt_dir, config = config['CVAE_MODEL_TRAIN'])

    device = torch.device(f"cuda:{config['GPU_LIST'][0]}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

if __name__ == '__main__':
    config = read_yaml('./configs/brca/ss_cvae_ablation.yaml')
    model_name = config['CVAE_MODEL_TRAIN']['model_name']

    test_loader = get_dataloader(config)
    model, device = get_model(config)

    test_Z = []
    test_S = []
    test_labels = []

    for img, label, mask in tqdm(test_loader):
        img = img.to(device)
        mask = mask.unsqueeze(1).to(device)
        if model_name == 'double_infogan':
            _, _ , z_mu, s_mu = model.discriminator_step(img)
        else:
            z_mu, logvar_z, s_mu, logvar_s = model.encode(img)
    

        test_Z.extend(z_mu.detach().cpu().numpy())
        test_labels.extend(label)
        test_S.extend(s_mu.detach().cpu().numpy())

    test_Z = np.array(test_Z)
    test_S = np.array(test_S)
    test_labels = np.array(test_labels)

    ss_z = silhouette_score(test_Z, test_labels)
    ss_s = silhouette_score(test_S, test_labels)

    print(f"Background Silhoutte Score - {ss_z}")
    print(f"Salient Silhoutte Score - {ss_s}")