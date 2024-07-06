import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_dataloader(config):
    test_dir = f"{config['PROJECT_DIR']}{config['CONTRASTIVE_MODEL_TRAIN']['test_dir']}"
    test_batch_size = config['CONTRASTIVE_MODEL_TRAIN']['test_batch_size']

    from dataloader.brca_loader import BRCA_TIL_VS_Other_File_Loader
    test_ds = BRCA_TIL_VS_Other_File_Loader(test_dir)

    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle = False, num_workers = 4)
    return test_loader

def get_model(config):
    model_name = config['CONTRASTIVE_MODEL_TRAIN']['model_name']
    dataset_name = config['CONTRASTIVE_MODEL_TRAIN']['dataset']
    version_number = config['CONTRASTIVE_MODEL_TRAIN']['version_number']
    version_name = f'v_{version_number}'
    gpu_list = config['GPU_LIST']
    chkpt_dir = f"{config['PROJECT_DIR']}{config['CONTRASTIVE_MODEL_TRAIN']['chkpt_dir']}/{dataset_name}/checkpoints/{version_name}/epoch={config['CONTRASTIVE_MODEL_TRAIN']['epoch_number']}.ckpt"
    if model_name == 'contrastive_ss_cvae':
        from models.contrastive_ss_cvae import ContrastiveSSCVAE
        model = ContrastiveSSCVAE.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'contrastive_phi_seg':
        from models.contrastive_phi_seg import ContrastivePhiSeg
        model = ContrastivePhiSeg.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'contrastive_resnet_cvae':
        from models.contrastive_resnet_cvae import ContrastiveResNetCVAE
        model = ContrastiveResNetCVAE.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'contrastive_mm_cvae':
        from models.mm_cvae import MM_cVAE
        model = MM_cVAE.load_from_checkpoint(chkpt_dir, config = config['CONTRASTIVE_MODEL_TRAIN'])
    elif model_name == 'contrastive_cvae':
        from models.cvae import cVAE
        model = cVAE.load_from_checkpoint(chkpt_dir, config = config['CONTRASTIVE_MODEL_TRAIN'])
    elif model_name == 'contrastive_double_infogan':
        from models.double_InfoGAN import Double_InfoGAN
        model = Double_InfoGAN.load_from_checkpoint(chkpt_dir, config = config['CONTRASTIVE_MODEL_TRAIN'])

    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device   

if __name__ == '__main__':
    config = read_yaml('./../configs/consep/double_infogan.yaml')

    model_name = config['CONTRASTIVE_MODEL_TRAIN']['model_name']
    test_loader = get_dataloader(config)
    model, device = get_model(config)

    test_Z = []
    test_S = []
    test_labels = []

    for img, label, mask in test_loader:
        img = img.to(device)
        mask = mask.unsqueeze(1).to(device)
        if model_name == 'contrastive_mm_cvae' or model_name == 'resnet_cvae' or model_name == 'contrastive_cvae':
            z_mu, logvar_z, s_mu, logvar_s = model.encode(img)
            test_Z.extend(z_mu.detach().cpu().numpy())
            test_labels.extend(label)
            test_S.extend(s_mu.detach().cpu().numpy())
        elif model_name == 'chc_vae':
            z_mu, logvar_z, s_mu, logvar_s = model.encode(img, mask)
            test_Z.extend(z_mu.detach().cpu().numpy())
            test_labels.extend(label)
            test_S.extend(s_mu.detach().cpu().numpy())
        elif model_name == 'contrastive_ss_cvae' or model_name == 'contrastive_resnet_cvae':
            s_mu, logvar_s, s_s = model.encode(img)
            test_labels.extend(label)
            test_S.extend(s_mu.detach().cpu().numpy())
        elif model_name == 'contrastive_phi_seg':
            s_mu = model.encode(img, mask)
            test_labels.extend(label)
            test_S.extend(s_mu.detach().cpu().numpy())
        elif model_name == 'contrastive_double_infogan':
            _, _ , z_mu, s_mu = model.discriminator_step(img)
            test_labels.extend(label)
            test_S.extend(s_mu.detach().cpu().numpy())

    test_S = np.array(test_S)
    # test_Z = np.array(test_Z)
    test_labels = np.array(test_labels)