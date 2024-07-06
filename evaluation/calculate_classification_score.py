import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
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
    gpu_list = config['GPU_LIST']
    version_number = config['CVAE_MODEL_TRAIN']['version_number']
    version_name = f'v_{version_number}'
    chkpt_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['chkpt_dir']}/{version_name}/epoch={config['CVAE_MODEL_TRAIN']['epoch_number']}.ckpt"

    if model_name != 'double_infogan': 
        background_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['background_disentanglement_penalty']
        salient_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_disentanglement_penalty']
        salient_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size']
        background_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['background_latent_size']

    if model_name == 'chc_vae':
        from models.chc_vae import CHC_VAE
        model = CHC_VAE.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'ch_vae':
        from models.ch_vae import CH_VAE
        model = CH_VAE.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'mm_cvae':
        from models.mm_cvae import MM_cVAE
        model = MM_cVAE.load_from_checkpoint(chkpt_dir, config = config['CVAE_MODEL_TRAIN'])
    elif model_name == 'ss_cvae':
        from models.ss_cvae import SS_cVAE
        model = SS_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, background_disentanglement_penalty=background_disentanglement_penalty, salient_disentanglement_penalty=salient_disentanglement_penalty)
    elif model_name == 'ss_cvae_ablation':
        from models.ss_cvae_ablation import SS_cVAE
        model = SS_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, background_disentanglement_penalty=background_disentanglement_penalty, salient_disentanglement_penalty=salient_disentanglement_penalty)
    elif model_name == 'resnet_cvae':
        from models.resnet_cvae import ResNet_cVAE
        model = ResNet_cVAE.load_from_checkpoint(chkpt_dir, config = config['CVAE_MODEL_TRAIN'])
    elif model_name == 'cvae':
        from models.cvae import cVAE
        model = cVAE.load_from_checkpoint(chkpt_dir, config = config['CVAE_MODEL_TRAIN'])
    elif model_name == 'double_infogan':
        from models.double_InfoGAN import Double_InfoGAN
        model = Double_InfoGAN.load_from_checkpoint(chkpt_dir, config = config['CVAE_MODEL_TRAIN'])

    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device
    
def perform_classification(data, labels, latent_space_name = 'Salient'):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in skf.split(data, labels):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Train a Logistic Regression model
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    print(f"Average 5-Fold CV Accuracy for {latent_space_name}: {np.mean(accuracies)}+/-{np.std(accuracies)}")

if __name__ == '__main__':
    config = read_yaml('./../configs/consep/double_infogan.yaml')

    model_name = config['CVAE_MODEL_TRAIN']['model_name']
    test_loader = get_dataloader(config)
    model, device = get_model(config)

    test_Z = []
    test_S = []
    test_labels = []

    for img, label, mask in tqdm(test_loader):
        img = img.to(device)
        mask = mask.unsqueeze(1).to(device)
        if model_name == 'ss_cvae' or model_name == 'ss_cvae_ablation':
            z_mu, logvar_z, F_z, s_mu, logvar_s, F_s = model.encode(img)
            test_Z.extend(z_mu.detach().cpu().numpy())
            test_labels.extend(label)
            test_S.extend(s_mu.detach().cpu().numpy())
        elif model_name == 'mm_cvae' or model_name == 'resnet_cvae' or model_name == 'cvae':
            z_mu, logvar_z, s_mu, logvar_s = model.encode(img)
            test_Z.extend(z_mu.detach().cpu().numpy())
            test_labels.extend(label)
            test_S.extend(s_mu.detach().cpu().numpy())
        elif model_name == 'chc_vae' or model_name == 'ch_vae':
            z_mu, logvar_z, s_mu, logvar_s = model.encode(img, mask)
            test_Z.extend(z_mu.detach().cpu().numpy())
            test_labels.extend(label)
            test_S.extend(s_mu.detach().cpu().numpy())
        elif model_name == 'double_infogan':
            _, _ , z_mu, s_mu = model.discriminator_step(img)
            test_Z.extend(z_mu.detach().cpu().numpy())
            test_labels.extend(label)
            test_S.extend(s_mu.detach().cpu().numpy())

    test_S = np.array(test_S)
    test_Z = np.array(test_Z)
    test_labels = np.array(test_labels)
    
    if model_name != 'binary_classifier' and model_name != 'vae':
        perform_classification(test_S, test_labels, latent_space_name = 'Salient')
    
    perform_classification(test_Z, test_labels, latent_space_name = 'Background')