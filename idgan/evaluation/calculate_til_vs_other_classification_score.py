import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import BRCA_MTL2BIN_File_Loader
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_dataloader(config):
    in_distribution_flag = config['CVAE_MODEL_TRAIN']['in_distribution_flag']
    if in_distribution_flag:
        test_dir = config['TIL_VS_OTHER_PATCH_PREPROCESSING']['test_dir']
    else:
        test_dir = None
    test_batch_size = config['CVAE_MODEL_TRAIN']['test_batch_size']

    from dataloader.brca_loader import BRCA_TIL_VS_Other_File_Loader
    test_ds = BRCA_TIL_VS_Other_File_Loader(test_dir)

    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle = False, num_workers = 4)
    return test_loader

def get_model(config):
    model_name = config['CVAE_MODEL_TRAIN']['model_name']
    version_number = config['CVAE_MODEL_TRAIN']['version_number']
    version_name = f'v_{version_number}'
    chkpt_dir = f"{config['CVAE_MODEL_TRAIN']['chkpt_dir']}/{version_name}/epoch={config['CVAE_MODEL_TRAIN']['epoch_number']}.ckpt"
    background_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['background_disentanglement_penalty']
    salient_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_disentanglement_penalty']
    salient_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size']
    background_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['background_latent_size']

    if model_name == 'mtl_cvae':
        from trainer.mtl_cvae_trainer import MTL_cVAE
        model = MTL_cVAE.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, background_disentanglement_penalty = background_disentanglement_penalty, salient_disentanglement_penalty = salient_disentanglement_penalty)
    elif model_name == 'chc_vae':
        from models.chc_vae_v2 import CHC_VAE
        model = CHC_VAE.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'mm_cvae':
        from trainer.mm_cvae_trainer import Conv_MM_cVAE
        model = Conv_MM_cVAE.load_from_checkpoint(chkpt_dir, config = config)
    elif model_name == 'mtl_cvae_ablation':
        from trainer.mtl_cvae_ablation_trainer import MTL_cVAE_ABLATION_TRAINER
        model = MTL_cVAE_ABLATION_TRAINER.load_from_checkpoint(chkpt_dir, salient_latent_size = salient_latent_size, background_latent_size = background_latent_size, salient_disentanglement_penalty = salient_disentanglement_penalty, background_disentanglement_penalty = background_disentanglement_penalty)
    elif model_name == 'binary_classifier':
        from trainer.binary_classifier_trainer import BinaryClassifierTrainer
        model = BinaryClassifierTrainer.load_from_checkpoint(chkpt_dir, hparams = config['CVAE_MODEL_TRAIN'])
    elif model_name == 'vae':
        from trainer.vae_trainer import VAETrainer
        model = VAETrainer.load_from_checkpoint(chkpt_dir, hparams = config['CVAE_MODEL_TRAIN'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    config = read_yaml('./../configs/config_brca.yaml')

    model_name = config['CVAE_MODEL_TRAIN']['model_name']
    test_loader = get_dataloader(config)
    model, device = get_model(config)

    test_Z = []
    test_S = []
    test_labels = []

    for img, label, mask in test_loader:
        img = img.to(device)
        mask = mask.unsqueeze(1).to(device)
        if model_name == 'mtl_cvae' or model_name == 'mmcvae_paired':
            z_mu, logvar_z, F_z, s_mu, logvar_s, F_s = model.encode(img)
            test_Z.extend(z_mu.detach().cpu().numpy())
            test_labels.extend(label)
            test_S.extend(s_mu.detach().cpu().numpy())
        elif model_name == 'mm_cvae':
            z_mu, logvar_z, s_mu, logvar_s = model.encode(img)
            test_Z.extend(z_mu.detach().cpu().numpy())
            test_labels.extend(label)
            test_S.extend(s_mu.detach().cpu().numpy())
        elif model_name == 'binary_classifier':
            z_mu = model(img)
            test_Z.extend(z_mu.detach().cpu().numpy())
            test_labels.extend(label)
        elif model_name == 'vae':
            z_mu, logvar_z, recon, class_logits = model(img)
            test_Z.extend(z_mu.detach().cpu().numpy())
            test_labels.extend(label)
        elif model_name == 'chc_vae':
            z_mu, logvar_z, s_mu, logvar_s = model.encode(img, mask)
            test_Z.extend(z_mu.detach().cpu().numpy())
            test_labels.extend(label)
            test_S.extend(s_mu.detach().cpu().numpy())

    test_S = np.array(test_S)
    test_Z = np.array(test_Z)
    test_labels = np.array(test_labels)
    
    if model_name != 'binary_classifier' and model_name != 'vae':
        perform_classification(test_S, test_labels, latent_space_name = 'Salient')
    
    perform_classification(test_Z, test_labels, latent_space_name = 'Background')