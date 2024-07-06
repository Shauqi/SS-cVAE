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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_dataloader(config):
    test_dir = config['CONTRASTIVE_MODEL_TRAIN']['in_distribution']['val_dir']
    test_batch_size = config['CONTRASTIVE_MODEL_TRAIN']['test_batch_size']

    from dataloader.brca_loader import BRCA_TIL_VS_Other_File_Loader
    test_ds = BRCA_TIL_VS_Other_File_Loader(test_dir)

    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle = False, num_workers = 4)
    return test_loader

def get_model(config):
    version_number = config['CONTRASTIVE_MODEL_TRAIN']['version_number']
    version_name = f'v_{version_number}'
    chkpt_path = os.path.join(config['CONTRASTIVE_MODEL_TRAIN']['output_dir'], config['CONTRASTIVE_MODEL_TRAIN']['dataset'], 'checkpoints', version_name, 'epoch='+ str(config['CONTRASTIVE_MODEL_TRAIN']['epoch_number']) +'.ckpt')

    from models.contrastive_phi_seg import ContrastivePhiSeg
    model = ContrastivePhiSeg.load_from_checkpoint(chkpt_path, config = config)
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

    test = []
    test_labels = []

    for img, label, mask in tqdm(test_loader):
        img = img.to(device)
        mask = mask.unsqueeze(1).to(device)
        z = model.encode(img, mask)

        test.extend(z.detach().cpu().numpy())
        test_labels.extend(label)

    test = np.array(test)
    test_labels = np.array(test_labels)

    perform_classification(test, test_labels, latent_space_name = 'Salient')