import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import torch
import yaml
from dataloader import BRCA_MTL_File_Loader, BRCA_MTL2BIN_Paired_File_Loader, BRCA_MTL2BIN_File_Loader
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_dataloader(config):
    in_distribution_flag = config['CLASSIFIER_MODEL_TRAIN']['in_distribution_flag']
    dataset = config['CLASSIFIER_MODEL_TRAIN']['dataset']
    if in_distribution_flag:
        test_dir = config['CLASSIFIER_MODEL_TRAIN']['in_distribution']['test_dir']
    else:
        test_dir = config['CLASSIFIER_MODEL_TRAIN']['out_distribution']['test_dir']

    test_batch_size = config['CLASSIFIER_MODEL_TRAIN']['test_batch_size']

    if dataset == 'BRCA_Synth':
        test_dataset = BRCA_MTL_File_Loader(test_dir)

    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=4)

    return test_loader

def perform_classification(data, labels):
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

    print(f"Average 5-Fold CV Accuracy: {np.mean(accuracies)}+/-{np.std(accuracies)}")

def get_model(config, checkpoint_path):
    if config['CVAE_MODEL_TRAIN']['model_name'] == 'binary_classifier' or config['CVAE_MODEL_TRAIN']['model_name'] == 'vae':
        config['CLASSIFIER_MODEL_TRAIN']['latent_size'] = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size'] + config['CVAE_MODEL_TRAIN']['model_parameters']['background_latent_size']
    else:
        config['CLASSIFIER_MODEL_TRAIN']['latent_size'] = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size']

    version_number = str(config['CVAE_MODEL_TRAIN']['version_number'])
    version_name = f'v_{version_number}'
    chkpt_path = f"{config['CVAE_MODEL_TRAIN']['chkpt_dir']}/{version_name}/epoch={config['CVAE_MODEL_TRAIN']['epoch_number']}.ckpt"
    config['CVAE_MODEL_TRAIN']['chkpt_path'] = chkpt_path
    from trainer.mclass_classifier_cvae_trainer import MClassClassifierCVAETrainer
    model = MClassClassifierCVAETrainer.load_from_checkpoint(checkpoint_path, classifier_hparams = config['CLASSIFIER_MODEL_TRAIN'], cvae_hparams = config['CVAE_MODEL_TRAIN'])

    return model

if __name__ == '__main__':
    config = read_yaml('./../configs/config_brca.yaml')
    version_number = config['CLASSIFIER_MODEL_TRAIN']['version_number']
    version_name = f'v_{version_number}'
    checkpoint_path = f"{config['CLASSIFIER_MODEL_TRAIN']['chkpt_dir']}/{version_name}/epoch={config['CLASSIFIER_MODEL_TRAIN']['epoch_number']}.ckpt"

    # Initialize your ClassifierTrainer class
    model = get_model(config, checkpoint_path)
    model.eval()  # Set the model to evaluation mode

    test_loader = get_dataloader(config)

    # Initialize lists to store predictions and ground truth labels
    predictions = []
    true_labels = []

    # Iterate through the test data and make predictions
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            outputs = model.encode(inputs)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    perform_classification(predictions, true_labels)