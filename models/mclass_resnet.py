import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import yaml
import random
from dataloader.downstream_loader import Paired_BIN_File_Loader, File_Loader, MulticlassFileLoader
import torch.nn.functional as F
from model_components.ResNetModules import ResNetBlock, ResNetEncoder

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

class MClassClassifier(pl.LightningModule):
    def __init__(self, config, train_dataset = None, valid_dataset = None):
        super(MClassClassifier, self).__init__()
        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        num_classes = self.config['num_classes']

        self.encoder = ResNetEncoder()
        self.fc = nn.Linear(128, num_classes)

        self.val_loss = 0.0
        self.correct_predictions = 0
        self.total_samples = 0

    def forward(self, x):
        mu, var = self.encoder(x)
        out = self.fc(mu)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config['learning_rate'], momentum=self.config['momentum'])
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.config['train_batch_size'], shuffle=True, num_workers=4)
        return dataloader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)  # Get the predicted class indices
        acc = torch.sum(preds == y).item() / len(y)
        return loss, preds, y

    def validation_epoch_end(self, outputs):
        # Calculate the average validation loss
        avg_val_loss = sum([loss for loss, _, _ in outputs]) / len(outputs)
        
        # Calculate the accuracy
        for _, preds, y in outputs:
            self.correct_predictions += torch.sum(preds == y).item()
            self.total_samples += len(y)

        avg_val_accuracy = self.correct_predictions / self.total_samples
        
        self.log('avg_val_loss', avg_val_loss, prog_bar=True)
        self.log('avg_val_acc', avg_val_accuracy, prog_bar=True)
        
        # Reset counters for the next validation epoch
        self.correct_predictions = 0
        self.total_samples = 0

    def val_dataloader(self):
        dataloader = DataLoader(self.valid_dataset, batch_size=self.config['val_batch_size'], num_workers=4)
        return dataloader
    

if __name__ == '__main__':
    random.seed(42)
    config = read_yaml('./../configs/config_downstream_classifier.yaml')
    train_dataset_path = config['train_dir']
    train_ds = MulticlassFileLoader(train_dataset_path)
    valid_dataset_path = config['val_dir']
    valid_ds = MulticlassFileLoader(valid_dataset_path)

    model = MClassClassifier(config, train_dataset = train_ds, valid_dataset = valid_ds)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # train_loader = model.train_dataloader()
    # for batch in train_loader:
    #     loss = model.training_step(batch, 0)
    #     print(loss)
    #     break

    # valid_loader = model.val_dataloader()
    # for batch in valid_loader:
    #     loss, preds, y = model.validation_step(batch, 0)
    #     print(loss, preds, y)
    #     break
