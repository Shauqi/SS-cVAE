import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import yaml
from models import get_model_from_checkpoints
from dataloader.downstream_loader import MulticlassFileLoader
import random

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

class MClassClassifierCVAE(pl.LightningModule):
    def __init__(self, config, train_ds = None, valid_ds = None):
        super(MClassClassifierCVAE, self).__init__()

        self.config = config
        self.train_ds = train_ds
        self.valid_ds = valid_ds

        self.train_batch_size = self.config['train_batch_size']
        self.valid_batch_size = self.config['val_batch_size']
        self.num_classes = self.config['num_classes']

        self.cvae_model, device = get_model_from_checkpoints(self.config)

        self.z_convs = self.cvae_model.z_convs
        self.s_convs = self.cvae_model.s_convs

        self.fc = nn.Sequential(nn.Linear(128*2, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, self.num_classes))

        self.val_loss = 0.0
        self.correct_predictions = 0
        self.total_samples = 0

    def encode(self, x):
        mu_s, logvar_s = self.s_convs(x)
        mu_z, logvar_z = self.z_convs(x)
        concatenated = torch.concatenate((mu_s, mu_z), dim=1)
        return concatenated
    
    def forward(self, s):
        out = self.fc(s)
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        s = self.encode(x)
        logits = self(s)
        loss = nn.functional.cross_entropy(logits, y)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=True, num_workers=4)
        return dataloader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        s = self.encode(x)
        logits = self(s)
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
        dataloader = DataLoader(self.valid_ds, batch_size=self.valid_batch_size, num_workers=4)
        return dataloader
    
if __name__ == '__main__':
    random.seed(42)
    config = read_yaml('./../configs/config_downstream_cvae_brca_consep.yaml')

    train_dataset_path = config['train_dir']
    train_ds = MulticlassFileLoader(train_dataset_path)
    valid_dataset_path = config['val_dir']
    valid_ds = MulticlassFileLoader(valid_dataset_path)

    model = MClassClassifierCVAE(config, train_ds = train_ds, valid_ds = valid_ds)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # train_loader = model.train_dataloader()
    # for batch in train_loader:
    #     loss = model.training_step(batch, 0)
    #     print(loss)
    #     break

    valid_loader = model.val_dataloader()
    for batch in valid_loader:
        loss, preds, y = model.validation_step(batch, 0)
        print(loss, preds, y)
        break