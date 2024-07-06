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

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MClassClassifierTransfer(pl.LightningModule):
    def __init__(self, config, train_dataset = None, valid_dataset = None):
        super(MClassClassifierTransfer, self).__init__()
        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        num_classes = self.config['num_classes']

        pretrained_config = config['PRETRAINING']

        checkpoint_dir = pretrained_config['chkpt_dir']
        checkpoint_path = os.path.join(checkpoint_dir, pretrained_config['dataset'], 'checkpoints', f"v_{pretrained_config['version_number']}", f"epoch={pretrained_config['epoch_number']}.ckpt")

        from models.mclass_resnet_brca_pretrain import MClassClassifierPretrain
        # pretrained_model = MClassClassifierPretrain.load_from_checkpoint(checkpoint_path, config = pretrained_config)
        pretrained_model = MClassClassifierPretrain(config = pretrained_config)

        self.in_planes = 64
        self.encoder = pretrained_model.encoder
        self.linear = nn.Linear(128, num_classes)

        self.val_loss = 0.0
        self.correct_predictions = 0
        self.total_samples = 0


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        out = self.linear(mu)
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
    config = read_yaml('./../configs/config_downstream_classifier_brca_consep.yaml')
    train_dataset_path = config['train_dir']
    train_ds = MulticlassFileLoader(train_dataset_path)
    valid_dataset_path = config['val_dir']
    valid_ds = MulticlassFileLoader(valid_dataset_path)

    model = MClassClassifierTransfer(config, train_dataset = train_ds, valid_dataset = valid_ds)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = model.train_dataloader()
    for batch in train_loader:
        loss = model.training_step(batch, 0)
        print(loss)
        break

    # valid_loader = model.val_dataloader()
    # for batch in valid_loader:
    #     loss, preds, y = model.validation_step(batch, 0)
    #     print(loss, preds, y)
    #     break
