import os
import pytorch_lightning as pl
from utils import set_seeds
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from dataloader.downstream_loader import MulticlassFileLoader

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_dataset(config):
    train_ds = MulticlassFileLoader(config['train_dir'])
    valid_ds = MulticlassFileLoader(config['val_dir'])
    return train_ds, valid_ds

def get_model(config, train_ds, valid_ds):
    model_name = config['model_name']
    if model_name == 'downstream_cvae':
        from models.mclass_resnet_cvae import MClassClassifierCVAE
        model = MClassClassifierCVAE(config, train_ds, valid_ds)
    else:
        from models.mclass_resnet import MClassClassifier
        model = MClassClassifier(config, train_ds, valid_ds)
    return model

def get_logger_and_callback(config):
    debug = config['debug']
    dataset = config['dataset']
    model_name = config['model_name']
    output_dir = os.path.join(config['output_dir'], dataset)
    version_number = str(config['version_number'])
    if debug:
        logger = TensorBoardLogger(save_dir=output_dir, version_number=version_number, name=f'{dataset}_{model_name}')
        callback = []
    else:
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok = True)
        version_name = f'v_{version_number}'
        checkpoint_path = f'{checkpoint_dir}/{version_name}'
        os.makedirs(checkpoint_path, exist_ok = True)
        logger = WandbLogger(save_dir=output_dir, project = f'{dataset}_{model_name}', version=version_number, name=version_name)
        callback = ModelCheckpoint(dirpath = checkpoint_path, monitor = "avg_val_acc", mode = "max", save_top_k = 20, filename='{epoch:02d}')

    return logger, callback

if __name__ == '__main__':
    config = read_yaml('./configs/config_downstream_cvae.yaml')['PRETRAINING']
    debug = config['debug']

    train_ds, valid_ds = get_dataset(config)

    model = get_model(config, train_ds, valid_ds)
    logger, callback = get_logger_and_callback(config)

    if debug:
        trainer = pl.Trainer(max_epochs = 1, logger=logger, gpus=1, fast_dev_run=True)
    else:
        trainer = pl.Trainer(max_epochs = config['max_epochs'], logger=logger, gpus=1, callbacks = callback, check_val_every_n_epoch = 5)

    trainer.fit(model)