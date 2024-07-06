import argparse
import os
import yaml
import pytorch_lightning as pl
from models.double_InfoGAN import Double_InfoGAN
from torch.utils.data import DataLoader
from dataloader.brca_loader import BRCA_BIN_File_Loader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from utils import set_seeds
import random

def read_config(config_file):
    yaml_file = open(config_file, 'r')
    config = yaml.safe_load(yaml_file)
    return config

def get_datasets(config):
    train_dir = f"{config['PROJECT_DIR']}{config['train_dir']}"
    valid_dir = f"{config['PROJECT_DIR']}{config['valid_dir']}"
    test_dir = f"{config['PROJECT_DIR']}{config['test_dir']}"
    train_batch_size = config['train_batch_size']
    valid_batch_size = config['valid_batch_size']
    test_batch_size = config['test_batch_size']

    train_ds = BRCA_BIN_File_Loader(data_dir = train_dir)
    valid_ds = BRCA_BIN_File_Loader(data_dir = valid_dir, shuffle = True)
    test_ds = BRCA_BIN_File_Loader(data_dir = test_dir, shuffle = True)

    train_loader = DataLoader(train_ds, batch_size = train_batch_size, shuffle = True, num_workers = 4)
    valid_loader = DataLoader(valid_ds, batch_size = valid_batch_size, shuffle = False, num_workers = 4)
    test_loader = DataLoader(test_ds, batch_size = test_batch_size, shuffle = False, num_workers = 4)

    batch_test = next(iter(test_loader))
    return train_loader, valid_loader, batch_test

def get_model(config, batch_test, save_path):
    model = Double_InfoGAN(config=config, batch_test=batch_test)
    return model

def get_logger_and_callbacks(config):
    debug = config['debug']
    model_name = config['model_name']
    dataset = config['dataset']
    output_dir = f"{config['PROJECT_DIR']}{config['output_dir']}"
    os.makedirs(output_dir, exist_ok = True)
    version_number = str(config['version_number'])
    version_name = f'v_{version_number}'
    checkpoint_path = f'{output_dir}/checkpoints/{version_name}'
    if debug:
        logger = TensorBoardLogger(save_dir=output_dir, version_number=version_number, name=model_name)
        checkpoint_callback = []
    else:
        logger = WandbLogger(save_dir=output_dir, project = f"{dataset}_{model_name}", version = version_number, name = version_name)
        checkpoint_callback = ModelCheckpoint(dirpath = checkpoint_path, monitor = "val_ss_s", mode = "max", save_top_k = 10, filename='{epoch:02d}')

    return logger, checkpoint_callback


if __name__ == "__main__":
    config = read_config('./configs/brca/double_infogan.yaml')['CVAE_MODEL_TRAIN']

    if config["seed"] == "None":
        seed = random.randint(1,1000)
        config["seed"] = seed
    else:
        seed = config["seed"]

    set_seeds(seed)

    epochs = config['n_epoch']
    output_dir = f"{config['PROJECT_DIR']}{config['output_dir']}"
    os.makedirs(output_dir, exist_ok = True)
    version_number = str(config['version_number'])
    version_name = f'v_{version_number}'
    debug = config['debug']
    model_name = config['model_name']
    gpu_list = config['GPU_LIST']
    save_path = f"{output_dir}/valid/{version_name}/"

    train_loader, val_loader, batch_test = get_datasets(config)

    model = get_model(config, batch_test, save_path)

    logger, checkpoint_callback = get_logger_and_callbacks(config)

    if debug:
        trainer = pl.Trainer(max_epochs=1, gpus=gpu_list, fast_dev_run=True, logger = logger)
    else:
        trainer = pl.Trainer(max_epochs=epochs, gpus=gpu_list, logger = logger, callbacks=[checkpoint_callback], check_val_every_n_epoch=5)

    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)