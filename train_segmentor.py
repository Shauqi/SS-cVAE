import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
# from models.phi_seg import PhiSeg
# from models.hae import HAE
from models.unet import UNET

def read_yaml(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def get_datasets(config):
    train_dir = config['SEGMENTATION_MODEL_TRAIN']['in_distribution']['train_dir']
    val_dir = config['SEGMENTATION_MODEL_TRAIN']['in_distribution']['val_dir']
    test_dir = config['SEGMENTATION_MODEL_TRAIN']['in_distribution']['test_dir']

    from dataloader.brca_loader import BRCA_BIN_File_Loader
    train_ds = BRCA_BIN_File_Loader(train_dir)
    valid_ds = BRCA_BIN_File_Loader(val_dir, shuffle=False)
    test_ds = BRCA_BIN_File_Loader(test_dir, shuffle=False)

    return train_ds, valid_ds, test_ds

def get_logger_and_callback(config):
    debug = config['SEGMENTATION_MODEL_TRAIN']['debug']
    dataset = config['SEGMENTATION_MODEL_TRAIN']['dataset']
    model_name = config['SEGMENTATION_MODEL_TRAIN']['model_name']
    output_dir = os.path.join(config['SEGMENTATION_MODEL_TRAIN']['output_dir'], dataset)
    version_number = str(config['SEGMENTATION_MODEL_TRAIN']['version_number'])
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
        callback = ModelCheckpoint(dirpath = checkpoint_path, monitor = "Validation Dice", mode = "max", save_top_k = 20, filename='{epoch:02d}')

    return logger, callback

if __name__ == '__main__':
    config = read_yaml('./configs/config_brca.yaml')
    debug = config['SEGMENTATION_MODEL_TRAIN']['debug']
    validation_epoch_interval = config['SEGMENTATION_MODEL_TRAIN']['validation_epoch_interval']
    train_dataset, val_dataset, test_dataset = get_datasets(config)

    # model = PhiSeg(config, train_dataset, val_dataset)
    model = UNET(config, train_dataset, val_dataset)
    logger, callback = get_logger_and_callback(config)

    if debug:
        trainer = pl.Trainer(max_epochs = 1, logger=logger, gpus=1, fast_dev_run=True)
    else:
        trainer = pl.Trainer(max_epochs = config['SEGMENTATION_MODEL_TRAIN']['max_epochs'], logger=logger, gpus=1, callbacks = callback, check_val_every_n_epoch = validation_epoch_interval)

    trainer.fit(model)