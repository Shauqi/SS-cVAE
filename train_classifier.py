import os
import pytorch_lightning as pl
from utils import set_seeds
from dataloader import BRCA_MTL_File_Loader
import yaml
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_datasets(config):
    dataset = config['CLASSIFIER_MODEL_TRAIN']['dataset']
    train_dir = config['CLASSIFIER_MODEL_TRAIN']['in_distribution']['train_dir']
    val_dir = config['CLASSIFIER_MODEL_TRAIN']['in_distribution']['val_dir']
    test_dir = config['CLASSIFIER_MODEL_TRAIN']['in_distribution']['test_dir']

    if dataset == 'BRCA_Synth':
        train_dataset = BRCA_MTL_File_Loader(train_dir)
        valid_dataset = BRCA_MTL_File_Loader(val_dir)
        test_dataset = BRCA_MTL_File_Loader(test_dir)

    return train_dataset, valid_dataset, test_dataset

def get_logger_and_callback(config, debug):
    dataset = config['CLASSIFIER_MODEL_TRAIN']['dataset']
    model_name = config['CLASSIFIER_MODEL_TRAIN']['model_name']
    debug = config['CLASSIFIER_MODEL_TRAIN']['debug']
    version_number = str(config['CLASSIFIER_MODEL_TRAIN']['version_number'])
    version_name = f'v_{version_number}'
    output_dir = config['CLASSIFIER_MODEL_TRAIN']['output_dir']
    
    if debug:
        logger = TensorBoardLogger(save_dir=output_dir, version_number=version_number, name=f'{dataset}_{model_name}')
        callback = []
    else:
        checkpoint_dir = config['CLASSIFIER_MODEL_TRAIN']['chkpt_dir']
        checkpoint_path = f'{checkpoint_dir}/{version_name}'
        os.makedirs(checkpoint_path, exist_ok = True)
        logger = WandbLogger(save_dir=output_dir, project = f'{dataset}_{model_name}', version=version_number, name=version_name)
        callback = ModelCheckpoint(dirpath = checkpoint_path, monitor = "avg_val_acc", mode = "max", save_top_k = 5, filename='{epoch:02d}')

    return logger, callback

def get_model(config, train_dataset, valid_dataset):
    if config['CVAE_MODEL_TRAIN']['model_name'] == 'binary_classifier' or config['CVAE_MODEL_TRAIN']['model_name'] == 'vae':
        config['CLASSIFIER_MODEL_TRAIN']['latent_size'] = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size'] + config['CVAE_MODEL_TRAIN']['model_parameters']['background_latent_size']
    else:
        config['CLASSIFIER_MODEL_TRAIN']['latent_size'] = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size']

    from trainer.mclass_classifier_cvae_trainer import MClassClassifierCVAETrainer
    version_number = str(config['CVAE_MODEL_TRAIN']['version_number'])
    version_name = f'v_{version_number}'
    chkpt_path = f"{config['CVAE_MODEL_TRAIN']['chkpt_dir']}/{version_name}/epoch={config['CVAE_MODEL_TRAIN']['epoch_number']}.ckpt"
    config['CVAE_MODEL_TRAIN']['chkpt_path'] = chkpt_path
    model = MClassClassifierCVAETrainer(config['CLASSIFIER_MODEL_TRAIN'], config['CVAE_MODEL_TRAIN'], train_dataset, valid_dataset)
    return model


if __name__ == "__main__":
    set_seeds()
    config = read_yaml('./configs/config_brca.yaml')
    debug = config['CLASSIFIER_MODEL_TRAIN']['debug']
    num_epochs = config['CLASSIFIER_MODEL_TRAIN']['max_epochs']

    train_dataset, valid_dataset, test_dataset = get_datasets(config)

    logger, checkpoint_callback = get_logger_and_callback(config, debug)

    if debug:
        trainer = pl.Trainer(max_epochs=1, gpus=1, fast_dev_run=True, logger=logger)
    else:
        trainer = pl.Trainer(max_epochs=num_epochs, gpus=1, logger=logger, callbacks=[checkpoint_callback], log_every_n_steps=10)

    model = get_model(config, train_dataset, valid_dataset)
    trainer.fit(model)