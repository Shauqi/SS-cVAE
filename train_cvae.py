import os
import pytorch_lightning as pl
from utils import set_seeds
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from dataloader import get_datasets

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_trainer(config):
    dataset = config['CVAE_MODEL_TRAIN']['dataset']
    model_name = config['CVAE_MODEL_TRAIN']['model_name']
    debug = config['CVAE_MODEL_TRAIN']['debug']
    resume_from_checkpoint = config['CVAE_MODEL_TRAIN']['resume_from_checkpoint']
    version_number = str(config['CVAE_MODEL_TRAIN']['version_number'])
    version_name = f'v_{version_number}'
    epochs = config['CVAE_MODEL_TRAIN']['max_epochs']
    gpu_list = config['GPU_LIST']
    validation_interval_epoch = 5

    if not debug:
        checkpoint_path = os.path.join(f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['chkpt_dir']}", version_name)
        os.makedirs(checkpoint_path, exist_ok = True)
        output_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['output_dir']}"
        save_path = f'{output_dir}/wandb/{model_name}/{dataset}/{version_name}/'

        logger = WandbLogger(save_dir = output_dir, project = f'{dataset}_{model_name}', version = version_number, name = version_name)
        checkpoint_callback = ModelCheckpoint(dirpath = checkpoint_path, monitor = "val_ss_s", mode = "max", save_top_k = 10, filename='{epoch:02d}')
    else:
        output_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['output_dir']}"
        save_path = f'{output_dir}/wandb/{model_name}/{dataset}/{version_name}/'
        logger = TensorBoardLogger(save_dir=output_dir, version=version_number, name=f'{dataset}_{model_name}')

    if debug:
        trainer = pl.Trainer(fast_dev_run=1, gpus=gpu_list, devices=1, logger = logger, check_val_every_n_epoch=validation_interval_epoch)
    else:
        trainer = pl.Trainer(max_epochs=epochs, gpus=gpu_list, devices=1, logger = logger, callbacks=[checkpoint_callback], check_val_every_n_epoch=validation_interval_epoch)
    
    if resume_from_checkpoint:
        resume_checkpoint_path = f'{checkpoint_path}/epoch=94.ckpt'
        trainer.resume_from_checkpoint = resume_checkpoint_path

    config['CVAE_MODEL_TRAIN']['save_path'] = save_path

    return trainer

def get_model(config):
    model_name = config['CVAE_MODEL_TRAIN']['model_name']
    save_img_epoch = 500
    if model_name != 'double_infogan':
        background_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['background_disentanglement_penalty']
        salient_disentanglement_penalty = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_disentanglement_penalty']
        salient_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['salient_latent_size']
        background_latent_size = config['CVAE_MODEL_TRAIN']['model_parameters']['background_latent_size']

    train_ds, valid_ds, test_ds = get_datasets(config)
    if model_name == 'mm_cvae':
        from models.mm_cvae import MM_cVAE
        model = MM_cVAE(config['CVAE_MODEL_TRAIN'], train_ds = train_ds, valid_ds = valid_ds, test_ds = test_ds)
    elif model_name == 'resnet_cvae':
        from models.resnet_cvae import ResNet_cVAE
        model = ResNet_cVAE(config['CVAE_MODEL_TRAIN'], train_ds = train_ds, valid_ds = valid_ds, test_ds = test_ds)
    elif model_name == 'cvae':
        from models.cvae import cVAE
        model = cVAE(config['CVAE_MODEL_TRAIN'], train_ds = train_ds, valid_ds = valid_ds, test_ds = test_ds)
    elif model_name == 'double_infogan':
        from models.double_InfoGAN import Double_InfoGAN
        model = Double_InfoGAN(config['CVAE_MODEL_TRAIN'], train_ds = train_ds, valid_ds = valid_ds, test_ds = test_ds)
    elif model_name == 'ss_cvae_one_stage_ablation':
        from models.ss_cvae_one_stage_ablation import SS_cVAE
        model = SS_cVAE(config['CVAE_MODEL_TRAIN'], train_ds = train_ds, valid_ds = valid_ds)
    elif model_name == 'ss_cvae_ablation':
        from models.ss_cvae_ablation import SS_cVAE
        model = SS_cVAE(config['CVAE_MODEL_TRAIN'], train_ds = train_ds, valid_ds = valid_ds)

    return model

if __name__ == '__main__':
    set_seeds()
    config = read_yaml('./configs/brca/ss_cvae_ablation.yaml')
    trainer = get_trainer(config)
    model = get_model(config)
    trainer.fit(model)