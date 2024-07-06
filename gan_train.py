import random
import numpy as np
import argparse
import os
from os import path
from tqdm import tqdm
import time
import copy
import torch
from torch import nn
from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import DisentEvaluator
from gan_training.config import (load_config, build_models, build_optimizers, build_lr_scheduler,)
from dataloader import BRCA_GAN_File_Loader, BRCA_BIN_File_Loader
from torch.utils.data import DataLoader

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

config_path = './configs/config_brca_resnet_cvae.yaml'
config = load_config(config_path)
is_cuda = torch.cuda.is_available()

# = = = = = Customized Configurations = = = = = #
checkpoint_dir = config['GAN_TRAIN']['CHECKPOINT_DIR']
max_iter = config['GAN_TRAIN']['max_iter']

# Short hands
batch_size = config['GAN_TRAIN']['batch_size']
d_steps = config['GAN_TRAIN']['d_steps']
restart_every = config['GAN_TRAIN']['restart_every']
inception_every = config['GAN_TRAIN']['inception_every']
save_every = config['GAN_TRAIN']['save_every']
backup_every = config['GAN_TRAIN']['backup_every']
train_batch_size = config['GAN_TRAIN']['train_batch_size']
valid_batch_size = config['GAN_TRAIN']['val_batch_size']
test_batch_size = config['GAN_TRAIN']['test_batch_size']

out_dir = config['GAN_TRAIN']['OUTPUT_DIR']

# Create missing directories
os.makedirs(out_dir, exist_ok = True)
os.makedirs(checkpoint_dir, exist_ok = True)

# Logger
checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
device = torch.device("cuda:0" if is_cuda else "cpu")

train_dir = config['GAN_TRAIN']['train_dir']
val_dir = config['GAN_TRAIN']['val_dir']
test_dir = config['GAN_TRAIN']['test_dir']

train_ds = BRCA_BIN_File_Loader(train_dir)
valid_ds = BRCA_BIN_File_Loader(val_dir)
test_ds = BRCA_BIN_File_Loader(test_dir)

train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle = True, num_workers = 4)
val_loader = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle = False, num_workers = 4)
test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle = False, num_workers = 4)

# Create models
dvae, generator, discriminator = build_models(config)
# Put models on gpu if needed
dvae = dvae.to(device)
generator = generator.to(device)
discriminator = discriminator.to(device)

g_optimizer, d_optimizer = build_optimizers(generator, discriminator, dvae, config)

# Register modules to checkpoint
checkpoint_io.register_modules(generator=generator, discriminator=discriminator, g_optimizer=g_optimizer,d_optimizer=d_optimizer,)

# Logger
logger = Logger(log_dir=path.join(out_dir, 'logs'), img_dir=path.join(out_dir, 'imgs'), monitoring=config['GAN_TRAIN']['monitoring'], monitoring_dir=path.join(out_dir, 'monitoring'))

# Distributions
zdist = get_zdist(config['GAN_TRAIN']['z_dist']['type'], config['GAN_TRAIN']['z_dist']['dim'], device=device)

# Save for tests
ntest = config['CVAE_MODEL_TRAIN']['test_batch_size']

x_next, y_next, mask_next = next(iter(test_loader))
x_test = torch.cat([x_next[:3], x_next[-3:]], dim = 0)
y_test = torch.cat([y_next[:3], y_next[-3:]], dim = 0)
mask_next = mask_next.unsqueeze(1)
mask_test = torch.cat([mask_next[:3], mask_next[-3:]], dim = 0)

x_test = x_test.to(device)
mask_test = mask_test.to(device)

# Test generator
if config['GAN_TRAIN']['take_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Train
tstart = t0 = time.time()
it = epoch_idx = -1

# Load checkpoint if existant
it = checkpoint_io.load('model.pt')
if it != -1:
    logger.load_stats('stats.p')

# Reinitialize model average if needed
if (config['GAN_TRAIN']['take_model_average']
        and config['GAN_TRAIN']['model_average_reinit']):
    update_average(generator_test, generator, 0.)

# Learning rate anneling
g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)

# Trainer
trainer = Trainer(dvae, generator, discriminator, g_optimizer, d_optimizer, reg_param=config['GAN_TRAIN']['reg_param'], w_info = config['GAN_TRAIN']['w_info'])

# Training loop
tqdm.write('Start training...')
pbar = tqdm(total=max_iter)
if it > 0:
    pbar.update(it)

out = False
while not out:
    epoch_idx += 1
    tqdm.write('Start epoch %d...' % epoch_idx)

    for x_real, _, x_mask in train_loader:
        it += 1
        pbar.update(1)
        g_scheduler.step()
        d_scheduler.step()

        d_lr = d_optimizer.param_groups[0]['lr']
        g_lr = g_optimizer.param_groups[0]['lr']
        logger.add('learning_rates', 'discriminator', d_lr, it=it)
        logger.add('learning_rates', 'generator', g_lr, it=it)

        x_real = x_real.to(device)
        x_mask = x_mask.unsqueeze(1).to(device)

        # Discriminator updates
        # z = zdist.sample((batch_size,))
        dloss, reg, cs = trainer.discriminator_trainstep(x_real)
        logger.add('losses', 'discriminator', dloss, it=it)
        logger.add('losses', 'regularizer', reg, it=it)

        # Generators updates
        if ((it + 1) % d_steps) == 0:
            # z = zdist.sample((batch_size,))
            gloss, encloss = trainer.generator_trainstep(cs)
            logger.add('losses', 'generator', gloss, it=it)
            logger.add('losses', 'encoder', encloss, it=it)

            if config['GAN_TRAIN']['take_model_average']:
                update_average(generator_test, generator,
                              beta=config['GAN_TRAIN']['model_average_beta'])

        # Print stats
        g_loss_last = logger.get_last('losses', 'generator')
        d_loss_last = logger.get_last('losses', 'discriminator')
        e_loss_last = logger.get_last('losses', 'encoder')
        d_reg_last = logger.get_last('losses', 'regularizer')
        tqdm.write('[epoch %0d, it %4d] g_loss = %.4f, d_loss = %.4f, e_loss = %.4f, reg=%.4f'
              % (epoch_idx, it, g_loss_last, d_loss_last, e_loss_last, d_reg_last))

        if (it % config['GAN_TRAIN']['sample_every']) == 0:
            tqdm.write('Creating samples...')
            # z_test = zdist.sample((x_test.shape[0],))
            # z_test = z_test.to(device)
            trainer.validation_step(x_test, path.join(out_dir, 'imgs'), it)

        # (iii) Backup if necessary
        if ((it + 1) % backup_every) == 0:
            tqdm.write('Saving backup...')
            checkpoint_io.save(it, 'model_%08d.pt' % it)
            logger.save_stats('stats_%08d.p' % it)
            checkpoint_io.save(it, 'model.pt')
            logger.save_stats('stats.p')

        # (iv) Save checkpoint if necessary
        if time.time() - t0 > save_every:
            tqdm.write('Saving checkpoint...')
            checkpoint_io.save(it, 'model.pt')
            logger.save_stats('stats.p')
            t0 = time.time()

            if (restart_every > 0 and t0 - tstart > restart_every):
                exit(3)

        if it >= max_iter:
            tqdm.write('Saving backup...')
            checkpoint_io.save(it, 'model_%08d.pt' % it)
            logger.save_stats('stats_%08d.p' % it)
            checkpoint_io.save(it, 'model.pt')
            logger.save_stats('stats.p')
            out = True
            break