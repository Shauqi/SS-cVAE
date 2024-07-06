import argparse
import os
import sys
from torch import nn
from os import path
import copy

sys.path.append("/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/codes/idgan/")

from utils.helpers import FormatterNoDuplicate, check_bounds, set_seed
from utils.visualize import Visualizer
from utils.viz_helpers import get_samples
from gan_training.config import (load_config, build_models)
from gan_training.inputs import get_dataset
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
import torch

def get_nsamples(data_loader, N):
    samples = torch.stack([data_loader.dataset[i][0] for i in range(N)], dim=0)
    return samples


config = load_config("/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/codes/idgan/test_config.yaml")

PLOT_TYPES = ['generate-samples', 'data-samples', 'reconstruct', "traversals",
              'reconstruct-traverse', "gif-traversals", "all"]


def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    description = "CLI for plotting using pretrained models of `disvae`"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    parser.add_argument('name', type=str,
                        help="Name of the model for storing and loading purposes.")
    parser.add_argument("plots", type=str, nargs='+', choices=PLOT_TYPES,
                        help="List of all plots to generate. `generate-samples`: random decoded samples. `data-samples` samples from the dataset. `reconstruct` first rnows//2 will be the original and rest will be the corresponding reconstructions. `traversals` traverses the most important rnows dimensions with ncols different samples from the prior or posterior. `reconstruct-traverse` first row for original, second are reconstructions, rest are traversals. `gif-traversals` grid of gifs where rows are latent dimensions, columns are examples, each gif shows posterior traversals. `all` runs every plot.")
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed. Can be `None` for stochastic behavior.')
    parser.add_argument('-r', '--n-rows', type=int, default=6,
                        help='The number of rows to visualize (if applicable).')
    parser.add_argument('-c', '--n-cols', type=int, default=7,
                        help='The number of columns to visualize (if applicable).')
    parser.add_argument('-t', '--max-traversal', default=2,
                        type=lambda v: check_bounds(v, lb=0, is_inclusive=False,
                                                    type=float, name="max-traversal"),
                        help='The maximum displacement induced by a latent traversal. Symmetrical traversals are assumed. If `m>=0.5` then uses absolute value traversal, if `m<0.5` uses a percentage of the distribution (quantile). E.g. for the prior the distribution is a standard normal so `m=0.45` corresponds to an absolute value of `1.645` because `2m=90%%` of a standard normal is between `-1.645` and `1.645`. Note in the case of the posterior, the distribution is not standard normal anymore.')
    parser.add_argument('-i', '--idcs', type=int, nargs='+', default=[],
                        help='List of indices to of images to put at the begining of the samples.')
    parser.add_argument('-u', '--upsample-factor', default=1,
                        type=lambda v: check_bounds(v, lb=1, is_inclusive=True,
                                                    type=int, name="upsample-factor"),
                        help='The scale factor with which to upsample the image (if applicable).')
    parser.add_argument('--is-show-loss', action='store_true',
                        help='Displays the loss on the figures (if applicable).')
    parser.add_argument('--is-posterior', action='store_true',
                        help='Traverses the posterior instead of the prior.')
    args = parser.parse_args()

    return args


def main(args):
    """Main function for plotting fro pretrained models.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    set_seed(args.seed)
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda else "cpu")
    out_dir = config['training']['out_dir']
    checkpoint_dir = path.join(out_dir, 'chkpts')
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
    experiment_name = args.name
    dvae, generator, discriminator = build_models(config)
    dvae_dir = "/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/output/idgan/"
    dvae_ckpt_path = os.path.join(dvae_dir, config['dvae']['runname'], 'chkpts', config['dvae']['ckptname'])
    dvae_ckpt = torch.load(dvae_ckpt_path)['model_states']['net']
    dvae.load_state_dict(dvae_ckpt)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Use multiple GPUs if possible
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
    dvae = nn.DataParallel(dvae)

    # Register modules to checkpoint
    checkpoint_io.register_modules(generator=generator, discriminator=discriminator,)

    # Test generator
    if config['test']['use_model_average']:
        generator_test = copy.deepcopy(generator)
        checkpoint_io.register_modules(generator_test=generator_test)
    else:
        generator_test = generator

    # Load checkpoint if existant
    it = checkpoint_io.load('model.pt')
    z_dist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    model_dir = f"{dvae_dir}/idgan_til23_normalized/vis"
    dataset = "til_23_normalized"
    viz = Visualizer(model=dvae, generator=generator_test, dataset=dataset, model_dir=model_dir, z_dist = z_dist, dvae_latent_dim= config['dvae']['c_dim'], save_images=True, loss_of_interest=None,display_loss_per_dim=False, max_traversal=args.max_traversal,  upsample_factor=args.upsample_factor, img_size = (3,64,64))
    size = (args.n_rows, args.n_cols)
    # same samples for all plots: sample max then take first `x`data  for all plots
    num_samples = args.n_cols * args.n_rows
    test_dataset = get_dataset(name=config['data']['type'], data_dir=config['data']['test_dir'], size=config['data']['img_size'],)
    batch_size = config['test']['batch_size']
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=config['training']['nworkers'], shuffle=False, pin_memory=True, sampler=None, drop_last=True)
    samples = get_nsamples(test_loader, num_samples)
    z = z_dist.sample((num_samples,))

    if "all" in args.plots:
        args.plots = [p for p in PLOT_TYPES if p != "all"]

    for plot_type in args.plots:
        if plot_type == 'generate-samples':
            viz.generate_samples(size=size)
        elif plot_type == 'data-samples':
            viz.data_samples(samples, size=size)
        elif plot_type == "reconstruct":
            viz.reconstruct(samples, size=size)
        elif plot_type == 'traversals':
            viz.traversals(data=samples[0:1, ...] if args.is_posterior else None,
                          n_per_latent=args.n_cols,
                          n_latents=args.n_rows,
                          is_reorder_latents=True)
        elif plot_type == "reconstruct-traverse":
            viz.reconstruct_traverse(samples, z,
                                     is_posterior=args.is_posterior,
                                     n_latents=args.n_rows,
                                     n_per_latent=args.n_cols,
                                     is_show_text=args.is_show_loss)
        elif plot_type == "gif-traversals":
            viz.gif_traversals(samples[:args.n_cols, ...], n_latents=args.n_rows)
        else:
            raise ValueError("Unkown plot_type={}".format(plot_type))


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
