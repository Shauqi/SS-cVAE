import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
import os

def save_images(imgs, outfile):
    # imgs = imgs / 2 + 0.5     # unnormalize
    # torchvision.utils.save_image(imgs, outfile, nrow=nrow)
    print("Saving Images")
    imgs = imgs.cpu().numpy().reshape(imgs.shape[0], 64, 64, 3)
    plt.figure(figsize=[20, imgs.shape[0]*20])
    fig, axs = plt.subplots(1, imgs.shape[0])
    for i in tqdm(range(imgs.shape[0])):
        axs[i].imshow(imgs[i])
        axs[i].axis('off')
    plt.savefig(outfile)


def get_nsamples(data_loader, N):
    x = []
    y = []
    n = 0
    while n < N:
        x_next, y_next = next(iter(data_loader))
        x.append(x_next)
        y.append(y_next)
        n += x_next.size(0)
    x = torch.cat(x, dim=0)[:N]
    y = torch.cat(y, dim=0)[:N]
    return x, y


def update_average(model_tgt, model_src, beta):
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


def mmd(x, y, gammas, device):
    gammas = gammas.to(device)
    cost = torch.mean(gram_matrix(x, x, gammas=gammas)).to(device)
    cost += torch.mean(gram_matrix(y, y, gammas=gammas)).to(device)
    cost -= 2 * torch.mean(gram_matrix(x, y, gammas=gammas)).to(device)
    if cost < 0:
        return torch.tensor(0).to(device)
    return cost


def gram_matrix(x, y, gammas):
    gammas = gammas.unsqueeze(1)
    pairwise_distances = torch.cdist(x, y, p=2.0)

    pairwise_distances_sq = torch.square(pairwise_distances)
    tmp = torch.matmul(gammas, torch.reshape(pairwise_distances_sq, (1, -1)))
    tmp = torch.reshape(torch.sum(torch.exp(-tmp), 0), pairwise_distances_sq.shape)
    return tmp

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert_to_onehot(lblmap, nlabels):
    output = np.zeros((lblmap.shape[0], lblmap.shape[1], nlabels))
    for ii in range(nlabels):
        output[:, :, ii] = (lblmap == ii).astype(np.uint8)
    return output

# needs a torch tensor as input instead of numpy array
# accepts format HW and CHW
def convert_to_onehot_torch(lblmap, nlabels):
    if len(lblmap.shape) == 3:
        # 2D image
        output = torch.zeros((nlabels, lblmap.shape[-2], lblmap.shape[-1]))
        for ii in range(nlabels):
            lbl = (lblmap == ii).view(lblmap.shape[-2], lblmap.shape[-1])
            output[ii, :, :] = lbl
    elif len(lblmap.shape) == 4:
        # 3D images from brats are already one hot encoded
        output = lblmap
    return output.long()

def convert_batch_to_onehot(lblbatch, nlabels):
    out = []
    for ii in range(lblbatch.shape[0]):
        lbl = convert_to_onehot_torch(lblbatch[ii,...], nlabels)
        # TODO: check change
        out.append(lbl.unsqueeze(dim=0))
    result = torch.cat(out, dim=0)
    return result

def get_color(class_num):
    if class_num == 0:
        return [255, 255, 255]
    elif class_num == 1:
        return [255, 0, 0]
    elif class_num == 2:
        return [0, 255, 0]
    elif class_num == 3:
        return [0, 0, 255]