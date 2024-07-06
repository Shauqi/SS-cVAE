import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import argparse
import torch
import yaml
import numpy as np
from models.MM_cVAE import MM_cVAE, Conv_MM_cVAE
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataloader import TIL_loader, Filtered_TIL_loader
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from PIL import Image
import cv2

def add_labels(input_image, labels):
    """Adds labels next to rows of an image.

    Parameters
    ----------
    input_image : image
        The image to which to add the labels
    labels : list
        The list of labels to plot
    """
    new_width = input_image.width + 100
    new_size = (new_width, input_image.height)
    new_img = Image.new("RGB", new_size, color='white')
    new_img.paste(input_image, (0, 0))
    draw = ImageDraw.Draw(new_img)

    for i, s in enumerate(labels):
        draw.text(xy=(new_width - 100 + 0.005,
                      int((i / len(labels) + 1 / (2 * len(labels))) * input_image.height)),
                  text=s,
                  fill=(0, 0, 0))

    return new_img

ckpt_dir = "/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/checkpoints/classification/MM_cVAE/epoch=101-step=12954.ckpt"

model = Conv_MM_cVAE.load_from_checkpoint(ckpt_dir, background_disentanglement_penalty=10e3, salient_disentanglement_penalty=10e2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

data_dir = "/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/data/classification/TCGA-TILs"
# til_loader = TIL_loader(train_batch_size = 128, valid_batch_size = 128, test_batch_size = 1, num_workers = 4, dataset_dir = data_dir)
til_loader = Filtered_TIL_loader(train_batch_size = 2, valid_batch_size = 2, test_batch_size = 10, num_workers = 4, test_shuffle = True)
train_loader, val_loader, test_loader = til_loader.get_loaders()

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        print(img.shape)
        # img = F.to_pil_image(img)
        # axs[0, i].imshow(np.asarray(img))
        # axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

canvas = np.zeros((66*2,66*10,3))
index = 0
with torch.no_grad():
    for imgs, label in tqdm(test_loader):
        imgs = imgs.to(device)
        z_mu, _, s_mu, _ = model.encode(imgs)
        z_ = torch.cat([z_mu, s_mu], dim=1)
        recons = model.decode(z_)
        imgs = imgs.cpu()
        recons = recons.cpu()
        # to_plot = torch.cat([img.cpu(), recon.cpu()])
        # grid = make_grid(to_plot, nrow = 10)
        # img_grid = grid.mul_(255).clamp_(0, 255).permute(1, 2, 0)
        # # img_grid = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
        # img_grid = img_grid.to('cpu', torch.uint8).numpy()
        # grid_image = Image.fromarray(img_grid)
        # grid_image.save("/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/output/MM_cVAE/Filtered_TIL_23/reconstruction.png")
        break

for index in range(imgs.shape[0]):
    img = imgs[index].numpy().reshape(64,64,3)
    img= cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])
    recon = recons[index].numpy().reshape(64,64,3)
    recon = cv2.copyMakeBorder(recon,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])
    canvas[0:66,66*index:66*(index+1),:] = img
    canvas[66:132,66*index:66*(index+1),:] = recon

#         img = img.cpu().numpy().reshape(64,64,3)
#         recon = recon.cpu().numpy().reshape(64,64,3)
#         print(64*index, 64*(index+1), img.shape)

#         if index >= 9:
#             break
#         index += 1

plt.imsave("/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/output/MM_cVAE/TIL_23/reconstruction.png", canvas)