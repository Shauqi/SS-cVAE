import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils

utils.set_seeds(42)

def read_yaml(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def get_dataloader(config):
    test_dir = config['SEGMENTATION_MODEL_TRAIN']['in_distribution']['test_dir']
    test_batch_size = config['SEGMENTATION_MODEL_TRAIN']['test_batch_size']
    from dataloader.brca_loader import BRCA_BIN_File_Loader
    test_dataset = BRCA_BIN_File_Loader(test_dir)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    return test_dataloader

def get_model(config):
    model_name = config['SEGMENTATION_MODEL_TRAIN']['model_name']
    checkpoint_path = os.path.join(config['SEGMENTATION_MODEL_TRAIN']['output_dir'], config['SEGMENTATION_MODEL_TRAIN']['dataset'], 'checkpoints', 'v_' + str(config['SEGMENTATION_MODEL_TRAIN']['version_number']), 'epoch='+ str(config['SEGMENTATION_MODEL_TRAIN']['epoch_number']) +'.ckpt')
    if model_name == 'unet':
        from models.unet import UNET
        model = UNET.load_from_checkpoint(checkpoint_path, config = config)
    elif model_name == 'phi_seg':
        from models.phi_seg import PhiSeg
        model = PhiSeg.load_from_checkpoint(checkpoint_path, config=config)
    return model

def show_seg(x, mask, pred, output_dir, filename):
    indices = [0,4,7,8]
    fig, ax = plt.subplots(len(indices), 3, figsize=(10, 10))
    for index, i in enumerate(indices):
        x_i = x[i].permute(1,2,0).detach().cpu().numpy()
        pred_i = pred[i].detach().cpu().numpy()
        mask_i = mask[i].detach().cpu().numpy()
        mask_i = np.squeeze(mask_i)
        mask_rgb = np.ones((mask_i.shape[0], mask_i.shape[1], 3)) * 1
        mask_rgb[mask_i == 1] = [1, 0, 0]
        mask_rgb[mask_i == 2] = [0, 0, 1]
        pred_rgb = np.ones((pred_i.shape[0], pred_i.shape[1], 3)) * 1
        pred_rgb[pred_i == 1] = [1, 0, 0]
        pred_rgb[pred_i == 2] = [0, 0, 1]
        ax[index, 0].imshow(x_i)
        ax[index, 1].imshow(mask_rgb)
        ax[index, 2].imshow(pred_rgb)
        
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def show_segmentation_mask(x, y, y_hat, output_dir, filename):
    indices = [0,4,7,8]
    fig, ax = plt.subplots(len(indices), 3, figsize=(10, 10))

    for index, i in enumerate(indices):
        ax[index, 0].imshow(x[i].permute(1, 2, 0).cpu().numpy())
        y_i = np.squeeze(y[i].cpu().numpy())
        y_rgb = np.ones((y_i.shape[0], y_i.shape[1], 3))
        y_rgb[y_i == 1] = [1, 0, 0]
        y_rgb[y_i == 2] = [0, 0, 1]

        y_hat_i = np.squeeze(y_hat[i].cpu().numpy())
        y_hat_rgb = np.ones((y_hat_i.shape[0], y_hat_i.shape[1], 3))
        y_hat_rgb[y_hat_i == 1] = [1, 0, 0]
        y_hat_rgb[y_hat_i == 2] = [0, 0, 1]

        ax[index, 1].imshow(y_rgb)
        ax[index, 2].imshow(y_hat_rgb)

    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

if __name__ == '__main__':
    config = read_yaml('./../configs/config_brca.yaml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = config['SEGMENTATION_MODEL_TRAIN']['model_parameters']['num_classes']
    test_dataloader = get_dataloader(config)
    model = get_model(config)
    model = model.to(device)
    output_dir = os.path.join(config['SEGMENTATION_MODEL_TRAIN']['output_dir'], config['SEGMENTATION_MODEL_TRAIN']['dataset'], 'test_output', 'v_' + str(config['SEGMENTATION_MODEL_TRAIN']['version_number']))
    os.makedirs(output_dir, exist_ok=True)
    
    for img, label, mask in test_dataloader:
        img = img.to(device)
        mask = torch.unsqueeze(mask, 1).to(device)
        mu_post_flattened, pred = model.encode(img, mask)
        prediction_softmax = model.accumulate_output(pred, use_softmax=True)
        predicted_mask = torch.argmax(prediction_softmax, dim=1)
        show_segmentation_mask(img, mask, predicted_mask, output_dir, 'test')
        break