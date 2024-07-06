import sys
import os
sys.path.append(os.path.abspath('./../'))
import sys
import yaml
import torch
from medpy.metric import dc
import numpy as np

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

def calculate_batch_dice(predicted_mask, ground_truth_mask, n_classes):
    dice_list = []
    for pred, ground in zip(predicted_mask, ground_truth_mask):
        per_lbl_dice = []
        for lbl in range(n_classes):
            binary_pred = (pred == lbl) * 1
            binary_gt = (ground == lbl) * 1
            if torch.sum(binary_gt) == 0 and torch.sum(binary_pred) == 0:
                per_lbl_dice.append(1.0)
            elif torch.sum(binary_pred) > 0 and torch.sum(binary_gt) == 0 or torch.sum(binary_pred) == 0 and torch.sum(binary_gt) > 0:
                per_lbl_dice.append(0.0)
            else:
                per_lbl_dice.append(dc(binary_pred.detach().cpu().numpy(), binary_gt.detach().cpu().numpy()))
        dice_list.append(per_lbl_dice)

    batch_dice = np.array(dice_list).mean(axis=1)
    return batch_dice


if __name__ == '__main__':
    config = read_yaml('./../configs/config_brca.yaml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = config['SEGMENTATION_MODEL_TRAIN']['model_parameters']['num_classes']
    test_dataloader = get_dataloader(config)
    model = get_model(config)
    model = model.to(device)
    
    total_dice = np.empty((0))
    for img, labels, mask in test_dataloader:
        img = img.to(device)
        mask = mask.unsqueeze(1)
        pred, _, _, _, _ = model(img, mask)
        prediction_softmax = model.accumulate_output(pred, use_softmax=True)
        predicted_mask = torch.argmax(prediction_softmax, dim=1)
        batch_dice = calculate_batch_dice(predicted_mask, mask, num_classes)
        total_dice = np.append(total_dice, batch_dice)

    total_dice = total_dice.mean()
    print(total_dice)