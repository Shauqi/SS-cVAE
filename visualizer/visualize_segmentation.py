import matplotlib.pyplot as plt
import numpy as np

def visualize_masks(batch, masks, output_dir, filename):
    x, labels, _ = batch
    backgrounds = x[labels == 0]
    targets = x[labels != 0]

    background_masks = masks[labels == 0]
    target_masks = masks[labels != 0]

    min_index_len = min(len(backgrounds), len(targets))
    if min_index_len > 4:
        min_index_len = 4
    backgrounds = backgrounds[:min_index_len]
    targets = targets[:min_index_len]

    background_masks = background_masks[:min_index_len]
    target_masks = target_masks[:min_index_len]

    backgrounds = backgrounds.permute(0,2,3,1).detach().cpu().numpy()
    targets = targets.permute(0,2,3,1).detach().cpu().numpy()

    background_masks = background_masks.permute(0,2,3,1).detach().cpu().numpy()
    target_masks = target_masks.permute(0,2,3,1).detach().cpu().numpy()

    fig, axs = plt.subplots(min_index_len, 4, figsize=(4 * 10, min_index_len * 10))

    for row in range(min_index_len):
        axs[row, 0].imshow(np.squeeze(backgrounds[row]))
        axs[row, 0].axis('off')

        axs[row, 1].imshow(np.squeeze(background_masks[row]))
        axs[row, 1].axis('off')

        axs[row, 2].imshow(np.squeeze(targets[row]))
        axs[row, 2].axis('off')

        axs[row, 3].imshow(np.squeeze(target_masks[row]))
        axs[row, 3].axis('off')

    plt.savefig(f"{output_dir}/{filename}")
    plt.close()


def visualize_recons(batch, recons, output_dir, filename):
    x, labels, _ = batch
    backgrounds = x[labels == 0]
    targets = x[labels != 0]

    background_recons = recons[labels == 0]
    target_recons = recons[labels != 0]

    min_index_len = min(len(backgrounds), len(targets))
    if min_index_len > 4:
        min_index_len = 4
    backgrounds = backgrounds[:min_index_len]
    targets = targets[:min_index_len]

    background_recons = background_recons[:min_index_len]
    target_recons = target_recons[:min_index_len]

    backgrounds = backgrounds.permute(0,2,3,1).detach().cpu().numpy()
    targets = targets.permute(0,2,3,1).detach().cpu().numpy()

    background_recons = background_recons.permute(0,2,3,1).detach().cpu().numpy()
    target_recons = target_recons.permute(0,2,3,1).detach().cpu().numpy()

    fig, axs = plt.subplots(min_index_len, 4, figsize=(4 * 10, min_index_len * 10))

    for row in range(min_index_len):
        axs[row, 0].imshow(np.squeeze(backgrounds[row]))
        axs[row, 0].axis('off')

        axs[row, 1].imshow(np.squeeze(background_recons[row]))
        axs[row, 1].axis('off')

        axs[row, 2].imshow(np.squeeze(targets[row]))
        axs[row, 2].axis('off')

        axs[row, 3].imshow(np.squeeze(target_recons[row]))
        axs[row, 3].axis('off')

    plt.savefig(f"{output_dir}/{filename}")
    plt.close()