import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2

def visualize_type_mask(image, type_mask, output_dir, file_name, id_to_class, class_with_colors):
    # Extract contours from the type mask using cv2.findContours
    contours, _ = cv2.findContours((type_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours with color for each type
    for i, contour in enumerate(contours):
        class_id = type_mask[contour[0][0][1], contour[0][0][0]]
        class_name = id_to_class[class_id]
        color = class_with_colors[class_name]
        image = cv2.drawContours(image, [contour], -1, color, thickness=5)

    output_file = f"{output_dir}/{file_name}.png"
    # Display the overlayed image
    cv2.imwrite(output_file, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# def visualize_patches():
#     organ = 'brca'
#     data_partition = 'train'
#     data_label = 'negative'
#     dataset_dir = "/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/data/classification/TCGA-TILs"
#     pos_data_path = glob.glob(f"{dataset_dir}/images-tcga-tils/{organ}/{data_partition}/til-{data_label}/*")

#     plt.figure(figsize=[10, 50])
#     fig,axs =  plt.subplots(2,5)
#     index = 1490

#     for data_path in pos_data_path[index:index+10]:
#         image = np.array(Image.open(data_path).convert("RGB"))
#         row = int((index % 10) / 5)
#         col = index % 5
#         axs[row][col].imshow(image)
#         axs[row][col].axis('off')
#         axs[row][col].set_title(f"{index}")
#         index += 1

#     output_path = f"/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/output/MM_cVAE/TIL_23/{organ}"
#     os.makedirs(output_path, exist_ok = True)

#     plt.savefig(f"{output_path}/{data_partition}_{data_label}_data_for_filtering.png")