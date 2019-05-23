import torchvision.models as models
import torch
from tqdm import tqdm
from PIL import Image
from image_utils import load_image, save_image
from torch_utils import numpy_to_variable, variable_to_numpy
from image_utils import numpy_to_bytes
from api_utils import detect_label_numpy
import numpy as np
import torchvision
from api_utils import detect_label_file
import os
import pdb    

ori_dir = "/home/yantao/datasets/imagenet_100image/"
adv_dir = "/home/yantao/workspace/projects/baidu/Translation-Invariant-Attacks/output_images/"
images_name = os.listdir(ori_dir)

success_attacks_cls = 0
for idx, temp_image_name in enumerate(tqdm(images_name)):
    total_samples = len(images_name)
    print('idx: ', idx)
    temp_ori_image_path = os.path.join(ori_dir, temp_image_name)
    temp_adv_image_path = os.path.join(adv_dir, temp_image_name)

    google_label = detect_label_file(temp_ori_image_path)
    if len(google_label) > 0:
        pred_cls = google_label[0].description
    else:
        pred_cls = None
    print(pred_cls)

    google_label = detect_label_file(temp_adv_image_path)
    if len(google_label) > 0:
        output_cls = google_label[0].description
    else:
        output_cls = None
    print(output_cls)

    if output_cls != pred_cls:
        success_attacks_cls += 1
        
print('cls attack success rate: ', float(success_attacks_cls) / float(total_samples))
