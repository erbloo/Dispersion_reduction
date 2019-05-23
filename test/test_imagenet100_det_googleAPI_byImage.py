import torch
import torchvision
import numpy as np
import os
from PIL import Image

from api_utils import detect_objects_file, googleDet_to_Dictionary
from mAP import save_detection_to_file, calculate_mAP_from_files
from tqdm import tqdm

import pdb

ori_dir = "/home/yantao/datasets/imagenet_100image/"
adv_dir = "/home/yantao/workspace/projects/baidu/Translation-Invariant-Attacks/output_images/"
images_name = os.listdir(ori_dir)

for idx, temp_image_name in enumerate(tqdm(images_name)):
    total_samples = len(images_name)
    print('idx: ', idx)
    temp_ori_image_path = os.path.join(ori_dir, temp_image_name)
    temp_adv_image_path = os.path.join(adv_dir, temp_image_name)
    temp_image_name_noext = os.path.splitext(temp_image_name)[0]

    google_label_ori = detect_objects_file(temp_ori_image_path)
    google_label_ori = googleDet_to_Dictionary(google_label_ori, (224, 224))
    print(google_label_ori)

    google_label_pred = detect_objects_file(temp_adv_image_path)
    google_label_pred = googleDet_to_Dictionary(google_label_pred, (224, 224))
    print(google_label_pred)

    save_detection_to_file(google_label_ori, os.path.join('out', 'TIDIM_det_out', 'gt', temp_image_name_noext + '.txt'), 'ground_truth')
    save_detection_to_file(google_label_pred, os.path.join('out', 'TIDIM_det_out', 'pd', temp_image_name_noext + '.txt'), 'detection')

    mAP_score = calculate_mAP_from_files('out/TIDIM_det_out/gt', 'out/TIDIM_det_out/pd')



