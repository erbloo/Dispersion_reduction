import os
from tqdm import tqdm
from keras import backend as K
from PIL import Image
import numpy as np

from models.yolov3.yolov3_wrapper import YOLOv3
from models.retina_resnet50.keras_retina_resnet50 import KerasResNet50RetinaNetModel
from image_utils import load_image, save_image, save_bbox_img
from mAP import save_detection_to_file, calculate_mAP_from_files

import pdb                         

dataset_dir = "/home/yantao/datasets/ILSVRC1000/"

dataset_dir_ori = os.path.join(dataset_dir, 'original')
dataset_dir_adv = os.path.join(dataset_dir, 'adv_dispersion_opt_08_resnet152')

images_name = os.listdir(dataset_dir_ori)

model = KerasResNet50RetinaNetModel()
#model = YOLOv3(sess = K.get_session())

for idx, temp_image_name in enumerate(tqdm(images_name)):
    temp_image_name_noext = os.path.splitext(temp_image_name)[0]
    ori_img_path = os.path.join(dataset_dir_ori, temp_image_name)
    adv_img_path = os.path.join(dataset_dir_adv, temp_image_name)

    image_ori_np = load_image(data_format='channels_last', shape=(416, 416), bounds=(0, 255), abs_path=True, fpath=ori_img_path)
    Image.fromarray((image_ori_np).astype(np.uint8)).save('./out/ori.jpg')
    image_ori_pil = Image.fromarray(image_ori_np.astype(np.uint8))
    gt_out = model.predict(image_ori_pil)
    
    image_adv_np = load_image(data_format='channels_last', shape=(416, 416), bounds=(0, 255), abs_path=True, fpath=adv_img_path)
    Image.fromarray((image_adv_np).astype(np.uint8)).save('./out/temp_adv.jpg')
    image_adv_pil = Image.fromarray(image_adv_np.astype(np.uint8))
    pd_out = model.predict(image_adv_pil)

    save_detection_to_file(gt_out, os.path.join('out', 'ILSVRC_DispersionAttack_opt_det', 'gt', temp_image_name_noext + '.txt'), 'ground_truth')
    save_detection_to_file(pd_out, os.path.join('out', 'ILSVRC_DispersionAttack_opt_det', 'pd', temp_image_name_noext + '.txt'), 'detection')

    if gt_out:
        save_bbox_img('./out/ori.jpg', gt_out['boxes'], out_file='temp_ori_box.jpg')
    else:
        save_bbox_img('./out/ori.jpg', [], out_file='temp_ori_box.jpg')
    if pd_out:
        save_bbox_img('./out/temp_adv.jpg', pd_out['boxes'], out_file='temp_adv_box.jpg')
    else:
        save_bbox_img('./out/temp_adv.jpg', [], out_file='temp_adv_box.jpg')

mAP_score = calculate_mAP_from_files('out/ILSVRC_DispersionAttack_opt_det/gt', 'out/ILSVRC_DispersionAttack_opt_det/pd')
print('mAP_score: ', mAP_score)