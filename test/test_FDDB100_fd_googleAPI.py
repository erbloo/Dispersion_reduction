import torch
import torchvision
import numpy as np
import os
from PIL import Image

from image_utils import load_image, save_image, save_bbox_img, numpy_to_bytes
from torch_utils import numpy_to_variable, variable_to_numpy
from api_utils import detect_faces_file
from models.vgg import Vgg16
from models.resnet import Resnet152
from attacks.dispersion import DispersionAttack_opt, DispersionAttack
from attacks.mifgsm import MomentumIteratorAttack
from mAP import save_detection_to_file, calculate_mAP_from_files
from tqdm import tqdm

import pdb

# mAP       dispersion_opt_14       mi-FGSM
# budget=16       
# budget=32       


dataset_dir = "/home/yantao/datasets/FDDB_100image/"
images_name = os.listdir(dataset_dir)


model = Vgg16()
internal = [i for i in range(29)]
#attack = DispersionAttack(model, epsilon=16./255, step_size=1./255, steps=2000, is_test_api=True)
attack = DispersionAttack_opt(model, epsilon=16./255, steps=2000)

total_samples = 100
for idx, temp_image_name in enumerate(tqdm(images_name)):
    temp_image_name_noext = os.path.splitext(temp_image_name)[0]
    temp_image_path = os.path.join(dataset_dir, temp_image_name)
    image_np = load_image(data_format='channels_first', abs_path=True, fpath=temp_image_path)
    image = numpy_to_variable(image_np)
    adv = image

    adv_np = variable_to_numpy(adv)
    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/ori.jpg')
    google_label_ori = detect_faces_file('./out/ori.jpg')
    print(google_label_ori)

    adv, _ = attack(image, 
                    attack_layer_idx=28, 
                    internal=internal
                    )
    adv_np = variable_to_numpy(adv)

    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/temp_adv.jpg')
    google_label_pred = detect_faces_file('./out/temp_adv.jpg')
    print(google_label_pred)

    save_detection_to_file(google_label_ori, os.path.join('out', 'DispersionAttack_opt_det_out', 'gt', temp_image_name_noext + '.txt'), 'ground_truth')
    save_detection_to_file(google_label_pred, os.path.join('out', 'DispersionAttack_opt_det_out', 'pd', temp_image_name_noext + '.txt'), 'detection')

    if google_label_ori:
        save_bbox_img('./out/ori.jpg', google_label_ori['boxes'], out_file='temp_ori_box.jpg')
    else:
        save_bbox_img('./out/ori.jpg', [], out_file='temp_ori_box.jpg')
    if google_label_pred:
        save_bbox_img('./out/temp_adv.jpg', google_label_pred['boxes'], out_file='temp_adv_box.jpg')
    else:
        save_bbox_img('./out/temp_adv.jpg', [], out_file='temp_adv_box.jpg')

    linf = int(np.max(abs(image_np - adv_np)) * 255)
    print('linf: ', linf)
    l1 = np.mean(abs(image_np - adv_np)) * 255
    print('l1: ', l1)
    l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
    print('l2: ', l2)

    mAP_score = calculate_mAP_from_files('out/DispersionAttack_opt_det_out/gt', 'out/DispersionAttack_opt_det_out/pd')



'''
model = torchvision.models.vgg16(pretrained=True).cuda()
attack = MomentumIteratorAttack(model, decay_factor=1.0, epsilon=32./255, steps=2000, step_size=1./255, random_start=False)

total_samples = 100
for idx, temp_image_name in enumerate(tqdm(images_name)):
    temp_image_name_noext = os.path.splitext(temp_image_name)[0]
    temp_image_path = os.path.join(dataset_dir, temp_image_name)
    image_np = load_image(data_format='channels_first', abs_path=True, fpath=temp_image_path)
    image = numpy_to_variable(image_np)
    adv = image

    adv_np = variable_to_numpy(adv)
    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/ori.jpg')
    google_label_ori = detect_faces_file('./out/ori.jpg')
    print(google_label_ori)

    image_torch_nchw = torch.from_numpy(np.expand_dims(image_np, axis=0)).float()
    pred_nat = model(image_torch_nchw.cuda()).detach().cpu().numpy()
    label = np.argmax(pred_nat)
    label_tensor = torch.tensor(np.array([label]))
    adv = attack(image_torch_nchw, label_tensor)
    adv_np = variable_to_numpy(adv)
    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/temp_adv.jpg')
    google_label_pred = detect_faces_file('./out/temp_adv.jpg')
    print(google_label_pred)

    save_detection_to_file(google_label_ori, os.path.join('out', 'DispersionAttack_opt_det_out', 'gt', temp_image_name_noext + '.txt'), 'ground_truth')
    save_detection_to_file(google_label_pred, os.path.join('out', 'DispersionAttack_opt_det_out', 'pd', temp_image_name_noext + '.txt'), 'detection')

    if google_label_ori:
        save_bbox_img('./out/ori.jpg', google_label_ori['boxes'], out_file='temp_ori_box.jpg')
    else:
        save_bbox_img('./out/ori.jpg', [], out_file='temp_ori_box.jpg')
    if google_label_pred:
        save_bbox_img('./out/temp_adv.jpg', google_label_pred['boxes'], out_file='temp_adv_box.jpg')
    else:
        save_bbox_img('./out/temp_adv.jpg', [], out_file='temp_adv_box.jpg')

    linf = int(np.max(abs(image_np - adv_np)) * 255)
    print('linf: ', linf)
    l1 = np.mean(abs(image_np - adv_np)) * 255
    print('l1: ', l1)
    l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
    print('l2: ', l2)

    mAP_score = calculate_mAP_from_files('out/DispersionAttack_opt_det_out/gt', 'out/DispersionAttack_opt_det_out/pd')
'''


