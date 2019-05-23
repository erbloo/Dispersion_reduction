import torch
import torchvision
import numpy as np
import os
from PIL import Image

from image_utils import load_image, save_image, save_bbox_img, numpy_to_bytes
from torch_utils import numpy_to_variable, variable_to_numpy
from api_utils import detect_objects_file, googleDet_to_Dictionary
from models.vgg import Vgg16
from models.resnet import Resnet152
from attacks.dispersion import DispersionAttack_opt, DispersionAttack
from attacks.mifgsm import MomentumIteratorAttack
from attacks.DIM import DIM_Attack
from mAP import save_detection_to_file, calculate_mAP_from_files
from tqdm import tqdm

import pdb

# VGG16
# mAP       dispersion_opt_12       dispersion_opt_14       mi-FGSM(m=0.5)         DIM(m=0.5)       mi-FGSM(m=1.0)      DIM(m=1.0)      TI-DIM
# budget=16       37.57                   32.88                 42.06                40.89               42.62              36.52        33.98
# budget=32       20.48                   16.25                 24.34                32.61               26.06              22.34

# resnet152
# mAP       dispersion_opt_12       dispersion_opt_8       mi-FGSM(m=0.5)         DIM(m=0.5)       mi-FGSM(m=1.0)      DIM(m=1.0)
# budget=16                               33.33                                                         40.95              46.70
# budget=32    


dataset_dir = "/home/yantao/datasets/imagenet_100image/"
images_name = os.listdir(dataset_dir)

'''
#model = Vgg16()
#internal = [i for i in range(29)]

model = Resnet152()
internal = [i for i in range(9)]

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
    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/ori_det.jpg')
    google_label_ori = detect_objects_file('./out/ori_det.jpg')
    google_label_ori = googleDet_to_Dictionary(google_label_ori, adv_np.shape[-2:])
    print(google_label_ori)

    adv, _ = attack(image, 
                    attack_layer_idx=8, 
                    internal=internal
                    )
    adv_np = variable_to_numpy(adv)

    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/temp_adv_det.jpg')
    google_label_pred = detect_objects_file('./out/temp_adv_det.jpg')
    google_label_pred = googleDet_to_Dictionary(google_label_pred, adv_np.shape[-2:])
    print(google_label_pred)

    save_detection_to_file(google_label_ori, os.path.join('out', 'DispersionAttack_opt_det_out', 'gt', temp_image_name_noext + '.txt'), 'ground_truth')
    save_detection_to_file(google_label_pred, os.path.join('out', 'DispersionAttack_opt_det_out', 'pd', temp_image_name_noext + '.txt'), 'detection')

    if google_label_ori:
        save_bbox_img('./out/ori_det.jpg', google_label_ori['boxes'], out_file='temp_ori_box.jpg')
    else:
        save_bbox_img('./out/ori_det.jpg', [], out_file='temp_ori_box.jpg')
    if google_label_pred:
        save_bbox_img('./out/temp_adv_det.jpg', google_label_pred['boxes'], out_file='temp_adv_box.jpg')
    else:
        save_bbox_img('./out/temp_adv_det.jpg', [], out_file='temp_adv_box.jpg')

    linf = int(np.max(abs(image_np - adv_np)) * 255)
    print('linf: ', linf)
    l1 = np.mean(abs(image_np - adv_np)) * 255
    print('l1: ', l1)
    l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
    print('l2: ', l2)

    mAP_score = calculate_mAP_from_files('out/DispersionAttack_opt_det_out/gt', 'out/DispersionAttack_opt_det_out/pd')
'''



#model = torchvision.models.vgg16(pretrained=True).cuda()
model = torchvision.models.resnet152(pretrained=True).cuda()

attack = MomentumIteratorAttack(model, decay_factor=1.0, epsilon=16./255, steps=2000, step_size=1./255, random_start=False)
#attack = DIM_Attack(model, decay_factor=1, prob=0.5, epsilon=16./255, steps=20, step_size=2./255, image_resize=330, random_start=False) #steps=min(epsilon+4, epsilon*1.25)


total_samples = 100
for idx, temp_image_name in enumerate(tqdm(images_name)):
    temp_image_name_noext = os.path.splitext(temp_image_name)[0]
    temp_image_path = os.path.join(dataset_dir, temp_image_name)
    image_np = load_image(data_format='channels_first', abs_path=True, fpath=temp_image_path)
    image = numpy_to_variable(image_np)
    adv = image

    adv_np = variable_to_numpy(adv)
    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/ori.jpg')
    google_label_ori = detect_objects_file('./out/ori.jpg')
    google_label_ori = googleDet_to_Dictionary(google_label_ori, adv_np.shape[-2:])
    print(google_label_ori)

    image_torch_nchw = torch.from_numpy(np.expand_dims(image_np, axis=0)).float()
    pred_nat = model(image_torch_nchw.cuda()).detach().cpu().numpy()
    label = np.argmax(pred_nat)
    label_tensor = torch.tensor(np.array([label]))
    adv = attack(image_torch_nchw, label_tensor)
    adv_np = variable_to_numpy(adv)
    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/temp_adv.jpg')
    google_label_pred = detect_objects_file('./out/temp_adv.jpg')
    google_label_pred = googleDet_to_Dictionary(google_label_pred, adv_np.shape[-2:])
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



