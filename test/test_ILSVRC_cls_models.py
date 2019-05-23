import torchvision.models as models
import torch
from PIL import Image
from image_utils import load_image, save_image
from torch_utils import numpy_to_variable, variable_to_numpy
from image_utils import numpy_to_bytes
from models.vgg import Vgg16
from models.resnet import Resnet152
from attacks.dispersion import DispersionAttack_opt, DispersionAttack
from attacks.mifgsm import MomentumIteratorAttack
from api_utils import detect_label_numpy
import numpy as np
import torchvision
from api_utils import detect_label_file
from tqdm import tqdm
import os
import shutil
import pdb

# vgg16                          (got images)                          (got images)                        (got images)
# attack success rate       dispersion_opt_14_budget_16         dispersion_opt_3_budget_16          dispersion_opt_25_budget_16
#     inception_v3                0.713(713/1000)                     0.475(475/1000)                      0.645(645/1000)
#     densenet121                 0.654(654/1000)                     0.707(707/1000)                      0.552(552/1000)
#     retina                         18.28                               31.75                                33.97
#     yolo                           33.77                               42.31                                41.49

# resnet152                      (got images)                           (got images)                        (got images)
# attack success rate       dispersion_opt_5_budget_16         dispersion_opt_2_budget_16          dispersion_opt_8_budget_16
#     inception_v3                0.742(742/1000)                     0.463(463/1000)                      0.716(716/1000)
#     densenet121                 0.653                               0.369                                0.585
#     retina                      29.50                               28.32                                20.47
#     yolo                        41.61                               57.33                                38.49


result_file = 'ILSVRC_result.txt'
visited_image_names = []
if os.path.exists(result_file):
    with open(result_file, 'r') as txt_file:
        lines = txt_file.read().split('\n')
        for line in lines:
            visited_image_names.append(line.split(',')[0])
    
with open('labels.txt','r') as inf:
    imagenet_dict = eval(inf.read())

dataset_dir = "/home/yantao/datasets/ILSVRC/Data/DET/test"
images_name = os.listdir(dataset_dir)

#model = Vgg16()
#internal = [i for i in range(29)]
model = Resnet152()
internal = [i for i in range(9)]

attack_layer_idx = 8
layer_type = 'resnet152'

test_model = torchvision.models.inception_v3(pretrained='imagenet').cuda().eval()
#attack = DispersionAttack(model, epsilon=16./255, step_size=1./255, steps=2000, test_api=True)
attack = DispersionAttack_opt(model, epsilon=16./255, steps=2000, is_test_model=True)

#total_samples = len(images_name)
total_samples = 1000

success_attacks = 0
for idx, temp_image_name in enumerate(tqdm(images_name)):
    print('idx: ', idx)
    if idx >= 1000:
        break
    if temp_image_name in visited_image_names:
        print('visited.')
        continue
    temp_image_path = os.path.join(dataset_dir, temp_image_name)

    image_np = load_image(data_format='channels_first', abs_path=True, fpath=temp_image_path)

    #image_pil = Image.fromarray(np.transpose((image_np * 255).astype(np.uint8), (1, 2, 0)))
    #image_pil.save(os.path.join("/home/yantao/datasets/ILSVRC1000/original", temp_image_name))

    image = numpy_to_variable(image_np)

    adv = image

    pred_nat = test_model(adv).detach().cpu().numpy()
    gt_label = np.argmax(pred_nat)

    pred_cls = imagenet_dict[gt_label]
    print(pred_cls)

    adv, info_dict = attack(image, 
                            attack_layer_idx=attack_layer_idx, 
                            internal=internal, 
                            test_steps=500, 
                            gt_label=gt_label,
                            test_model=test_model)

    adv_pil = Image.fromarray(np.transpose((adv[0].detach().numpy() * 255).astype(np.uint8), (1, 2, 0)))
    adv_pil.save(os.path.join("/home/yantao/datasets/ILSVRC1000/adv_dispersion_opt_{0:02d}_".format(attack_layer_idx) + layer_type, temp_image_name))

    if bool(info_dict):
        output_label = info_dict['det_label']
        output_cls = imagenet_dict[output_label]
        print(output_cls)
        if(gt_label != output_label):
            success_attacks += 1
            with open(result_file, 'a') as txt_file:
                txt_file.write(temp_image_name + "," + "success\n")
        else:
            with open(result_file, 'a') as txt_file:
                txt_file.write(temp_image_name + "," + "fail\n")
    else:
        print("Attack failed.")
        with open(result_file, 'a') as txt_file:
            txt_file.write(temp_image_name + "," + "fail\n")
    adv_np = variable_to_numpy(adv)
    linf = int(np.max(abs(image_np - adv_np)) * 255)
    print('linf: ', linf)
    l1 = np.mean(abs(image_np - adv_np)) * 255
    print('l1: ', l1)
    l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
    print('l2: ', l2)
    print(" ")
    print('success_attacks: ', success_attacks)

    

print('attack success rate: ', float(success_attacks) / float(total_samples))
