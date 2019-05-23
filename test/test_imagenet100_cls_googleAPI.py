import torchvision.models as models
import torch
from tqdm import tqdm
from PIL import Image
from image_utils import load_image, save_image
from torch_utils import numpy_to_variable, variable_to_numpy
from image_utils import numpy_to_bytes
from models.vgg import Vgg16
from models.resnet import Resnet152
from attacks.dispersion import DispersionAttack_opt, DispersionAttack
from attacks.mifgsm import MomentumIteratorAttack
from attacks.DIM import DIM_Attack
from api_utils import detect_label_numpy
import numpy as np
import torchvision
from api_utils import detect_label_file
import os
import pdb

# Resnet152 [4, 5, 6, 7]
# Vgg16 [2, 7, 14, 21, 28]

#       VGG16
# attack success rate      std_opt_12      std_opt_14      std_12      std_14      mi-fgsm(m=0.5)     DIM(m=0.5)    mi-fgsm(m=1.0)     DIM(m=1.0)       TI-DIM
#           budget=16        0.72             0.77          0.77        0.71         0.56               0.52            0.59              0.61          
#           budget=32        0.98             0.98          0.91        0.92         0.89()             0.64                              0.79
#                                                                                                                                                        0.67(ensemble)

#      Resnet152
# attack success rate      std_opt_12      std_opt_8      std_12      std_14      mi-fgsm(m=0.5)     DIM(m=0.5)    mi-fgsm(m=1.0)     DIM(m=1.0)
#           budget=16                         0.75                                                                      0.53              0.51
#           budget=32       

dataset_dir = "/home/yantao/datasets/imagenet_100image/"
images_name = os.listdir(dataset_dir)


'''
# dispersion(opt) attack

#model = Vgg16()
#internal = [i for i in range(29)]

model = Resnet152()
internal = [i for i in range(9)]

#attack = DispersionAttack(model, epsilon=16./255, step_size=1./255, steps=2000, test_api=True)
attack = DispersionAttack_opt(model, epsilon=16./255, steps=2000, is_test_api=True)

total_samples = 100
success_attacks = 0
for idx, temp_image_name in enumerate(tqdm(images_name)):
    print('idx: ', idx)
    temp_image_path = os.path.join(dataset_dir, temp_image_name)
    image_np = load_image(data_format='channels_first', abs_path=True, fpath=temp_image_path)
    image = numpy_to_variable(image_np)

    adv = image

    adv_np = variable_to_numpy(adv)
    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/ori_cls.jpg')
    google_label = detect_label_file('./out/ori_cls.jpg')

    if len(google_label) > 0:
        pred_cls = google_label[0].description
    else:
        pred_cls = None
    print(pred_cls)

    temp_attack_success = 0
    adv, info_dict = attack(image, 
                            attack_layer_idx=8, 
                            internal=internal, 
                            test_steps=200, 
                            gt_label=pred_cls)
    print(info_dict)

    if bool(info_dict):
        output_cls = info_dict['det_label']
        if(output_cls != pred_cls):
            success_attacks += 1
    else:
        print("Attack failed.")
    
    adv_np = variable_to_numpy(adv)
    linf = int(np.max(abs(image_np - adv_np)) * 255)
    print('linf: ', linf)
    l1 = np.mean(abs(image_np - adv_np)) * 255
    print('l1: ', l1)
    l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
    print('l2: ', l2)

print('attack success rate: ', float(success_attacks) / float(total_samples))
'''


# mi-FGSM/DMI attack

#model = torchvision.models.vgg16(pretrained=True).cuda()
model = torchvision.models.resnet152(pretrained=True).cuda()

attack = MomentumIteratorAttack(model, decay_factor=1.0, epsilon=16./255, steps=2000, step_size=1./255, random_start=False)
#attack = DIM_Attack(model, decay_factor=1.0, prob=0.5, epsilon=16./255, steps=20, step_size=2./255, image_resize=330, random_start=False) #steps=min(epsilon+4, epsilon*1.25)

total_samples = 100
success_attacks = 0
for idx, temp_image_name in enumerate(tqdm(images_name)):
    print('idx: ', idx)
    temp_image_path = os.path.join(dataset_dir, temp_image_name)
    image_np = load_image(data_format='channels_first', abs_path=True, fpath=temp_image_path)
    Image.fromarray(np.transpose((image_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/ori.jpg')
    google_label = detect_label_file('./out/ori.jpg')
    if len(google_label) > 0:
        pred_cls = google_label[0].description
    else:
        pred_cls = None
    print(pred_cls)
    image_torch_nchw = torch.from_numpy(np.expand_dims(image_np, axis=0)).float()
    pred_nat = model(image_torch_nchw.cuda()).detach().cpu().numpy()
    label = np.argmax(pred_nat)
    label_tensor = torch.tensor(np.array([label]))
    adv = attack(image_torch_nchw, label_tensor)
    adv_np = variable_to_numpy(adv)
    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/temp.jpg')
    google_label = detect_label_file('./out/temp.jpg')
    if len(google_label) > 0:
        output_cls = google_label[0].description
    else:
        output_cls = None
    print(output_cls)
    print(" ")
    adv_np = variable_to_numpy(adv)
    linf = int(np.max(abs(image_np - adv_np)) * 255)
    print('linf: ', linf)
    l1 = np.mean(abs(image_np - adv_np)) * 255
    print('l1: ', l1)
    l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
    print('l2: ', l2)
    print(" ")
    if output_cls != pred_cls:
        success_attacks += 1
print('attack success rate: ', float(success_attacks) / float(total_samples))
