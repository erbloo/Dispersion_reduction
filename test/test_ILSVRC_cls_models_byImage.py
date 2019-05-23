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


dataset_dir = "/home/yantao/datasets/ILSVRC1000/"

dataset_dir_ori = os.path.join(dataset_dir, 'original')
dataset_dir_adv = os.path.join(dataset_dir, 'TI')

images_name = os.listdir(dataset_dir_ori)

test_model = torchvision.models.densenet121(pretrained='imagenet').cuda().eval()

success_attacks = 0
for idx, temp_image_name in enumerate(tqdm(images_name)):
    total_samples = len(images_name)
    ori_img_path = os.path.join(dataset_dir_ori, temp_image_name)
    adv_img_path = os.path.join(dataset_dir_adv, temp_image_name)

    image_ori_np = load_image(data_format='channels_first', shape=(224, 224), bounds=(0, 1), abs_path=True, fpath=ori_img_path)
    image_ori_var = numpy_to_variable(image_ori_np)
    gt_out = test_model(image_ori_var).detach().cpu().numpy()
    gt_label = np.argmax(gt_out)
    
    image_adv_np = load_image(data_format='channels_first', shape=(224, 224), bounds=(0, 1), abs_path=True, fpath=adv_img_path)
    image_adv_var = numpy_to_variable(image_adv_np)
    pd_out = test_model(image_adv_var).detach().cpu().numpy()
    pd_label = np.argmax(pd_out)

    linf = int(np.max(abs(image_ori_np - image_adv_np)) * 255)
    print('linf: ', linf)

    if gt_label != pd_label:
        success_attacks += 1

print('attack success rate: ', float(success_attacks) / float(total_samples))