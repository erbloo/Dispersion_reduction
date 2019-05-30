import sys
sys.path.append('/home/yantao/workspace/projects/baidu/Dispersion_reduction')
import os
import warnings
import torch
import numpy as np

from attacks.dispersion import DispersionAttack_gpu
from models.vgg import Vgg16
from models.resnet import Resnet152
from utils.image_utils import load_images, save_images

import pdb        


def main():
    model = Vgg16()
    internal = [i for i in range(29)]
    #model = Resnet152()
    #internal = [i for i in range(9)]

    attack_layer_idx = 14
    model_name = model.get_name()

    adversary = DispersionAttack_gpu(model, 
                                     epsilon=16/255., 
                                     step_size=2/255., 
                                     steps=20,
                                     )
    
    images_t, file_name_list = load_images(dir_path='images', size=[224, 224])
    images_var = images_t.cuda()
    advs_var = adversary(images_var, attack_layer_idx=attack_layer_idx, internal=internal)
    advs_t = advs_var.cpu()
    save_images(advs_t, dir_path='outputs', file_name_list=file_name_list)

if __name__ =="__main__":
    main()