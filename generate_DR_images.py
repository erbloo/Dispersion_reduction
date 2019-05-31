import sys
sys.path.append('/home/yantao/workspace/projects/baidu/Dispersion_reduction')
import os
import torch
import numpy as np
from tqdm import tqdm

from attacks.dispersion import DispersionAttack_gpu
from models.vgg import Vgg16
from models.resnet import Resnet152
from models.inception_v3 import Inception_v3
from utils.image_utils import load_images, save_images

import pdb        


def main():
    params = {
        'batch_size' : 8,
        'attack_model' : 'inception_v3',
        'input_dir_path' : '/home/yantao/datasets/imagenet_100image/original',
        'output_dir_path' : 'images_adv',
        'attack_num_steps' : 200,
    }

    if params['attack_model'] == 'vgg16':
        params['IMAGE_SIZE'] = 224
        attack_layer_idx = 14  # 0 ~ 28
        model = Vgg16(attack_layer_idx=attack_layer_idx)
    elif params['attack_model'] == 'resnet152':
        params['IMAGE_SIZE'] = 224
        attack_layer_idx = 8  # 0 ~ 8
        model = Resnet152(attack_layer_idx=attack_layer_idx)
    elif params['attack_model'] == 'inception_v3':
        params['IMAGE_SIZE'] = 299
        attack_layer_idx = 10 # 0 ~ 13
        model = Inception_v3(attack_layer_idx=attack_layer_idx)
    else:
        raise ValueError('Invalid attack model type.')

    model_name = model.get_name()
    params['output_dir_path'] = os.path.join('/home/yantao/datasets/imagenet_100image', 'DR_' + model_name + '_layer_{0}'.format(attack_layer_idx) + '_steps_{0}'.format(params['attack_num_steps']))
    if not os.path.exists(params['output_dir_path']):
        os.mkdir(params['output_dir_path'])
    adversary = DispersionAttack_gpu(model, 
                                     epsilon=16/255., 
                                     step_size=2/255., 
                                     steps=params['attack_num_steps'],
                                     )
    
    images_t, file_name_list = load_images(dir_path=params['input_dir_path'], 
                                           size=[params['IMAGE_SIZE'], params['IMAGE_SIZE']], 
                                           order='channel_first', 
                                           zero_one_bound=True, 
                                           to_tensor=True
                                           )

    idx = 0
    pbar = tqdm(total=int(len(images_t) / params['batch_size']) + 1)
    while(idx * params['batch_size'] <= len(images_t)):
        if (idx + 1) * params['batch_size'] <= len(images_t):
            temp_images_t = images_t[idx * params['batch_size']:(idx + 1) * params['batch_size']]
            temp_file_name_list = file_name_list[idx * params['batch_size']:(idx + 1) * params['batch_size']]
        else:
            temp_images_t = images_t[idx * params['batch_size']:]
            temp_file_name_list = file_name_list[idx * params['batch_size']:]

        advs_var = adversary(temp_images_t.cuda(), attack_layer_idx=attack_layer_idx)
        advs_t = advs_var.cpu()
        save_images(advs_t, dir_path=params['output_dir_path'], file_name_list=temp_file_name_list)
        pbar.update()
        idx += 1
    pbar.close()

if __name__ =="__main__":
    main()