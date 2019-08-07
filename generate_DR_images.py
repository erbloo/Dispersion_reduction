import sys
sys.path.append('/home/yantao/workspace/projects/baidu/Dispersion_reduction')
import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm

from attacks.dispersion import DispersionAttack_gpu, DispersionAttack_opt_gpu, transform_DR_attack
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
        'attack_num_steps' : 1000,
        'step_size' : 2,
        'learning_rate' : 5e-2,
        'attack_type' : 'ori_trans', # ori/opt
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
        attack_layer_idx = [7] # 0 ~ 13
        model = Inception_v3(attack_layer_idx=attack_layer_idx)
        model_ori = torchvision.models.inception_v3(pretrained=True).cuda().eval()
    else:
        raise ValueError('Invalid attack model type.')

    model_name = model.get_name()
    '''
    params['output_dir_path'] = os.path.join('/home/yantao/datasets/imagenet_100image', 'DR_' + params['attack_type'] + '_' + model_name + '_layer_{0}'.format(attack_layer_idx) + '_steps_{0}_{1:01d}'.format(params['attack_num_steps'], params['step_size']))
    if not os.path.exists(params['output_dir_path']):
        os.mkdir(params['output_dir_path'])
    '''
    params['output_dir_path'] = 'images_adv'
    
    if params['attack_type'] == 'ori':
        adversary = DispersionAttack_gpu(model, epsilon=16/255., step_size=params['step_size']/255., steps=params['attack_num_steps'])
    elif params['attack_type'] == 'opt':
        adversary = DispersionAttack_opt_gpu(model, epsilon=16/255., learning_rate=params['learning_rate'], steps=params['attack_num_steps'])
    elif params['attack_type'] == 'ori_trans':
        adversary = transform_DR_attack(model, epsilon=16/255., step_size=params['step_size']/255., steps=params['attack_num_steps'], prob=1.0, image_resize=330)
    else:
        raise ValueError('Invalid attack type.')
    
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

        advs_var = adversary(temp_images_t.cuda())
        advs_t = advs_var.cpu()
        save_images(advs_t, dir_path=params['output_dir_path'], file_name_list=temp_file_name_list)
        pbar.update()
        idx += 1
    pbar.close()

if __name__ =="__main__":
    main()