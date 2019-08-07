import sys
sys.path.append('/home/yantao/workspace/projects/baidu/Dispersion_reduction')

import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import inception
from test_models.inception_resnet_v2 import inception_resnet_v2
from utils.image_utils import load_images
from tqdm import tqdm

slim = tf.contrib.slim

import pdb


def main():
    params = {
        'batch_size' : 16,
        'checkpoint_path' : "test_models/inception_resnet_v2/ens/ens_adv_inception_resnet_v2.ckpt", # "test_models/inception_v3/ens3/ens3_adv_inception_v3.ckpt",
        'images_benign_dir' : '/home/yantao/datasets/imagenet_100image/original',
        'images_adv_dir' : "images_adv",
        'model_name' : 'inception_resnet_v2',
        'IMAGE_SIZE' : 299,
        "NUM_CLASSES" : 1001,
    }
    sess = tf.Session()
    images_benign, _ = load_images(dir_path=params['images_benign_dir'],
                                     size=[299, 299], 
                                     zero_one_bound=True
                                    )
    images_adv, _ = load_images(dir_path=params['images_adv_dir'],
                                     size=[299, 299], 
                                     zero_one_bound=True
                                    )

    input_ph = tf.placeholder(tf.float32, shape=[None, params['IMAGE_SIZE'], params['IMAGE_SIZE'], 3])
    if params['model_name'] == 'inception_v3':
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, _ =  inception.inception_v3(
                input_ph, 
                num_classes=params['NUM_CLASSES'], is_training=False, reuse=None)
    elif params['model_name'] == 'inception_resnet_v2':
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, _ =  inception_resnet_v2.inception_resnet_v2(
                input_ph, 
                num_classes=params['NUM_CLASSES'], is_training=False, reuse=None)
    else:
        raise ValueError('Invalid model name: %s' % (params['model_name']))
    
    saver = tf.train.Saver()
    saver.restore(sess, params['checkpoint_path'])

    idx = 0
    pbar = tqdm(total=int(len(images_benign) / params['batch_size']) + 1)
    pd_labels_benign = []
    pd_labels_adv = []
    while(idx * params['batch_size'] <= len(images_benign)):
        if (idx + 1) * params['batch_size'] <= len(images_benign):
            temp_images_benign = images_benign[idx * params['batch_size']:(idx + 1) * params['batch_size']]
            temp_images_adv = images_adv[idx * params['batch_size']:(idx + 1) * params['batch_size']]
        else:
            temp_images_benign = images_benign[idx * params['batch_size']:]
            temp_images_adv = images_adv[idx * params['batch_size']:]

        feed_dict = {
            input_ph : temp_images_benign,
        }
        temp_outputs = sess.run(logits, feed_dict=feed_dict)
        temp_labels = temp_outputs.argmax(axis=1).tolist()
        pd_labels_benign += temp_labels

        feed_dict = {
            input_ph : temp_images_adv,
        }
        temp_outputs = sess.run(logits, feed_dict=feed_dict)
        temp_labels = temp_outputs.argmax(axis=1).tolist()
        pd_labels_adv += temp_labels
        pbar.update()
        idx += 1
    pbar.close()

    corrects = 0
    for l_benign, l_adv in zip(pd_labels_benign, pd_labels_adv):
        if l_adv == l_benign:
            corrects += 1
    acc = float(corrects) / len(pd_labels_benign)
    print('labels benign: ', pd_labels_benign)
    print('labels adv: ', pd_labels_adv)
    print('acc: ', acc)
    

if __name__ == "__main__":
    main()

