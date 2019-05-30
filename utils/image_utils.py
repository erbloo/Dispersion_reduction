from PIL import Image, ImageFont, ImageDraw
import numpy as np
from io import BytesIO
from google.cloud.vision import types
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os

import pdb


def load_images(dir_path='images', size=[224, 224]):
    '''load images from the diractory to channel first tensor.
    '''
    file_list = os.listdir(dir_path)
    images_chw_list = []
    if not file_list:
        warnings.warn("No image loaded.")
    for file_name in file_list:
        img_pil = Image.open(os.path.join(dir_path, file_name)).convert('RGB').resize(size)
        temp_img_np = np.array(img_pil).astype(np.float32) / 255.
        temp_img_chw = np.transpose(temp_img_np, (2, 0, 1))
        images_chw_list.append(temp_img_chw)
    images_bchw = np.array(images_chw_list)
    return torch.from_numpy(images_bchw), file_list

def save_images(images_t, dir_path='outputs', file_name_list=None):
    '''save channel first tensor batch to images. 
    '''
    images_bwhc = np.transpose(images_t.numpy(), (0, 2, 3, 1))
    if file_name_list is None:
        file_name_list = []
        for idx in range(len(images_bwhc)):
            file_name_list.append('adv_{0:04d}.jpg'.format(idx))
    assert len(file_name_list) == len(images_bwhc)

    for image_whc, file_name in zip(images_bwhc, file_name_list):
        image_whc = (image_whc * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_whc).save(os.path.join(dir_path, file_name))  

def numpy_to_bytes(image, format='JPEG'):
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype is not np.uint8:
        image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    byte_io = BytesIO()
    image.save(byte_io, format=format)
    image = byte_io.getvalue()
    image = types.Image(content=image)
    return image

def to_coordinates(image, coords):
    top = coords[0] * image.size[1]
    left = coords[1] * image.size[0]
    bottom = coords[2] * image.size[1]
    right = coords[3] * image.size[0]
    return [top, left, bottom, right]

def draw_boxes(image, labels, boxes):
    thickness = (image.size[0] + image.size[1]) // 300
    '''
    draw = ImageDraw.Draw(image)
    draw.rectangle([200, 300, 400, 500], outline=(255, 0, 255))
    del draw

    # write to stdout
    image.show()
    '''
    boxes = [boxes[1]]
    labels = [labels[1]]
    for i, box in enumerate(boxes):
        draw = ImageDraw.Draw(image)
        label = labels[i]
        label_size = draw.textsize(label)
        top, left, bottom, right = to_coordinates(image, box)
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=(255, 0, 255))
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=(255, 0, 255))
        draw.text(text_origin.tolist(), label, fill=(0, 0, 0))
        del draw
    
    image.show()

def save_bbox_img(img, bbox_list, from_path=True, out_file='temp.jpg'):
    from PIL import Image, ImageDraw

    if from_path:
        source_img = Image.open(img).convert("RGB")
    else:
        source_img = Image.fromarray(img)

    draw = ImageDraw.Draw(source_img)
    for top, left, bottom, right in bbox_list:
        draw.rectangle([int(left), int(top), int(right), int(bottom)])

    source_img.save(out_file)

def visualize_features(intermediate_features, output_dir, file_prefix='', data_format='channels_last', image_size=(224, 224), only_first_channel=True):
    if data_format == 'channels_last':
        intermediate_features = np.transpose(intermediate_features, (2, 0, 1))
    
    for feature_idx, temp_feature in enumerate(tqdm(intermediate_features)):
        if only_first_channel and feature_idx != 0:
            break
        temp_file_path = os.path.join(output_dir, file_prefix + '_{0:03d}.png'.format(feature_idx))
        plt.imshow(temp_feature)
        plt.colorbar()
        plt.savefig(temp_file_path)
        plt.close()

def visualize_features_compare(ori_features, adv_features, output_dir, file_prefix='', data_format='channels_last', image_size=(224, 224), only_first_channel=True):
    if data_format == 'channels_last':
        ori_features = np.transpose(ori_features, (2, 0, 1))
        adv_features = np.transpose(adv_features, (2, 0, 1))

    for feature_idx, (temp_ori_feature, temp_adv_feature) in enumerate(tqdm(zip(ori_features, adv_features))):
        if only_first_channel and feature_idx != 0:
            break
        temp_file_path = os.path.join(output_dir, file_prefix + '_{0:03d}.png'.format(feature_idx))

        temp_ori_feature = cv2.resize(temp_ori_feature, image_size)
        temp_adv_feature = cv2.resize(temp_adv_feature, image_size)

        fig=plt.figure(figsize=(10, 10))
        fig.add_subplot(1, 3, 1)
        plt.imshow(temp_ori_feature)
        fig.add_subplot(1, 3, 2)
        plt.imshow(temp_adv_feature)
        fig.add_subplot(1, 3, 3)
        plt.imshow(0 * np.ones((10, 10)).astype(np.uint8))
        plt.colorbar()
        plt.savefig(temp_file_path)
        plt.close()

        fig, axes = plt.subplots(nrows=1, ncols=2)
        for idx, ax in enumerate(axes.flat):
            if idx == 0:
                im = ax.imshow(temp_ori_feature) #, vmin=0, vmax=10
            elif idx == 1:
                im = ax.imshow(temp_adv_feature) #, vmin=0, vmax=10

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.savefig(temp_file_path)
        plt.close()