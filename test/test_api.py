import torchvision.models as models
import torch
from PIL import Image
from image_utils import load_image, save_image
from torch_utils import numpy_to_variable, variable_to_numpy
from image_utils import numpy_to_bytes
from models.vgg import Vgg16
from models.resnet import Resnet152
from attacks.dispersion import DispersionAttack
from attacks.mifgsm import MomentumIteratorAttack
from api_utils import detect_label_numpy, detect_text_numpy, detect_text_file, googleDet_to_Dictionary
from api_utils import detect_safe_search_numpy, detect_objects_numpy, detect_faces_numpy, detect_faces_file, detect_objects_file
import numpy as np
import torchvision
import pdb

# Resnet152 [4, 5, 6, 7]
# Vgg16 [2, 7, 14, 21, 28]
image_np = load_image(data_format='channels_first', fname='face.jpeg')

'''
image = numpy_to_variable(image_np)
model = Vgg16()
internal = [i for i in range(29)]
attack = DispersionAttack(model, epsilon=60./255, step_size=1./255, steps=1000)
adv = attack(image, internal=internal)
adv_np = variable_to_numpy(adv)

image = numpy_to_bytes(image_np)
ret = detect_faces_numpy(image)
# ret = detect_faces_file('images/porn2.jpg')
print(ret)
#save_image(adv_np)
'''

output_dic = detect_faces_file('images/porn.jpg')

bbox_list = output_dic['boxes']

from PIL import Image, ImageFont, ImageDraw, ImageEnhance

source_img = Image.open('images/porn.jpg').convert("RGB")

draw = ImageDraw.Draw(source_img)
for top, left, bottom, right in bbox_list:
    draw.rectangle([int(left), int(top), int(right), int(bottom)])

source_img.save('temp.jpg')