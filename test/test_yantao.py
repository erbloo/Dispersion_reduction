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
import pdb

# Resnet152 [4, 5, 6, 7]
# Vgg16 [2, 7, 14, 21, 28]
image_np = load_image(data_format='channels_first', fname='0000.png')
image = numpy_to_variable(image_np)


model = Vgg16()
internal = [i for i in range(29)]
#attack = DispersionAttack(model, epsilon=255./255, step_size=7./255, steps=1000, test_api=True)
attack = DispersionAttack_opt(model, epsilon=16./255, steps=2000, test_api=True)
adv = image

adv_np = variable_to_numpy(adv)
Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/3.jpg')
google_label = detect_label_file('./out/3.jpg')

if len(google_label) > 0:
    pred_cls = google_label[0].description
else:
    pred_cls = None
print(pred_cls)

for temp_attack_layer_idx in internal:
    
    if temp_attack_layer_idx != 12 and temp_attack_layer_idx != 14:
        continue
    
    print('temp_attack_layer_idx: ', temp_attack_layer_idx)
    adv, info_dict = attack(image, attack_layer_idx=temp_attack_layer_idx, internal=internal, test_steps=200, gt_label=pred_cls)
    print(info_dict)
    adv_np = variable_to_numpy(adv)
    linf = int(np.max(abs(image_np - adv_np)) * 255)
    print('linf: ', linf)
    l1 = np.mean(abs(image_np - adv_np)) * 255
    print('l1: ', l1)
    l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
    print('l2: ', l2)


'''
model = torchvision.models.vgg16(pretrained=True).cuda()
attack = MomentumIteratorAttack(model, decay_factor=0.5, epsilon=60./255, steps=1000, step_size=1./255, random_start=False)
image_torch_nchw = torch.from_numpy(np.expand_dims(image_np, axis=0)).float()
pred_nat = model(image_torch_nchw.cuda()).detach().cpu().numpy()
label = np.argmax(pred_nat)
label_tensor = torch.tensor(np.array([label]))
adv = attack(image_torch_nchw, label_tensor)
adv_np = variable_to_numpy(adv)
Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/3.jpg')
google_label = detect_label_file('./out/3.jpg')
if len(google_label) > 0:
    pred_cls = google_label[0].description
else:
    pred_cls = None
print(pred_cls)

linf = int(np.max(abs(image_np - adv_np)) * 255)
print('linf: ', linf)
l1 = np.mean(abs(image_np - adv_np)) * 255
print('l1: ', l1)
l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
print('l2: ', l2)
'''