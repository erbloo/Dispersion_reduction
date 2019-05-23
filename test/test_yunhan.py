import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2
from torch_utils import variable_to_numpy
from torch.autograd import Variable
import numpy as np
import foolbox
from api_utils import detect_label, detect_text

with open('labels.txt','r') as inf:
    imagenet_dict = eval(inf.read())

def numpy_to_variable(image, device):
    image = image / 255
    x_image = np.expand_dims(image, axis=0)
    x_image = Variable(torch.tensor(x_image), requires_grad=True)
    x_image = x_image.to(device)
    x_image.retain_grad()
    return x_image

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.model = models.vgg16(pretrained=True).cuda().eval()
        features = list(self.model.features)
        self.features = torch.nn.ModuleList(features).cuda().eval()

    def forward(self, x):
        results = []
        last_layer = self.model(x)
        for ii, model in enumerate(self.features):
            x = model(x)
            if isinstance(model, torch.nn.modules.conv.Conv2d):
                results.append(x)
        results.append(last_layer)
        return results

class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.model = models.vgg19(pretrained=True).cuda().eval()
        features = list(self.model.features)
        self.features = torch.nn.ModuleList(features).cuda().eval()

    def forward(self, x):
        results = []
        last_layer = self.model(x)
        for ii, model in enumerate(self.features):
            x = model(x)
            if isinstance(model, torch.nn.modules.conv.Conv2d):
                results.append(x)
        results.append(last_layer)
        return results



def attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


cuda = torch.device('cuda:0')
image = cv2.imread("images/origin.jpg")
image = cv2.resize(image, (224, 224))
image = image[...,::-1]
image = (np.transpose(image, (2, 0, 1))).astype(np.float32)
image = numpy_to_variable(image, cuda)

model = Vgg16()
k=6

#learning_rate = 1e-4
#optimizer = torch.optim.Adam(x_adv, lr=learning_rate)

loss = None
x_adv = image
x_adv.retain_grad()

for i in range(1000):
    preds = model(x_adv)
    internal_logits = preds[-2]
    final_logits = preds[-1]
    label = np.argmax(variable_to_numpy(final_logits))
    loss = internal_logits.std()
    
    # save image
    x_adv_np = x_adv.cpu().detach().numpy()
    x_adv_np = np.squeeze(x_adv_np)
    x_adv_np = np.transpose(x_adv_np, (1, 2, 0))
    x_adv_np = (x_adv_np * 255).astype(np.uint8)

    if (i % 50 == 0):
        Image.fromarray(x_adv_np).save('./out/car_%d.jpg' % i)
        google_label = detect_label('./out/car_%d.jpg' % i)
        print(i, variable_to_numpy(loss), imagenet_dict[label])
        print(google_label)

    loss.backward(retain_graph=True)
    grad = x_adv.grad.data
    x_adv = attack(x_adv, 0.01, grad)
    x_adv.retain_grad()
