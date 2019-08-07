import torchvision.models as models
import torch

import pdb

class Vgg16(torch.nn.Module):
    def __init__(self, attack_layer_idx=-1):
        super(Vgg16, self).__init__()
        self.model = models.vgg16(pretrained=True).cuda().eval()
        self.features = torch.nn.Sequential(*list(self.model.features)[:attack_layer_idx]).cuda().eval()
        self._model_name = 'vgg16'

    def prediction(self, x):
        pred = self.model(x)
        x = self.features(x)
        return x, pred

    def get_name(self):
        return self._model_name

if __name__ == "__main__":
    Vgg16()
