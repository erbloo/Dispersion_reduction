import torchvision.models as models
import torch
import pdb

class Resnet152(torch.nn.Module):
    def __init__(self, attack_layer_idx=-1):
        super(Resnet152, self).__init__()
        self.model = models.resnet152(pretrained=True).cuda().eval()
        self.features = torch.nn.Sequential(*list(self.model.children())[:attack_layer_idx]).cuda().eval()
        self._model_name = 'resnet152'

    def prediction(self, x):
        pred = self.model(x)
        x = self.features(x)
        return x, pred

    def get_name(self):
        return self._model_name

if __name__ == "__main__":
    Resnet152()