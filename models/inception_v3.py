import torchvision.models as models
import torch
import pdb

class Inception_v3(torch.nn.Module):
    def __init__(self, attack_layer_idx=[-1]):
        super(Inception_v3, self).__init__()
        model = models.inception_v3(pretrained=True).cuda().eval()
        self.features_list = []
        for temp_layer_idx in attack_layer_idx:
            features = torch.nn.Sequential(*list(model.children())[:temp_layer_idx]).cuda().eval()
            self.features_list.append(features)
        self._model_name = 'inception_v3'

    def prediction(self, x):
        logits = []
        for feature in self.features_list:
            temp_logits = feature(x)
            logits.append(temp_logits)
        return logits

    def get_name(self):
        return self._model_name

if __name__ == "__main__":
    Inception_v3()