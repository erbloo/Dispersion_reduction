import torchvision
import copy
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from PIL import Image
from utils.api_utils import detect_label_file
from attacks.DIM import _tranform_resize_padding
import os

import pdb


class transform_DR_attack(object):
    def __init__(self, model, epsilon, step_size, steps, prob=0.5, image_resize=330):
        self.step_size = step_size
        self.epsilon = epsilon
        self.steps = steps
        self.model = copy.deepcopy(model)
        self.prob = prob
        self.image_resize = image_resize

    def __call__(self, X_nat_var):
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        X_var = copy.deepcopy(X_nat_var)
        for i in range(self.steps):
            X_var = X_var.requires_grad_()

            rnd = np.random.rand()
            if rnd < self.prob:
                transformer = _tranform_resize_padding(X_nat_var.shape[-2], X_nat_var.shape[-1], self.image_resize, resize_back=True)
                X_trans_var = transformer(X_var)
            else:
                X_trans_var = X_var
            loss = torch.tensor(0).float().cuda()
            logit_list = self.model.prediction(X_trans_var)
            for logit in logit_list:
                #loss += -1 * logit.std()
                loss += -1 * logit.var()
            self.model.zero_grad()
            loss.backward()
            grad = X_var.grad.data

            X_var = X_var.detach() + self.step_size * grad.sign_()
            X_var = torch.max(torch.min(X_var, X_nat_var + self.epsilon), X_nat_var - self.epsilon)
            X_var = torch.clamp(X_var, 0, 1)

        return X_var.detach()

class DispersionAttack_gpu(object):
    def __init__(self, model, epsilon, step_size, steps):
        
        self.step_size = step_size
        self.epsilon = epsilon
        self.steps = steps
        self.model = copy.deepcopy(model)

    def __call__(self, X_nat_var):
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        X_var = copy.deepcopy(X_nat_var)
        for i in range(self.steps):
            X_var = X_var.requires_grad_()
            logit = self.model.prediction(X_var)
            loss = -1 * logit.std()
            #loss = -1 * logit.var()
            self.model.zero_grad()
            loss.backward()
            grad = X_var.grad.data

            X_var = X_var.detach() + self.step_size * grad.sign_()
            X_var = torch.max(torch.min(X_var, X_nat_var + self.epsilon), X_nat_var - self.epsilon)
            X_var = torch.clamp(X_var, 0, 1)

        return X_var.detach()


class DispersionAttack_opt_gpu(object):
    def __init__(self, model, epsilon, learning_rate=5e-2, steps=20):
        
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.steps = steps
        self.model = copy.deepcopy(model)

    def __call__(self, X_nat_var):
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        X_var = copy.deepcopy(X_nat_var)
        optimizer = AdamOptimizer(X_var.shape)
        for i in range(self.steps):
            X_var = X_var.requires_grad_()
            logit = self.model.prediction(X_var)
            loss = -1 * logit.std()
            self.model.zero_grad()
            loss.backward()
            grad = X_var.grad.data

            X_var = X_var.detach() + optimizer(grad, learning_rate=self.learning_rate)
            X_var = torch.max(torch.min(X_var, X_nat_var + self.epsilon), X_nat_var - self.epsilon)
            X_var = torch.clamp(X_var, 0, 1)

        return X_var.detach()


class AdamOptimizer:
    """Basic Adam optimizer implementation that can minimize w.r.t.
    a single variable.

    Parameters
    ----------
    shape : tuple
        shape of the variable w.r.t. which the loss should be minimized.

    """

    def __init__(self, shape):
        self.m = torch.tensor(np.zeros(shape).astype(np.float32)).cuda()
        self.v = torch.tensor(np.zeros(shape).astype(np.float32)).cuda()
        self.t = 0

    def __call__(self, gradient, learning_rate,
                 beta1=0.9, beta2=0.999, epsilon=10e-8):

        self.t += 1

        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2

        bias_correction_1 = 1 - beta1 ** self.t
        bias_correction_2 = 1 - beta2 ** self.t

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2

        return -learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)