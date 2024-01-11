import torch
from torch.autograd import Function
import torch.nn.functional as F

from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Define the Grad-CAM class
class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        return self.model(x)[0]

    def backward(self, pred, target_class):
        self.model.zero_grad()
        one_hot = torch.zeros_like(pred)
        one_hot[:, target_class] = 1
        pred.backward(gradient=one_hot, retain_graph=True)

    def generate(self, x, target_class):
        output = self.forward(x)
        self.backward(output, target_class)

        target = self.target_layer
        gradient = self.gradients

        alpha = gradient.mean(dim=(2, 3), keepdim=True)
        weights = F.relu(alpha * target).mean(dim=(0, 1), keepdim=True)
        gcam = (weights * target).sum(dim=1, keepdim=True)

        gcam = F.relu(gcam)
        gcam = F.interpolate(gcam, x.shape[2:], mode="bilinear", align_corners=False)
        gcam = gcam - gcam.min()
        gcam = gcam / gcam.max()

        return gcam