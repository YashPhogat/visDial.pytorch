import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import pdb
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import numpy as np

imsize = 224
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    return image

data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
 )])

model_fit = models.vgg16(pretrained=True)
model_fit.eval()

class AlexNetConv4(nn.Module):
    def __init__(self):
        super(AlexNetConv4, self).__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(model_fit.features.children())[:]
        )

    def forward(self, x):
        x = self.features(x)
        return x


def vgg_16(path_to_img):
    image = image_loader(data_transforms,path_to_img)

    model = AlexNetConv4()
    # layer = model_fit._modules.get('avgpool')
    #
    # my_embedding = torch.zeros(512,7,7)
    # def copy_data(m, i, o):
    #     my_embedding.copy_(o.data.squeeze())
    #
    # h = layer.register_forward_hook(copy_data)
    img = model(image)

    # h.remove()
    new_embedding = img.permute(0,2,3,1)
    return new_embedding


