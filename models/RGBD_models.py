from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from torchvision import models


def ResNet50(in_planes, out_planes, pretrained=False):
    if pretrained is True:
        model = models.resnet50(pretrained=True)
        print("Pretrained model is loaded")
    else:
        model = models.resnet50(pretrained=False)
    if in_planes == 4:
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(model.fc.in_features, out_planes)
    return model

def resnext50(in_planes, out_planes, pretrained=False):
    if pretrained is True:
        model = models.resnext50_32x4d(pretrained=True)
        print("Pretrained model is loaded")
    else:
        model = models.resnext50_32x4d(pretrained=False)
    if in_planes == 4:
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(model.fc.in_features, out_planes)
    return model

def mobilenet(in_planes, out_planes, pretrained=False):
    if pretrained is True:
        model = models.mobilenet_v2(pretrained=True)
        print("Pretrained model is loaded")
    else:
        model = models.mobilenet_v2(pretrained=False)
    if in_planes == 4:
        model.features[0][0] = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
        nn.init.kaiming_normal_(model.features[0][0].weight, mode='fan_out', nonlinearity='relu')
    # Parameters of newly constructed modules have requires_grad=True by default
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, out_planes)
    return model

def inception(in_planes, out_planes, pretrained=False):
    if pretrained is True:
        model = models.inception_v3(pretrained=True)
        print("Pretrained model is loaded")
    else:
        model = models.inception_v3(pretrained=False)
    if in_planes == 4:
        model.Conv2d_1a_3x3["conv"] = nn.Conv2d(4, 32, kernel_size=3, stride=2, bias=False)
        nn.init.kaiming_normal_(model.Conv2d_1a_3x3["conv"].weight, mode='fan_out', nonlinearity='relu')
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(model.fc.in_features, out_planes)
    return model

def vgg16(in_planes, out_planes, pretrained=False):
    if pretrained is True:
        model = models.vgg16(pretrained=True)
        print("Pretrained model is loaded")
    else:
        model = models.vgg16(pretrained=False)
    if in_planes == 4:
        model.features[0] = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(model.features[0].weight, mode='fan_out', nonlinearity='relu')
    # Parameters of newly constructed modules have requires_grad=True by default
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, out_planes)
    return model
