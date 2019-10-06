import torch
from torchvision import models

def getPretrainedModel(arch):
    if arch == 'alexnet':
        return models.alexnet(pretrained=True)
    elif arch == 'resnet18':
        return models.resnet18(pretrained=True)
    elif arch == 'resnet34':
        return models.resnet34(pretrained=True)
    elif arch == 'resnet50':
        return models.resnet50(pretrained=True)
    elif arch == 'resnet101':
        return models.resnet101(pretrained=True)
    elif arch == 'resnet152':
        return models.resnet152(pretrained=True)
    elif arch == 'squeezenet1_0':
        return models.squeezenet1_0(pretrained=True)
    elif arch == 'squeezenet1_1':
        return models.squeezenet1_0(pretrained=True)
    elif arch == 'vgg11':
        return models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        return models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        return models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        return models.vgg19(pretrained=True)
    elif arch == 'densenet121':
        return models.densenet121(pretrained=True)
    elif arch == 'densenet161':
        return models.densenet161(pretrained=True)
    elif arch == 'densenet169':
        return models.densenet169(pretrained=True)
    elif arch == 'densenet201':
        return models.densenet201(pretrained=True)
    elif arch == 'inception_v3':
        return models.inception_v3(pretrained=True)
    else:
        print("Error, invalid architecture - ",arch)
        exit()
        return 0
