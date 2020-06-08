# author: Christina Funke and Judy Borowski

# torch imports
import torch
import torch.nn as nn

# torchvision imports
import torchvision
import torchvision.models as models

# custom imports
import bagnets_pytorch

DEVICE = torch.device('cuda')


def load_model(net, regularization=0, pretrained=True):
    print('- load', net)
    if net == 'resnet50':
        model = get_resnet(50, pretrained)

    # bagnets
    elif net == 'bagnet8':
        model = get_bagnet(regularization, rf=8)
    elif net == 'bagnet16':
        model = get_bagnet(regularization, rf=16)
    elif net == 'bagnet32':
        model = get_bagnet(regularization, rf=32)

    return model.to(DEVICE)


def get_resnet(layer=50, pretrained=True):
    print('- pretrained:', pretrained)
    if layer == 50:
        resnet = models.resnet50(pretrained=pretrained)
    if layer == 34:
        resnet = models.resnet34(pretrained=pretrained)

    num_ftrs = resnet.fc.in_features
    # AdapticeAvgPool: necessary for arbitrary image sizes. (Note that the
    # results are slightly different from the results of AvgPool2d(). In
    # additional tests, I found that there are small numerical differences
    # between the implementations)
    resnet.avgpool = nn.AdaptiveAvgPool2d(1)

    # new readout: one class instead of 1000 classes
    resnet.fc = nn.Linear(num_ftrs, 1)  # 1 for BCE
    return resnet


def get_bagnet(regularization=0, rf=8):
    # rf = receptive field size
    # load pretrained BagNet
    if rf == 8:
        model = bagnets_pytorch.bagnet8(regularization, pretrained=True)
    elif rf == 16:
        model = bagnets_pytorch.bagnet16(regularization, pretrained=True)
    elif rf == 32:
        model = bagnets_pytorch.bagnet32(regularization, pretrained=True)

    # new readout: two classes instead of 1000 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    return model
