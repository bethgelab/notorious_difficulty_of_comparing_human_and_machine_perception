# torch imports
import torch
import torch.nn as nn

# torchvision imports
import torchvision
import torchvision.models as models

DEVICE  = torch.device('cuda')
def load_model(net, regularization=0, pretrained=True):
    print('- load', net)
    if net == 'resnet50':
        model = get_resnet(50, pretrained)
    else:
        print('WARNING: model not implemented')
        
    return model.to(DEVICE)


def get_resnet(layer=50, pretrained=True):
    print('- pretrained:', pretrained)
    if layer == 50:
        resnet = models.resnet50(pretrained=pretrained)
    if layer == 34:
        resnet = models.resnet34(pretrained=pretrained)
        
    num_ftrs = resnet.fc.in_features
    # AdapticeAvgPool: necessary for arbitrary image sizes. (Note that the results are slightly different from the results of AvgPool2d(). In additional tests, I found that there are small numerical differences between the implementations)
    resnet.avgpool = nn.AdaptiveAvgPool2d(1)
                                   
    # new readout: one class instead of 1000 classes
    resnet.fc = nn.Linear(num_ftrs, 1)  # 1 for BCE
    return resnet

