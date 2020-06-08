# basic imports
import os
from PIL import Image
import numpy as np

# torch imports
import torch
from torch.utils.data.sampler import *

# torchvision imports
import torchvision
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.datasets.folder import IMG_EXTENSIONS     

# custom imports
import utils

DEVICE  = torch.device('cuda')
   
    
def find_classes(dir):
    classes = ['diff', 'equal'] # we have only two classes
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
    
    
def make_dataset(dir, class_to_idx, extensions):
    ''' helper to read SVRT dataset
    '''
    images = []
    dir = os.path.expanduser(dir)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                idx_class=int(fname[7]) # the image names start with 'sample_1_...' or 'sample_0_...'
                item = (path, idx_class)
                images.append(item)
    return images


class MixedImageFolder(datasets.DatasetFolder):
    '''
    all images are in the folder root. There are two classes and the images are named:
    sample_0_xxx.png 
    sample_1_xxx.png
    '''
    def __init__(self, root, extensions=IMG_EXTENSIONS, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx, extensions)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.samples = imgs
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        
def load_dataset_svrt(set_num, batch_size, split, trainimages=0, dat_augment=False, return_dataset=False, prep_method='imagenet'):
    dat_dir = '../stimuli/problem_' + str(set_num) + '/' 

    if dat_augment:
        print('- do data augmentation')
        prepfun = utils.prep_imagenet_augment
    else:
        prepfun = utils.prep_imagenet

    # need this for adversarial examples
    if prep_method == 'orig':
        prepfun = None

    #load dataset
    dataset = MixedImageFolder(root=
            dat_dir + split + '/', transform=prepfun)

    if split == 'train':
        shuffle = True
        dataset = utils.get_subset(dataset, trainimages) # change the size of the trainingset
    else:
        shuffle = False

    # define dataloader
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    if return_dataset:
        return dataloader, dataset
    else:
        return dataloader

    
def adjust_learning_rate_svrt(optimizer, epoch, init_lr, step, epoch_decay):
    """Sets the learning rate to the initial LR decayed by 10 every epoch_decay epochs"""        
    lr = init_lr * (0.5 ** (epoch // epoch_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr