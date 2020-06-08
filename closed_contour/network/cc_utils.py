# author: Christina Funke

# basic imports
import os
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

# torch imports
import torch
from torch.utils.data.sampler import *
from torch.utils import data

# torchvision imports
import torchvision
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import find_classes
from torchvision import transforms

# custom imports
import utils
sys.path.append('../stimuli_gen/')
import add_background
import linedrawing_polygon

DEVICE = torch.device('cuda')


def unique_test(fname, idx_class, num_images):
    '''
    [o = open, c = closed, y = yes, n = no]
    example dataset of size 10
    1, 2, 3, 4, 5,    6, 7, 8, 9, 10
    c, o, c, o, c,    o, c, o, c, o
    n, y, n, y, n,    n, y, n, y, n      num_images = 4
    n, y, n, n, n,    n, y, n, n, n      num_images = 2

    example dataset of size 12
    1, 2, 3, 4, 5, 6,     7, 8, 9, 10, 11, 12
    c, o, c, o, c, o,     c, o, c, o,  c,  o
    n, y, n, y, n, y      y, n, y, n,  y,  n      num_images = 6
    n, y, n, y, n, n      y, n, y, n,  n,  n      num_images = 4
    '''
    if fname[0] == 'v':  # val
        sample = int(fname[3:-4])
        size = 2800
    elif fname[0:2] == 'te':  # test
        sample = int(fname[4:-4])
        size = 5600
    elif fname[0:2] == 'tr':  # train
        if fname[0:9] == 'trainmany':  # large trainingset
            sample = int(fname[9:-4])
            size = 280000
        else:  # small trainingset
            sample = int(fname[5:-4])
            size = 28000

    if num_images > size // 2:
        raise ValueError(
            "num_trainimages is larger than available dataset / 2 ")
    if num_images % 1:
        raise ValueError("num_trainimages has to be even number")

    if idx_class == 0:  # class 0: closed, uneven numbers
        return (sample > size // 2) and (sample <= size // 2 + num_images)
    elif idx_class == 1:  # class 1: open, even numbers
        return sample <= num_images


def pair_test(fname, idx_class, num_images):
    '''
    [o = open, c = closed, y = yes, n = no]
    example dataset of size 10
    1, 2, 3, 4, 5,    6, 7, 8, 9, 10
    c, o, c, o, c,    o, c, o, c, o
    y, y, y, y, n,    n, n, n, n, n      num_images = 4
    y, y, n, n, n,    n, n, n, n, n      num_images = 2
    '''
    if fname[0] == 'v':  # val
        sample = int(fname[3:-4])
        size = 2800
    elif fname[0:2] == 'te':  # test
        sample = int(fname[4:-4])
        size = 5600
    elif fname[0:2] == 'tr':  # train
        if fname[0:9] == 'trainmany':  # large trainingset
            sample = int(fname[9:-4])
            size = 280000
        else:  # small trainingset
            sample = int(fname[5:-4])
            size = 28000

    if num_images > size:
        raise ValueError("num_trainimages is larger than available dataset")
    return sample <= num_images


def make_dataset(dir, class_to_idx, extensions, test_fun, num_images):
    '''
    necessary for class CustomImageFolder
    '''
    images = []
    dir = os.path.expanduser(dir)
    c0 = 0
    c1 = 0
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(
                        fname, extensions) and test_fun(
                        fname, class_to_idx[target], num_images):
                    if class_to_idx[target]:
                        c1 += 1
                    else:
                        c0 += 1
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    print('- number of images in: class0', c0, ' / class1', c1)

    return images


class CustomImageFolder(datasets.DatasetFolder):
    '''
    There are always pairs of open/closed images that are almost identical.

    Requirements dataset:
    - size: valset: 2800 images, testset: 5600 images, trainset: 28 000 images
    - closed: uneven numbers (valset: val1.png - val2799.png)
    - open: even numbers (val2.png - val2800.png)
    '''

    def __init__(
            self,
            root,
            test_fun,
            num_images,
            extensions=IMG_EXTENSIONS,
            transform=None,
            target_transform=None,
            loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(
            root,
            class_to_idx,
            extensions,
            test_fun,
            num_images)
        if len(imgs) == 0:
            raise(
                RuntimeError(
                    "Found 0 images in subfolders of: " +
                    root +
                    "\n"
                    "Supported image extensions are: " +
                    ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.samples = imgs
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.test_fun = test_fun
        self.num_images = num_images


def load_dataset_cc(
        set_num,
        contrast,
        batch_size,
        split,
        prep_method='imagenet',
        regularization=0,
        num_trainimages=None,
        dat_augment=0,
        unique='pairs',
        return_dataset=0,
        crop_margin=0):
    """ load data set for closed contours.
    """

    # path to stimuli
    dat_dir = '../stimuli/set' + str(set_num) + '/' + contrast + '/'

    # determine data preprocessing
    if dat_augment:
        print('- do data augmentation')
        prepfun = utils.prep_imagenet_augment
    elif regularization == 1:
        prepfun = utils.prep_imagenet_super_augment
    else:
        prepfun = utils.prep_imagenet

    # crop images
    if crop_margin:
        prepfun = transforms.Compose([
            transforms.CenterCrop(256),
            prepfun
        ])

    # need this for adversarial examples
    if prep_method == 'orig':
        prepfun = None
        if crop_margin:
            prepfun = transforms.CenterCrop(256)

    # load dataset
    print('split', split)
    if split == 'train' or split == 'trainmany':
        if unique == 'unique':
            print('- use unique dataset')
            dataset = CustomImageFolder(
                root=dat_dir + split + '/',
                test_fun=unique_test,
                num_images=num_trainimages,
                transform=prepfun)
        elif unique == 'pairs':
            print('- use paired dataset')
            dataset = CustomImageFolder(
                root=dat_dir + split + '/',
                test_fun=pair_test,
                num_images=num_trainimages,
                transform=prepfun)
        else:
            raise ValueError("unique has to be unique or pairs")
        shuffle = True
    else:
        print('dat_dir + split', dat_dir + split)
        dataset = torchvision.datasets.ImageFolder(
            root=dat_dir + split + '/', transform=prepfun)
        shuffle = False

    print('- ', dataset.class_to_idx)
    # define dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=shuffle, batch_size=batch_size)
    if return_dataset:
        return dataloader, dataset
    else:
        return dataloader


class Dataset_OTF(data.Dataset):
    ''' generate data online
    '''

    def __init__(self, epoch_len, crop_margin):
        'Initialization'
        self.len = epoch_len
        self.crop_margin = crop_margin

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        'Generates one sample of data'

        # set seed
        # random seed: use idx because of num_workers = 8, and time because of
        # multiple epochs
        np.random.seed((idx + 1) * int(time.time()) % (2**32 - 1))
        # np.random.seed(torch.initial_seed() % (2**32 - 1)) # torch seed is
        # controlled by data.Dataloader (doesn't work)

        # get label
        label = np.random.randint(2)    # pick open/closed randomly
        # label = idx%2                 # i%2 ensures that open and closed
        # images alternate

        # get linedrawing
        img = linedrawing_polygon.set1_otf(label)

        # add background
        img_bg = add_background.getbg_otf(img, 0)

        # crop margin
        if self.crop_margin:
            img_bg = img_bg[16:288 - 16, 16:288 - 16, :]

        # apply preprocessing
        img_prep = utils.prep_imagenet(img_bg)
        return img_prep, label


def get_a_pair2(set_num, contrast, idx, crop_margin=0):
    ''' Get an image pair (closed and open version).
    This function is used for the heatmaps.
    '''
    assert idx < 1000
    dat_dir = 'stimuli_for_heatmap'

    img_open = plt.imread(dat_dir + '/test28.png')  # even numbers
    img_closed = plt.imread(dat_dir + '/test27.png')
    img_open_prep = np.copy(img_open)
    img_closed_prep = np.copy(img_closed)

    # apply preprocessing
    prepfun = utils.prep_imagenet

    # crop margin
    if crop_margin:
        img_closed_prep = img_closed_prep[16:288 - 16, 16:288 - 16, :]
        img_open_prep = img_open_prep[16:288 - 16, 16:288 - 16, :]
        img_closed = img_closed[16:288 - 16, 16:288 - 16, :]
        img_open = img_open[16:288 - 16, 16:288 - 16, :]

    img_open_prep = prepfun(img_open_prep)
    img_closed_prep = prepfun(img_closed_prep)

    return (img_open, img_open_prep, 1), (img_closed, img_closed_prep, 0)
