# author: Christina Funke and Judy Borowski

# basic imports
import os
from PIL import Image
import numpy as np
import itertools

# torch imports
import torch
from torch.utils.data.sampler import *

# torchvision imports
import torchvision
from torchvision import transforms
import torchvision.utils as vutils  # to save images to tensorboard

DEVICE = torch.device('cuda')

SAVE_LOSS_ITER_TRAIN = 1  # save performance etc every ~th batch
SAVE_LOSS_ITER_VAL = 1  # save performance etc every ~th batch
PRINT_LOSS_ITER = 40  # print performance etc every ~th batch

# load images
prep_imagenet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# data augmentation
prep_imagenet_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# super data augmentation including frame and rotation
prep_imagenet_super_augment = transforms.Compose([
    # pixels calculated analytically. Gray background values from Christina
    transforms.Pad(60, (118, 118, 118)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_subset(dataset, num_trainimages):
    '''
    reduce the size of the trainingset. Make sure that half of the images belongs to each class
    trainimages: size of new dataset
    '''
    size_dataset = len(dataset.samples)
    if num_trainimages > size_dataset:
        raise ValueError("num_trainimages is larger than available dataset")
    indices = list(
        np.append(
            np.arange(
                num_trainimages //
                2),
            np.arange(
                size_dataset //
                2,
                size_dataset //
                2 +
                num_trainimages //
                2)))
    # print(len(indices))
    subset = torch.utils.data.dataset.Subset(dataset, indices)
    return subset


def eval_perf(outputs, labels):
    """evaluate model performance"""
    sigm = torch.nn.Sigmoid()(outputs)
    predicted = (sigm > 0.5).float()
    # if predicted.sum().item()!=64:
    #      print(predicted.sum().item())
    #print('number of images classified as class1: ', predicted.sum().item(), ' / ', labels.sum().item())
    return (predicted == labels).sum().float() / outputs.size(0)


def adjust_learning_rate(optimizer, epoch, init_lr, epoch_decay):
    """Sets the learning rate to the initial LR decayed by 10 every epoch_decay epochs"""
    lr = init_lr * (0.1 ** (epoch // epoch_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_plateau(optimizer, epoch, init_lr, counter_lr_adjust):
    """Sets the learning rate to the initial LR decayed by 10
    when there is a plateau in the validation loss"""
    lr = init_lr * (0.1 ** counter_lr_adjust)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(
        net,
        model,
        regularization,
        trainloader,
        optimizer,
        criterion,
        writer,
        epoch,
        checkpointdir,
        step):
    """train the net
    """

    model.train()
    # save memory and computation cost by not calculating the grad
    torch.set_grad_enabled(True)
    losses = AverageMeter()        # object to record loss
    perfs = AverageMeter()         # object to record performance

    print('step', step)

    for i, data in enumerate(trainloader, 0):
        inputs_tensor, labels_tensor = data  # get the inputs
        # move to gpu. dimensions: [64, 3, 288, 288]
        inputs = inputs_tensor.to(DEVICE)
        # move to gpu # we need float() for BCE
        labels = labels_tensor.to(DEVICE).float()

        if net == 'bagnet':
            # forward pass, squeeze for BCE
            outputs = torch.squeeze(model(inputs, regularization))
        else:
            # forward pass, squeeze for BCE
            outputs = torch.squeeze(model(inputs))

        loss = criterion(outputs, labels)
        perf = eval_perf(outputs, labels)       # compute performance

        # update losses and perfs objects
        batch_size = inputs.size(0)
        losses.update(loss.item(), batch_size)
        perfs.update(perf.item(), batch_size)

        # tensorboard
        if i % SAVE_LOSS_ITER_TRAIN == SAVE_LOSS_ITER_TRAIN - 1:
            writer.add_scalar('loss/train',
                              losses.val,
                              step)
            writer.add_scalar('perf/train',
                              perfs.val,
                              step)
            writer.add_scalar('epoch_step/train',
                              epoch,
                              step)
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                break
            writer.add_scalar('lr',
                              lr,
                              step)
        # plot first batch of stimuli
        if i == 0:
            writer.add_image('stimuli0/train',
                             inputs_tensor[0],
                             step)
            writer.add_image('stimuli1/train',
                             inputs_tensor[1],
                             step)
            writer.add_image('stimuli2/train',
                             inputs_tensor[2],
                             step)

        if i % PRINT_LOSS_ITER == PRINT_LOSS_ITER - 1:
            print('\tEpoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Perfs {perf.val:.4f} ({perf.avg:.4f})\t'
                  .format(
                      epoch, i, len(trainloader),
                      loss=losses,
                      perf=perfs
                  ))
        step += 1

        # optimization
        if net == 'bagnet':             # only update every 10 batches
            if i == 0:                  # only zero all the gradients at the very beginning
                optimizer.zero_grad()   # (and again after every tenth batch further down)
            loss.backward()             # compute gradients
            if i % 10 == 0:
                optimizer.step()        # perform optimization step
                optimizer.zero_grad()   # zero all the gradients after every tenth batch
        else:
            optimizer.zero_grad()       # zero the parameter gradients
            loss.backward()             # compute gradients
            optimizer.step()            # perform optimization step

    return perfs.avg, step


def validate(
        net,
        model,
        regularization,
        valloader,
        criterion,
        writer,
        epoch,
        step):
    """validate the model
    """

    model.eval()
    # save memory and computation cost by not calculating the grad
    torch.set_grad_enabled(False)

    losses = AverageMeter()         # object to record loss
    perfs = AverageMeter()          # object to record performance

    for i, data in enumerate(valloader):
        inputs_tensor, labels_tensor = data  # get the inputs
        # move to gpu. dimensions: [64, 3, 288, 288]
        inputs = inputs_tensor.to(DEVICE)
        # move to gpu # we need float() for BCE
        labels = labels_tensor.to(DEVICE).float()

        if net == 'bagnet':
            # forward pass, squeeze for BCE
            outputs = torch.squeeze(model(inputs, regularization))
        else:
            # forward pass, squeeze for BCE
            outputs = torch.squeeze(model(inputs))

        loss = criterion(outputs, labels)
        perf = eval_perf(outputs, labels)   # compute performance

        # update losses and perfs objects
        batch_size = inputs.size(0)
        losses.update(loss, batch_size)
        perfs.update(perf, batch_size)

        # plot first batch of stimuli
        if i == 0:
            writer.add_image('stimuli0/val',
                             inputs_tensor[0],
                             step)
            writer.add_image('stimuli1/val',
                             inputs_tensor[1],
                             step)
            writer.add_image('stimuli2/val',
                             inputs_tensor[2],
                             step)

    # write to tensorboard. Avg over entire val set.
    writer.add_scalar('loss/val',
                      losses.avg,
                      step)
    writer.add_scalar('perf/val',
                      perfs.avg,
                      step)

    # print
    print('\tEpoch: [{0}][{1}/{2}]\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Perfs {perf.val:.4f} ({perf.avg:.4f})\t'
          .format(
              # i started counting at 0, hence +1 correction. len(loader) = #
              # batch_size
              epoch, i + 1, len(valloader),
              loss=losses,
              perf=perfs
          ))

    return perfs.avg.item(), losses.avg.item()
