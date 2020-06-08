# authors: Christina Funke and Judy Borowski

# basic imports
import os
import argparse
import numpy as np
import csv

# torch imports
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

# tensorboard
from tensorboardX import SummaryWriter

# for experiment name
import datetime

# custom imports
import cc_utils
import utils
import my_models
from adabound import AdaBound

TOP_DIR = '/../'

DEVICE = torch.device('cuda')


def main():
    parser = argparse.ArgumentParser(description='Training CC')
    parser.add_argument('-lr', type=float, help='learning rate (3e-4)')
    parser.add_argument('-net', help='network (resnet50, resnet34,...)')
    parser.add_argument(
        '-num_trainimages',
        type=int,
        default=28000,
        help='number of training images (number < 28000). For otf1: number of images per epoch')
    parser.add_argument(
        '-dat_augment',
        default=1,
        type=int,
        help='data augmentation during training? (0 or 1)')
    parser.add_argument(
        '-otf',
        default=1,
        type=int,
        help='on the fly data generation? (0 or 1')
    parser.add_argument('-unique', default='pairs', help='(pairs, unique)')
    # training bagnets requires smaller batch_size, otherwise memory issues
    parser.add_argument(
        '-batch_size',
        default=64,
        type=int,
        help='batchsize: default 64, for bagnets smaller b/c of memory issues')
    parser.add_argument(
        '-optimizer',
        default='Adam',
        help='The default optimizer is Adam. Optionally, you can choose adabound ("adabound"), which is used for BagNet training.')
    parser.add_argument(
        '-contrast',
        default='contrastrandom',
        help='The default is to train on random contrast images. You can choose "contrast0"')
    parser.add_argument(
        '-n_epochs',
        default=10,
        type=int,
        help='number of epochs')
    parser.add_argument(
        '-regularization',
        default=0,
        type=int,
        help='Flag to choose (1) or not choose (0, default) regularization techniques: scaling, rotation and dropout.')
    parser.add_argument(
        '-load_checkpoint',
        default='',
        help='String to choose loading given checkpoint from best precision (1) or to opt for leaving initialization at ImageNet/random (empty string, default).')
    parser.add_argument('-load_checkpoint_epoch', default=0, type=int, help='')
    parser.add_argument(
        '-crop_margin',
        default=0,
        type=int,
        help='crop 16 px margin from each side (1), keep original image (0)')

    args = parser.parse_args()

    print('regularization', args.regularization)

    # set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    epochs = args.n_epochs
    print('number of epochs:', epochs)
    # after this many epochs the learning rate decays by a factor of 0.1
    epoch_decay = epochs // 2
    now = datetime.datetime.now()  # add the date to the experiment name
    exp_name = args.net + '_lr' + str(args.lr) + '_numtrain' + str(args.num_trainimages) + '_augment' + str(args.dat_augment) + '_' + str(args.unique) + '_batchsize' + str(args.batch_size) + '_optimizer' + str(
        args.optimizer) + '_' + str(args.contrast) + '_reg' + str(args.regularization) + '_otf' + str(args.otf) + '_cropmargin' + str(args.crop_margin) + '_' + str(now.month) + str(now.day) + str(now.year)

    if args.load_checkpoint:
        exp_name = '_CONTINUED_FINETUNING_' + exp_name

    # load model
    print('load model')
    if args.net[:6] == 'bagnet':
        model = my_models.load_model(args.net, args.regularization)
    else:
        model = my_models.load_model(args.net)

    # load checkpoint if resuming fine-tuning from later epoch
    if args.load_checkpoint:
        model.load_state_dict(
            torch.load(
                'cc_checkpoints/' +
                args.load_checkpoint +
                '/best_prec.pt'))

    # load dataset
    print('load dataset')
    valloader = cc_utils.load_dataset_cc(set_num=1,
                                         contrast=args.contrast,
                                         batch_size=args.batch_size,
                                         split='val',
                                         regularization=args.regularization,  # whether to use super-augmentation
                                         crop_margin=args.crop_margin)  # crop 16px margin

    if args.otf:  # online datageneration. Works only for set1, contrast0, unique, no dataaugmentation or regularisation
        dataset = cc_utils.Dataset_OTF(
            epoch_len=args.num_trainimages,
            crop_margin=args.crop_margin)
        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=8)
    else:
        trainloader = cc_utils.load_dataset_cc(set_num=1,
                                               contrast=args.contrast,
                                               batch_size=args.batch_size,
                                               split='trainmany',  # CAREFUL! This is the LARGE dataset
                                               regularization=args.regularization,  # whether to use super-augmentation
                                               num_trainimages=args.num_trainimages,
                                               dat_augment=args.dat_augment,
                                               unique=args.unique,  # number of images in the trainingset
                                               crop_margin=args.crop_margin)
    # loss criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            filter(
                lambda p: p.requires_grad,
                model.parameters()),
            lr=args.lr)  # skip parameters that have requires_grad==False
    elif args.optimizer == 'adabound':
        optimizer = AdaBound(
            filter(
                lambda p: p.requires_grad, 
                model.parameters()), 
            lr=args.lr, final_lr=0.1)

    # create new checkpoints- and tensorboard-directories
    for version in range(100):
        checkpointdir = 'cc_checkpoints/' + \
            exp_name + '_v' + str(version) + '/'
        tensorboarddir = 'cc_tensorboard_logs/' + \
            exp_name + '_v' + str(version) + '/'
        # if checkpointdir already exists, skip it
        if not os.path.exists(checkpointdir):
            break
    print('tensorboarddir', tensorboarddir)
    os.makedirs(checkpointdir)
    os.makedirs(tensorboarddir)

    # create writer
    writer = SummaryWriter(tensorboarddir)
    print('writing to this tensorboarddir', tensorboarddir)

    # steps (x-axis) for plotting tensorboard
    step = 0
    best_prec = 0
    first_epoch = 0

    val_loss = []  # list to store all validation losses to detect plateau and potentially decrease the lr
    # if fine-tuning is continued, load old loss values to guarantee lr
    # adjustment works properly
    if args.load_checkpoint:
        with open('cc_checkpoints/' + args.load_checkpoint + '/epoch_loss_lr.csv', newline='') as csvfile:
            training_log = csv.reader(
                csvfile, delimiter=',', lineterminator='\n')
            for row_counter, row in enumerate(training_log):
                if row_counter == 1:  # skip title row
                    for list_idx in range(int(row[0])):
                        val_loss.append('NaN')
                    val_loss.append(row[1])
                elif row_counter > 1:
                    val_loss.append(float(row[1]))
            # first epoch is the one after the last epoch in the csv file
            first_epoch = int(row[0]) + 1
        csvfile.close()

    n_epoch_plateau = 25  # number of epochs over which the presence of a plateau is evaluated
    counter_lr_adjust = 1
    epoch_of_last_lr_adjust = 0

    with open(checkpointdir + '/epoch_loss_lr.csv', 'w') as csvFile:
        csv_writer = csv.writer(csvFile, delimiter=',', lineterminator='\n')
        csv_writer.writerow(['epoch', 'prec', 'loss', 'lr'])
    csvFile.close()

    net_string = args.net[:6]

    for epoch in range(first_epoch, epochs):
        print('current epoch ', epoch)

        print('train model')
        _, step = utils.train(net_string, model, args.regularization, trainloader,
                              optimizer, criterion, writer, epoch, checkpointdir, step)

        # validate after every epoch
        print('validate model after training')
        prec, loss = utils.validate(
            net_string, model, args.regularization, valloader, criterion, writer, epoch, step)
        val_loss.append(loss)

        # save to csv file
        with open(checkpointdir + '/epoch_loss_lr.csv', 'a') as csvFile:
            csv_writer = csv.writer(
                csvFile, delimiter=',', lineterminator='\n')
            for param_group in optimizer.param_groups:  # find current lr
                curr_lr = param_group['lr']
            csv_writer.writerow([epoch, prec, loss, curr_lr])
        csvFile.close()

        # after more than n_epoch_plateaus, check if there is a plateau
        if epoch >= n_epoch_plateau:
            # only adjust lr if no adjustment has ever happened or
            # if the last adjustment happened more than n_epoch_plateau epochs
            # ago
            if epoch_of_last_lr_adjust == 0 or epoch - \
                    n_epoch_plateau >= epoch_of_last_lr_adjust:
                adjust_lr_counter = 0
                print('len(val_loss)', len(val_loss))
                for idx in range(epoch - n_epoch_plateau + 2, epoch + 1):
                    print('idx', idx)
                    if val_loss[idx] - val_loss[idx - 1] < 0.05:
                        adjust_lr_counter += 1
                    else:
                        break
                if adjust_lr_counter == n_epoch_plateau - 1:
                    print('adjust lr!!!')
                    utils.adjust_learning_rate_plateau(
                        optimizer, epoch, args.lr, counter_lr_adjust)
                    counter_lr_adjust += 1
                    epoch_of_last_lr_adjust = epoch

        # remember best prec on valset and save checkpoint
        if prec > best_prec:
            best_prec = prec
            torch.save(model.state_dict(), checkpointdir + '/best_prec.pt')

        # save checkpoint for every epoch
        torch.save(
            model.state_dict(),
            checkpointdir +
            '/epoch' +
            str(epoch) +
            '_step' +
            str(step) +
            '.pt')

    # close writer
    writer.close()

    print('Wohoooo, completely done!')


if __name__ == '__main__':
    main()
