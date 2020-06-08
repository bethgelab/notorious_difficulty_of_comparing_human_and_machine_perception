# basic imports
import pandas as pd
import os
import argparse
import numpy as np

# torch imports
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

# tensorboard
from tensorboardX import SummaryWriter

# custom imports
import utils
import svrt_utils
import my_models

DEVICE  = torch.device('cuda')
RES_DIR = '../results/'

def main():
    parser = argparse.ArgumentParser(description='Training SVRT')
    parser.add_argument('-lr', type=float, help='learning rate (3e-4)')
    parser.add_argument('-net', help='network (resnet50, resnet34,...)')
    parser.add_argument('-num_trainimages', type=int, default=28000, help='number of training images (number < 28000')
    parser.add_argument('-dat_augment', default=1, type=int, help='data augmentation during training? (0 or 1)')
    parser.add_argument('-set_num', default=1, type=int, help='problem number (1 - 23)')
    parser.add_argument('-epoch_multiplier', default=2, type=int, help='1 corresponds to 28000 images')
    parser.add_argument('-save', default=0, type=int, help='1 to save results as csv file')
    parser.add_argument('-pretrained', default=1, type=int, help='1 to load pretrained weights, 0 to train from scratch')
    parser.add_argument('-optimizer', default='Adam', help='The default optimizer is Adam. Optionally, you can choose SGD w/ momentum, then choose "SGDm"')

    args = parser.parse_args()
        
    # set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    batch_size  = 64 
    epochs = args.epoch_multiplier * 28000 // args.num_trainimages # adapt the number of epochs if the size of the trainingset changes (2 epoch for the full dataset)
    print('number of epochs:', epochs)

    epoch_decay = epochs // 3 # after this many epochs the learning rate decays by a factor of 0.5
    exp_name    = args.net + '_set' + str(args.set_num) + '_pretrained' + str(args.pretrained) + '_lr' + str(args.lr) + '_numtrain' + str(args.num_trainimages) + '_augment' + str(args.dat_augment) + '_epochmult' + str(args.epoch_multiplier)
    # load model
    print('load model')
    model = my_models.load_model(args.net, bool(args.pretrained))

    # load dataset
    print('load dataset')
    valloader = svrt_utils.load_dataset_svrt(set_num = args.set_num,
                                            batch_size = batch_size, 
                                            split = 'val')                                              
    trainloader = svrt_utils.load_dataset_svrt(set_num = args.set_num, 
                                              batch_size = batch_size,
                                              split = 'train',
                                              trainimages = args.num_trainimages,
                                              dat_augment = args.dat_augment) #number of images in the trainingset
    # loss criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) # skip parameters that have requires_grad==False
    elif args.optimizer == 'SGDm':
        # optimizer as in original resnet training in He et al. 2015
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr,
                                    momentum=0.9, # as in original resnet training in He et al. 2015
                                    weight_decay=1e-4) # as in original resnet training in He et al. 2015

    # create new checkpoints- and tensorboard-directories
    for version in range(100):
        checkpointdir = 'svrt_checkpoints/' + exp_name + '_v' + str(version) + '/'
        tensorboarddir = 'svrt_tensorboard_logs/' + exp_name + '_v'+str(version) + '/'
        # if checkpointdir already exists, skip it
        if not os.path.exists(checkpointdir):
            break
    os.makedirs(checkpointdir)
    os.makedirs(tensorboarddir)

    # create writer
    writer = SummaryWriter(tensorboarddir)
    print('writing to this tensorboarddir', tensorboarddir)
    
    # steps (x-axis) for plotting tensorboard
    print('len(trainloader)', len(trainloader), 'len(valloader)', len(valloader))
    step = 0
    best_prec = 0
    # loop through several epochs
    for epoch in range(epochs):
        svrt_utils.adjust_learning_rate_svrt(optimizer, epoch, args.lr, step, epoch_decay)
        
        print('train model')
        pc_train, step = utils.train(model, trainloader, optimizer, criterion, writer, epoch, checkpointdir, step)
        
        # reduce the number of validations according to the following rules:
        # always validate after last epoch
        # if there are more than 10 epochs: validate only 10 times during the training
        if epoch == 0 or epoch == epochs - 1 or epochs <= 10 or epoch % (epochs // 10) == 0: 
            print('validate model after training')
            prec, tmp = utils.validate(model, valloader, criterion, writer, epoch, step)
        
            # remember best prec on valset and save checkpoint
            if prec > best_prec:
                best_prec = prec
                torch.save(model.state_dict(), checkpointdir + '/best_prec.pt')

    # add performance to dataframe and save it to csv
    if args.save:
        columns = ['exp_name', 'problem', 'pc_train', 'pc_val', 'lr', 'num_trainimages', 'epochs', 'pretrained']
        if args.pretrained:
            fname = RES_DIR + 'exp_finetune.csv'
        else:
            fname = RES_DIR + 'exp_scratch.csv'

        if os.path.isfile(fname):
            df = pd.read_csv(fname)
        else:
            df = pd.DataFrame([], columns=columns)

        df = df.append(pd.DataFrame([[exp_name+ '_v' + str(version), args.set_num, pc_train, best_prec, args.lr, args.num_trainimages, epochs, args.pretrained]], columns=columns))
        # save dataframe
        df.to_csv(fname, index=False) 

    # save last checkpoint
    torch.save(model.state_dict(), checkpointdir + '/last_epoch.pt')
    
    # close writer
    writer.close()
    

if __name__ == '__main__':
    main()
