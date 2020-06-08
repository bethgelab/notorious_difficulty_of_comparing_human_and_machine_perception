# basic imports
import pandas as pd
import os.path
import os
import argparse
import numpy as np

# torch imports
import torch

# custom imports
import svrt_utils
import utils
import my_models

DEVICE  = torch.device('cuda')
RES_DIR = '../results/'
   
    
def get_performance(set_num, model):
    #load dataset
    testloader = svrt_utils.load_dataset_svrt(set_num=set_num,
                                          batch_size=64, 
                                          split='test')
    
    model.eval()
    torch.set_grad_enabled(False)  # save memory and computation cost by not calculating the grad  
    perfs = utils.AverageMeter()  # object to record performance
       
    for i, data in enumerate(testloader):
        inputs_tensor, labels_tensor = data  # get the inputs
        inputs = inputs_tensor.to(DEVICE)  # move to gpu. dimensions: [64, 3, 288, 288]
        labels = labels_tensor.to(DEVICE).float()  # move to gpu

        outputs = torch.squeeze(model(inputs))  # forward pass, squeeze for BCE
        perf = utils.eval_perf(outputs, labels)   # compute performance
        batch_size = inputs.size(0)
        perfs.update(perf, batch_size)

    return perfs.avg.item()


def main():
    '''
    this function generates a csv file that contains the performances for the different stimuli sets
    # python3 svrt_test.py -net resnet50 -pretrained 0
    '''
    parser = argparse.ArgumentParser(description='SVRT test')    
    parser.add_argument('-net', help='network')
    parser.add_argument('-pretrained', type=int, help='finetune: 1, scratch: 0')
    args = parser.parse_args()

    # evaluate the model with the learning rate that achieved the highest performance on the validation set
    if args.pretrained:  # finetune
        possible_lr = [0.0003, 0.0001, 6e-5]
        lrindex = np.array([[1., 1., 2.],
                       [0., 2., 2.],
                       [2., 1., 1.],
                       [1., 0., 2.],
                       [1., 0., 2.],
                       [0., 0., 1.],
                       [0., 1., 1.],
                       [1., 0., 1.],
                       [0., 0., 0.],
                       [0., 0., 2.],
                       [1., 0., 0.],
                       [0., 0., 1.],
                       [1., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 0.],
                       [0., 1., 1.],
                       [0., 0., 0.],
                       [0., 2., 0.],
                       [0., 0., 2.]]) # these values come from analysis/svrt_results.ipynb
    else:  # scratch
        possible_lr = [0.001, 0.0006, 0.0003]
        lrindex = np.array([[0., 2., 2.],
                       [0., 1., 1.],
                       [2., 0., 1.],
                       [1., 1., 1.],
                       [2., 0., 2.],
                       [1., 0., 2.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 0.],
                       [1., 0., 1.],
                       [0., 0., 0.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 0.],
                       [0., 0., 0.],
                       [1., 0., 1.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 0., 0.]])

    res = np.zeros([23, 3])
    for s, set_num in enumerate([20, 7, 21, 19, 1, 22, 5, 15, 16, 6, 17, 9, 13, 23, 8, 14, 4, 12, 10, 18, 3, 11, 2]):
        for n, num_train in enumerate([28000, 1000, 100]):
            lr = possible_lr[int(lrindex[s, n])]

            # load model
            print('load model')
            model = my_models.load_model(args.net)

            # load checkpoint
            if args.pretrained:
                exp_name = 'resnet50_set' + str(set_num) + '_lr' + str(lr) + '_numtrain' + str(num_train) + '_augment1_epochmult10_v1'
            else:
                exp_name = 'resnet50_set' + str(set_num) + '_pretrained0_lr' + str(lr) + '_numtrain' + str(num_train) + '_augment1_epochmult10_v1'
            try:
                model.load_state_dict(torch.load('svrt_checkpoints/' + exp_name + '/best_prec.pt'))
            except OSError:
                exp_name = exp_name[:-1] + '0' #try v0 if v1 does not exist
                model.load_state_dict(torch.load('svrt_checkpoints/' + exp_name + '/best_prec.pt'))


            # get performance 
            perf = get_performance(set_num = str(set_num), 
                                model = model)

            # add performance to results
            res[s][n] = perf

            # save result
            if args.pretrained:
                fname = RES_DIR + 'testset_finetune.npy'
            else:
                fname = RES_DIR + 'testset_scratch.npy'

            np.save(fname, res)

if __name__ == '__main__':
    main()