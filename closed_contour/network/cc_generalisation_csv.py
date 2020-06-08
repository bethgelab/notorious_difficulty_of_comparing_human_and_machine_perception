# generalisation tests
# author: Christina Funke

# basic imports
import pandas as pd
import os.path
import os
import argparse
from pathlib import Path

# torch imports
import torch

# custom imports
import cc_utils
import utils
import my_models

DEVICE = torch.device('cuda')
TOP_DIR = '../'
RES_DIR = TOP_DIR + 'results/imagelevel/'


def get_pred(outputs, labels):
    '''
    get prediction of model, batch_size has to be 1
    '''
    sigm = torch.nn.Sigmoid()(outputs)
    predicted = (sigm > 0.5).float()
    return predicted


def get_df(set_num, net_string, contrast, model, crop_margin):
    # load dataset
    testloader, testset = cc_utils.load_dataset_cc(set_num=set_num,
                                                   contrast=contrast,
                                                   batch_size=1,
                                                   split='test',
                                                   return_dataset=1,
                                                   crop_margin=crop_margin  # crop 16px margin
                                                   )

    model.eval()
    # save memory and computation cost by not calculating the grad
    torch.set_grad_enabled(False)

    columns = (["imagename", "prediction", "label", "logits"])
    df = pd.DataFrame([], columns=columns)

    for i, data in enumerate(testloader):
        inputs_tensor, labels_tensor = data  # get the inputs
        # move to gpu. dimensions: [64, 3, 288, 288]
        inputs = inputs_tensor.to(DEVICE)
        labels = labels_tensor.to(DEVICE).float()   # move to gpu

        if net_string == 'bagnet':
            # forward pass, squeeze for BCE
            outputs = torch.squeeze(model(inputs, 0))
        else:
            # forward pass, squeeze for BCE
            outputs = torch.squeeze(model(inputs))

        predicted = get_pred(outputs, labels)   # compute performance
        name = os.path.basename(testset.samples[i][0])[4:-4]
        df = df.append(pd.DataFrame([[name,
                                      predicted.detach().item(),
                                      labels.detach().item(),
                                      outputs.detach().item()]],
                                    columns=columns))
    return df


def main():
    '''
    this function generates a csv file that contains the predictions for the individual images
    '''
    parser = argparse.ArgumentParser(
        description='CC generalisation, results for individual images')
    parser.add_argument(
        '-exp_name',
        help='experiment name (has to be the same as the name of the checkpoint)')
    parser.add_argument('-net', help='network')
    parser.add_argument(
        '-set_nums',
        nargs='*',
        type=int,
        default=[],
        help='which testsets? No argument to do all sets')
    parser.add_argument(
        '-crop_margin',
        default=0,
        type=int,
        help='crop 16 px margin from each side (1), keep original image (0)')

    args = parser.parse_args()
    net_string = args.net[:6]
    # make results folder
    res_folder = os.path.join(RES_DIR, args.exp_name)
    if not Path(res_folder).is_dir():
        print('make new folder:', res_folder)
        os.makedirs(res_folder)

    # load model
    print('load model')
    model = my_models.load_model(args.net)

    # load checkpoint
    model.load_state_dict(
        torch.load(
            TOP_DIR +
            'network/cc_checkpoints/' +
            args.exp_name +
            '/best_prec.pt'))

    # if no set_nums are defined do all sets
    if args.set_nums == []:
        # leave set14, set15 and set16 out because they are curvy contours w/
        # different line widths and because the image files were corrupt
        args.set_nums = list(range(1, 14)) + list(range(17, 26))

    for set_num in args.set_nums:
        for contrast in [0, 0.2, 0.4, 0.6, 0.8, 1]:

            # get result df
            df = get_df(set_num=str(set_num),
                        net_string=net_string,
                        contrast='contrast' + str(contrast),
                        model=model,
                        crop_margin=args.crop_margin)

            # save dataframe
            fname = res_folder + '/set' + \
                str(set_num) + '_contrast' + str(contrast) + '.csv'
            df.to_csv(fname, index=False)


if __name__ == '__main__':
    main()
