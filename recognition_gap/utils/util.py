# This script contains all functions that have directly to do with the
# MIRC-search.

import os
import numpy as np
import torch
from scipy.special import softmax
from PIL import Image


def get_size_in_original_px_space_list():
    """Create list of pixels in original pixel space when each
    child is reduced to 80% of its parent's size (by cropping
    or resolution reduction).

    Returns:
        size_in_real_pixels_list: list with pixel sizes
    """

    cur_px_size = 224  # image size after preprocessing
    size_in_original_px_space_list = [cur_px_size]
    for i in range(20):
        cur_px_size = round(cur_px_size * 80 / 100)
        size_in_original_px_space_list.append(cur_px_size)

    return size_in_original_px_space_list


def extract_crops(image, size, stride=1):
    """Extract crops of size size from image using stride. Careful! This function only works for a batch_size = 1.

    Args:
        image:  torch tensor, dtype = torch.float32.
                expected dimensions: C X W X H,
                                     e.g. torch.Size([3, 224, 224])
        size:   size of the returned crops
        stride: number of pixels moved until next crop is extracted.
                Here: 1 (Even though, by default, BagNet-33 was trained with stride 8 and uses stride 8 when used to predict classes.)

    Returns:
        crops:  tensor of size number_of_crops x n_channels x crop_size x crop_size,
                e.g. torch.Size([36864, 3, 33, 33])
                type = torch.float32
    """
    image_permuted = image.permute(1, 2, 0)
    crops_unfolded = image_permuted.unfold(
        0, size, stride).unfold(
        1, size, stride)
    crops = crops_unfolded.contiguous().view((-1, 3, size, size))

    return crops


def get_logits_for_patches(patches, rf, model, DEVICE):
    """For each of the 1000 ImageNet classes, compute the logit of each patch.

    Args:
        patches:            tensor, dtype = torch.float32.
                            expected dimensions: n_patches x C X W X H,
                            e.g. torch.Size([36864, 3, 33, 33])
                            where 33 is the patch-size from BagNet
        rf:                 number of pixels in image crop for BagNet-33
                            rf stands for receptive field size
        DEVICE:             device where computations are run: "cuda" or "cpu"

    Returns:
        logits_for_patches: logit predictions for each patch
                            torch tensor of dimensions n_patches x 1000 classes
    """

    # check that patches are of correct dimensions
    if not (patches.shape[1] == 3 and patches.shape[2]
            == rf and patches.shape[3] == rf):
        raise ValueError(
            f"Patches are of unusual dimensions. We would have expected "
            f"torch.Size([x, 3, 33, 33]). As a consequence, logits cannot "
            f"be calculated. These are the dimensions: {patches.shape}"
        )

    logits_for_patches = torch.empty(
        [patches.shape[0], 1000], device=DEVICE, dtype=torch.float64
    )  # 1000 ImageNet classes
    _batch_size = 500  # arbitrary value
    for idx, this_patch in enumerate(torch.split(patches, _batch_size)):
        logits_cuda = model(this_patch)  # e.g. torch.Size([500, 1, 1, 1000])
        if this_patch.shape[0] < _batch_size:
            logits_for_patches[
                idx * _batch_size: idx * _batch_size + this_patch.shape[0]
            ] = torch.squeeze(
                logits_cuda
            )  # (500, 1000)
        else:
            logits_for_patches[
                idx * _batch_size: idx * _batch_size + _batch_size
            ] = torch.squeeze(logits_cuda)

    return logits_for_patches


def get_prob_for_correct_classes_of_whole_img(logits_for_patches, target_list):
    """determine probability for whole image by adding up the
    individual probabilities for each true_label in the
    target_list

    Args:
        logits_for_patches:                 logit predictions for each patch
                                            torch tensor, dtype = torch.float32
                                            np_array of dimensions n_patches x 1000 classes
        target_list:                        list of tensors containing the int's of the correct class(es) out of 1000

    Returns:
        prob_of_whole_img_for_targets_only: probability of whole image summed over the target class(es)
                                            torch tensor torch.float32
    """

    logit_avg_whole_image = torch.mean(logits_for_patches, dim=0)
    prob_for_whole_image_targets_summed = get_prob_for_logits(
        logit_avg_whole_image[None, :], target_list
    )

    return prob_for_whole_image_targets_summed


def get_prob_and_custom_prob_per_crops(
    logits_for_patches,
    img_size_work_px_space,
    n_pixels_in_crop,
    descendent_specifier,
    target_list,
    rf,
    DEVICE,
):
    """Determine the probability and the custom probability (i.e. the non-Deep-Learning "logit", cf. Appendix C.2) for crops according to the descendent_specifier, i.e. either each crop or only the four corner crops.

    Note that for the grouping of patches into one crop, each directly neighboring patch is considered (stride 1: logits_for_patches_reshaped[start_row:stop_row:stride_1, start_col:stop_col:stride_1]). This enables us to both either select all data for all crops or only the data for the corner crops. This is in contrast to the value that was used to train and evaluate BagNet-33 (stride = 8).

    Args:
        logits_for_patches:            logit predictions for each patch
                                       torch tensor, dtype = torch.float32
                                       np_array of dimensions n_patches x 1000
        img_size_work_px_space:        number of image pixels in latest parent
        n_pixels_in_crop:              size of child crop
        descendent_specifier:          choice between selecting all crops ("stride1") or only four corner crops ("Ullman4")
        target_list:                   list of targets
        rf:                            number of pixels in image crop for BagNet-33
                                       rf stands for receptive field size

    Returns:
        prob_per_crop:                 list of length n_crops^2 containing the probabilities per relevant crop
        custom_prob_per_crop:          list of length n_crops^2 containing the custom probabilities per relevant crop
        """

    # When the crop is larger than 33x33 (or in fact 37x37 because that's the
    # next larger pixel size appearing in the decreasing order of pixels when
    # decreasing by 80% for each crop from 224 pixels), group patches into
    # crops to calculate the probaiblities and the custom probabilities
    if img_size_work_px_space > 37:
        # calculate how many crops there are
        n_crops = img_size_work_px_space - n_pixels_in_crop + 1
        # calculate how many patches contribute to one crop in one dimensions
        # (i.e. width or height)
        n_patches_contribute_to_crop = n_pixels_in_crop - rf + 1

        # make matrix square instead of one-dimensional along the patch-axis
        patch_square_length = int(np.sqrt(logits_for_patches.size()[0]))
        logits_for_patches_reshaped = torch.reshape(
            logits_for_patches,
            (patch_square_length,
             patch_square_length,
             logits_for_patches.shape[1]),
        )

        # loop through each crop
        prob_per_crop = []
        custom_prob_per_crop = []
        for start_row in range(n_crops):
            stop_row = start_row + n_patches_contribute_to_crop
            for start_col in range(n_crops):
                stop_col = start_col + n_patches_contribute_to_crop

                # average logits over patches
                logit_avg_of_cur_patch = torch.mean(
                    torch.mean(
                        logits_for_patches_reshaped[
                            start_row:stop_row, start_col:stop_col
                        ],
                        dim=0,
                    ),
                    dim=0,
                )

                # calculate probabilities
                prob_for_targets_summed = get_prob_for_logits(
                    logit_avg_of_cur_patch[None, :], target_list
                )
                prob_per_crop.append(prob_for_targets_summed)

                # calculate custom probabilities
                cur_custom_prob_per_crop = get_custom_prob(
                    logit_avg_of_cur_patch[None, :], target_list, DEVICE
                )
                custom_prob_per_crop.append(cur_custom_prob_per_crop[0])

    # patches correspond to crops
    else:
        custom_prob_per_crop = get_custom_prob(
            logits_for_patches, target_list, DEVICE)
        prob_per_crop = list(
            get_prob_for_logits(
                logits_for_patches,
                target_list))

    # if only the four corner crops are of interest ("Ullman4"), get that data
    # only
    if descendent_specifier == "Ullman4":
        prob_per_crop, custom_prob_per_crop = extract_corner_data_for_Ullman4(
            prob_per_crop, custom_prob_per_crop
        )

    return prob_per_crop, custom_prob_per_crop


def get_prob_for_logits(logits_n_patches_x_n_classes, target_list):
    """Calculate the probability for the given logits (which are possibly an average over patches, hence representing a crop and only giving n_patches = 1).

    Args:
        logits_n_patches_x_n_classes: logits
                                      dimensions: n_patches x 1000
                                      for whole_image or crop, n_patches = 1
        target_list:                  list of targets

    Returns:
        prob_for_targets_summed:      summed probability for all targets
    """
    prob_for_targets_separately = torch.nn.functional.softmax(
        logits_n_patches_x_n_classes, dim=1
    )[:, target_list]
    prob_for_targets_summed = torch.sum(prob_for_targets_separately, dim=1)

    return prob_for_targets_summed


def get_custom_prob(logits_for_patches, target_list, DEVICE):
    """Calculate the custom probability (i.e. the non-Deep-Learning "logit", cf. Appendix C.2) for n_patches patches based
    on the logit predictions and the true classes

    Args:
        logits_for_patches: logit predictions for each patch of BagNet
                            torch tensor, dtype = torch.float32
                            n_patches x 1000
        target_list:        list of targets

    Returns:
        logits_for_patches: list of custom probabilities for each Patch
                            len(logits_for_patches) = n_patches, e.g. 36864
    """
    logits_for_patches_true_label = logits_for_patches[:, target_list]
    logits_for_patches_non_correct = torch.empty(
        [logits_for_patches.size()[0], logits_for_patches.size()[1] - len(target_list)],
        device=DEVICE,
        dtype=torch.float64,
    )
    for idx_i, target_i in enumerate(target_list):
        # first sub-tensor
        if idx_i == 0:
            logits_for_patches_non_correct[:, : target_i.item(
            )] = logits_for_patches[:, : target_i.item()]
            # updates for next round
            prev_idx = target_i.item() + 1
            start_idx = target_i.item()
        # middle sub-tensor
        else:
            insertion_length = target_i.item() - prev_idx
            stop_idx = start_idx + insertion_length
            logits_for_patches_non_correct[:,
                                           start_idx:stop_idx] = logits_for_patches[:,
                                                                                    prev_idx: target_i.item()]
            # updates for next round
            prev_idx = target_i.item() + 1
            start_idx = stop_idx
    # last sub-tensor
    insertion_length = logits_for_patches.size()[1] - prev_idx
    stop_idx = start_idx + insertion_length
    logits_for_patches_non_correct[:, start_idx:stop_idx] = logits_for_patches[
        :, prev_idx:
    ]

    custom_prob_patch = torch.log(
        torch.sum(torch.exp(logits_for_patches_true_label), dim=1)
    ) - torch.log(torch.sum(torch.exp(logits_for_patches_non_correct), dim=1))
    return custom_prob_patch.tolist()


def extract_corner_data_for_Ullman4(prob_per_crop, custom_prob_per_crop):
    """extract the data that corresponds to the four corner crops

    Args:
        prob_per_crop:        list containing the probabilities of all crops
        custom_prob_per_crop: list containing the custom probabilities of all crops

    Returns:
        prob_per_crop:        list containing the probabilities of the corner crops only
        custom_prob_per_crop: list containing the custom probabilities of the corner crops only

    """
    # get the indices of the corner crops
    n_total_crops = len(prob_per_crop)
    idx_upper_right = int(np.sqrt(n_total_crops)) - 1
    idx_lower_right = n_total_crops - 1
    idx_lower_left = idx_lower_right - idx_upper_right
    # only keep corner crop data in prob_per_crop
    prob_per_crop_corner_only = list(
        prob_per_crop[i] for i in [
            0,
            idx_upper_right,
            idx_lower_left,
            idx_lower_right])
    # only keep corner crop data in custom_prob_per_crop
    custom_prob_per_crop_corner_only = list(
        custom_prob_per_crop[i]
        for i in [0, idx_upper_right, idx_lower_left, idx_lower_right]
    )

    return prob_per_crop_corner_only, custom_prob_per_crop_corner_only


def get_most_predictive_crop(
    idx_most_predictive_crop,
    new_image,
    crop_reduced_resolution_real_px_space,
    custom_prob_per_crop,
    descendent_specifier,
):
    """determine crop that corresponds to most predictive crop

    Args:
        idx_most_predictive_crop: index of most predictive crop (int)
        new_image:                tensor, dtype = torch.float32.
                                  expected dimensions: C X W X H,
                                  e.g. torch.Size([3, 224, 224])
        crop_reduced_resolution_real_px_space: tensor, dtype = torch.float32.
                                  expected dimensions: C X W X H,
                                  e.g. torch.Size([3, 179, 179])
        custom_prob_per_crop:     list of custom probability (i.e. the non-Deep-Learning "logit", cf. Appendix C.2) per crop
        descendent_specifier:     choice between selecting all crops ("stride1") or only four corner crops ("Ullman4")

    Returns:
        new_crop:                 tensor, dtype = torch.float32.
                                  expected dimensions: C X W X H,
                                  e.g. torch.Size([3, 224, 224])"""

    # the most predictive crop is the one w/ the reduced resolution
    if idx_most_predictive_crop == (len(custom_prob_per_crop) - 1):
        new_crop = crop_reduced_resolution_real_px_space
    # the most predictive crop is NOT the one w/ the reduced resolution
    else:
        n_pixels_in_image = new_image.shape[-1]
        n_pixels_in_crop = round(n_pixels_in_image * 80 / 100)

        if descendent_specifier == "stride1":
            n_crops = n_pixels_in_image - n_pixels_in_crop + 1

            start_row = int(idx_most_predictive_crop / n_crops)
            start_col = idx_most_predictive_crop - start_row * n_crops
            stop_row = start_row + n_pixels_in_crop
            stop_col = start_col + n_pixels_in_crop
        else:  # Ullman4
            # 0 - upper left
            # 1 - upper right
            # 2 - lower left
            # 3 - lower right

            # indices for rows
            if idx_most_predictive_crop in (0, 1):
                start_row = 0
                stop_row = n_pixels_in_crop
            else:
                start_row = n_pixels_in_image - n_pixels_in_crop
                stop_row = n_pixels_in_image
            # indices for columns
            if idx_most_predictive_crop in (0, 2):
                start_col = 0
                stop_col = n_pixels_in_crop
            else:
                start_col = n_pixels_in_image - n_pixels_in_crop
                stop_col = n_pixels_in_image
        new_crop = new_image[:, start_row:stop_row, start_col:stop_col]

    return new_crop


def get_resized_img(image, size, DEVICE):
    """Calculate the resized image.
    PIL requires uint8 images on cpu, hence the tranformation back and forth.

    Args:
        image:    torch tensor, dtype = torch.float32.
                  expected dimensions: CxHxW, e.g. torch.Size([3, 224, 224])
        size:     size of target image
        DEVICE:   device where computations are run: "cuda" or "cpu"

    Returns:
        im:       torch tensor, dtype = torch.float32
                  dimensions: CxHxW
    """
    # if the image is on cuda, move it to the cpu
    if image.device.type == DEVICE.type:
        image = image.cpu()
    image_np = image.numpy()
    im = image_np.copy()
    im = undo_normalization(im.transpose(1, 2, 0))
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    im = im.resize((size, size), Image.BILINEAR)
    im = np.array(im).astype(np.float32) / 255
    im = apply_normalization(im).transpose(2, 0, 1)
    im = torch.from_numpy(im).to(DEVICE, dtype=torch.float)
    return im


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def undo_normalization(X):
    """undo ImageNet normalization, that was applied in preprocessing, to a single
    image

    Args:
        X:     numpy array, dtype=float32.
               dimensions: HxWxC

    Returns:
        image: numpy array
    """
    image = X.copy()
    image *= std[None, None]  # None twice for width and height dimensions
    image += mean[None, None]
    image = np.clip(image, 1e-18, 1 - 1e-18)

    return image


def apply_normalization(X):
    """apply ImageNet normalization (the same one that was applied in preprocessing)
    to a single image

    Args:
        X:     numpy array, dtype=float32.
               dimensions: HxWxC

    Returns:
        image: numpy array
    """
    image = X.copy()
    image -= mean[None, None]  # None twice for width and height dimensions
    image /= std[None, None]
    return image


def get_img_identifier(Ullman_or_ImageNet, path, target_list):
    """Create and return an image identifier with its correct class. It is used as a key in dictionaries and as part of filenames for figures.

    Args:
        path:           absolute path to directory

    Returns:
        img_identifier: string to identify an image and its correct class
                        e.g. glasses_INclass836 or n02834397_ILSVRC2012_val_00019315
    """
    if Ullman_or_ImageNet == "Ullman":
        img_identifier = (
            f"{path[0].split(os.path.sep)[-2]}_INclass{int(target_list[0])}"
        )
    else:
        img_identifier = (path[0].split(f"val{os.path.sep}")[
            1].replace(os.path.sep, "_")[:-5])

    return img_identifier
