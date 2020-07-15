# This script contains the MIRC-search.

import os
import numpy as np
import torch
from collections import OrderedDict
import utils.util as util
import utils.data_out as data_out


# number of pixels in image crop for BagNet-33
# rf stands for receptive field size
rf = 33

size_in_original_px_space_list = util.get_size_in_original_px_space_list()


def perform_MIRC_search(
    image_from_loader,
    target_list,
    path,
    model,
    DEVICE,
    Ullman_or_ImageNet,
    descendent_specifier,
    exp_dir,
    write_or_append,
):
    """This procedure carries out the search for MIRCs - it's the meat of the search procedure!

    High level: The search starts with the original, preprocessed image. Its classification accuracy for the whole image is evaluated. If it is above 0.5, the image gets successively cropped and reduced in resolution until the classification probability falls below 0.5.

    Good to know:
    - The search for a MIRC is carried out per image.
    - This algorithm is greedy, i.e. only the path of the best-performing crop is followed.
    - There are two steps to evaluate the probabilities of the children: (1) evaluate image crops from cropping, and (2) evaluate image crops of reduced resolution.
    - The code obeys the following naming convention: *patch* := 33x33 and *crop* := any size.
    - There are three simultaneous ways of counting the sizes of the crops:
      1. img_size_original_px_space: This value indicates the image width or height covered of the original image in its original resolution, i.e. from the 224x224 image. It is always reduced, except for the case when the next child is not a corner crop but the crop of the reduced resolution.
      2. When size-variables or image-variables have 'real' in their name, they refer to the size or image that will be passed on to the next child.
      3. When size-variables or image-variables have 'work' in their name, they refer to the size or image that is processed to evaluate the children's probability and custom probability.
    - In this script, "custom probability" refers to the non-Deep-Learning-measure "logit" (cf. Appendix C.2). The term "logit" itself in this script is used in the Deep-Learning-sense.

    Args:
        image_from_loader,
        target_list,
        path,
        model,
        DEVICE,
        Ullman_or_ImageNet,
        descendent_specifier,
        exp_dir,
        write_or_append
    """

    image = torch.squeeze(image_from_loader.to(DEVICE))
    img_identifier = util.get_img_identifier(
        Ullman_or_ImageNet, path, target_list)

    # at first, check whether the probability of the whole image ("first
    # round") for the correct class is higher than 0.5.
    round_one = True
    size_original_i, reduced_res_counter = 0, 0

    # calculate probability of whole image
    img_size_original_px_space = size_in_original_px_space_list[size_original_i]
    patches = util.extract_crops(image, size=rf)
    logits_for_patches = util.get_logits_for_patches(
        patches, rf, model, DEVICE)
    prob_most_predictive_crop = util.get_prob_for_correct_classes_of_whole_img(
        logits_for_patches, target_list
    )
    print("prob_most_predictive_crop", prob_most_predictive_crop.item())
    prev_img_size_original_px_space = img_size_original_px_space
    img_size_work = image.shape[-1]

    # for saving to csv: update the pixel size with the value from real pixel
    # space and the probability of the current MIRC
    pix_size_MIRC = img_size_original_px_space
    prob_MIRC = prob_most_predictive_crop

    # save to npz
    data_out.write_to_npz(
        exp_dir,
        img_identifier,
        reduced_res_counter,
        image,
        prob_most_predictive_crop,
        img_size_original_px_space,
        target_list,
    )

    # Only enter the search procedure if the probability of the whole image is >
    # 0.5. Iterate through the search procedure until "none of [the]
    # five descendants reaches a recognition criterion [0.5]" (Ullman et al.
    # 2016)
    while prob_most_predictive_crop >= 0.5:
        if not round_one:
            # pick new minimal crop
            image = util.get_most_predictive_crop(
                idx_most_predictive_crop,
                image,
                crop_reduced_resolution_real_px_space,
                custom_prob_crop,
                descendent_specifier,
            )
            # determine image size
            # if image was cropped, increase the counter, else (i.e. if
            # resolution is reduced) don't do anything
            if idx_most_predictive_crop != len(prob_crop) - 1:
                size_original_i += 1
            img_size_original_px_space = size_in_original_px_space_list[size_original_i]
            img_size_work = image.shape[-1]

            # for saving to csv: update the pixel size with the value from real
            # pixel space and the probability of the current MIRC
            pix_size_MIRC = img_size_original_px_space
            prob_MIRC = prob_most_predictive_crop

            # increase counter of reduced resolution in case the real pixel
            # size stays the same, otherwise set it to 0
            reduced_res_counter = (
                reduced_res_counter +
                1 if prev_img_size_original_px_space == img_size_original_px_space else 0)

            # save image
            data_out.write_to_npz(
                exp_dir,
                img_identifier,
                reduced_res_counter,
                image,
                prob_most_predictive_crop,
                img_size_original_px_space,
                target_list,
            )

            prev_img_size_original_px_space = img_size_original_px_space

        child_crop_size_real_px_space = round(img_size_work * 80 / 100)

        # (1) evaluate image crops from cropping
        # extract patches
        # if child_crop_size_real_px_space is smaller than rf, first extract
        # crops of that size and then artificially blow them up to rf (this is
        # an artificial step to make the algorithm work with BagNets)
        if child_crop_size_real_px_space < rf:
            new_sub_patch_crops = util.extract_crops(
                image, size=child_crop_size_real_px_space
            )
            patches = torch.empty(
                [new_sub_patch_crops.shape[0], new_sub_patch_crops.shape[1], rf, rf],
                device=DEVICE,
            )
            for patch_i in range(new_sub_patch_crops.shape[0]):
                patches[patch_i] = util.get_resized_img(
                    new_sub_patch_crops[patch_i], rf, DEVICE
                )
        # if child_crop_size_real_px_space is equal to or larger than rf,
        # extract crops of size rf
        else:
            patches = util.extract_crops(image, size=rf)

        # evaluate patches
        logits_for_patches = util.get_logits_for_patches(
            patches, rf, model, DEVICE)
        # evaluate crops
        prob_crop, custom_prob_crop = util.get_prob_and_custom_prob_per_crops(
            logits_for_patches,
            img_size_work,
            child_crop_size_real_px_space,
            descendent_specifier,
            target_list,
            rf,
            DEVICE,
        )

        # (2) evaluate image crops of reduced resolution
        # resize the image
        crop_reduced_resolution_real_px_space = util.get_resized_img(
            image, child_crop_size_real_px_space, DEVICE
        )
        # if the resized image is smaller than rf, blow it up to rf (this is an
        # artificial step to make the algorithm work with BagNets)
        if child_crop_size_real_px_space < rf:
            crop_reduced_resolution_work_px_space = util.get_resized_img(
                image, rf, DEVICE
            )
        else:
            crop_reduced_resolution_work_px_space = (
                crop_reduced_resolution_real_px_space
            )
        # evaluate whole image
        patches_reduced_resolution = util.extract_crops(
            crop_reduced_resolution_work_px_space, size=rf
        )
        logits_for_patches_reduced_resolution = util.get_logits_for_patches(
            patches_reduced_resolution, rf, model, DEVICE
        )
        # append to prob_crop and custom_prob
        prob_red_res = util.get_prob_for_correct_classes_of_whole_img(
            logits_for_patches_reduced_resolution, target_list
        )
        custom_prob_red_res = util.get_custom_prob(
            torch.mean(logits_for_patches_reduced_resolution, dim=0)[None, :],
            target_list,
            DEVICE,
        )
        prob_crop.append(prob_red_res)
        custom_prob_crop.append(custom_prob_red_res[0])

        # pick crop that has highest contribution to correct class according to
        # custom probability
        idx_most_predictive_crop = custom_prob_crop.index(
            max(custom_prob_crop))
        prob_most_predictive_crop = prob_crop[idx_most_predictive_crop]

        # The smallest img size that makes sense as a MIRC is 3 - as the sub-MIRC will be
        # of size 2. After that, the 20% reduction would only again yield a
        # pixel size of 2, hence not reducing the MIRC any further
        if img_size_original_px_space == 3:
            print(
                f"breaking at img_size_original_px_space {img_size_original_px_space}"
            )
            break

        round_one = False

    # if there is no MIRC
    if round_one:
        img_size_original_px_space = -1
        reduced_res_counter += 1
    # if there is a MIRC
    else:
        # pick new minimal crop
        image = util.get_most_predictive_crop(
            idx_most_predictive_crop,
            image,
            crop_reduced_resolution_real_px_space,
            custom_prob_crop,
            descendent_specifier,
        )

        # determine image size
        # if img was cropped, increase the counter, else (i.e. if resolution is
        # reduced) don't do anything
        if idx_most_predictive_crop != len(prob_crop) - 1:
            size_original_i += 1
        img_size_original_px_space = size_in_original_px_space_list[size_original_i]
        # increase counter of reduced resolution in case the real pixel size
        # stays the same, otherwise set it to 0
        reduced_res_counter = (
            reduced_res_counter + 1
            if prev_img_size_original_px_space == img_size_original_px_space
            else 0
        )

    # save best sub-MIRC
    data_out.write_to_npz(
        exp_dir,
        img_identifier,
        reduced_res_counter,
        image,
        prob_most_predictive_crop,
        img_size_original_px_space,
        target_list,
    )

    # save data to csv
    data_out.save_data_to_csv(
        exp_dir,
        img_identifier,
        write_or_append,
        pix_size_MIRC,
        prob_MIRC,
        prob_most_predictive_crop,
    )
