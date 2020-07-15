# This script contains all functions that have to do with the saving data
# from the MIRC-search.


import os
import datetime
import numpy as np


def make_exp_dir(
        Ullman_or_ImageNet,
        list_as_one_class,
        start_idx,
        stop_idx,
        descendent_specifier):
    """Create directory to save data from current experiment, and return the path to it

    Args:
        All args are specifications of the experiment. See their description in configuration_for_experiment.py.

    Returns:
        exp_dir: path to experiment directory

    """

    # add the date to the experiment name
    now = datetime.datetime.now()
    datetime_identifier = f"{now.month}{now.day}{now.year}"

    # As several experiments might be run on the same day,
    # a number corresponding to the version will be appended
    # 100 was chosen as it is unlickely to run more than 100 experiments on a
    # singl day
    for version in range(100):
        exp_dir = os.path.join(
            "figures_and_data_from_experiments",
            (
                f"exp_"
                f"{datetime_identifier}"
                f"{Ullman_or_ImageNet}"
                f"_list_as_one_class{list_as_one_class}"
                f"_startidx{start_idx}"
                f"_stopidx{stop_idx}"
                f"_{descendent_specifier}"
                f"_v{version}"
            ),
        )
        # if exp_dir does not exist yet, create it and break the loop
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
            break

    return exp_dir


def make_dir_original_img_and_MIRC(exp_dir):
    """Make a new directory where the original images and the final MIRCs are saved, and return the path to it.

    Args:
        exp_dir:                           path to experiment directory

    Returns:
        exp_dir_MIRCs_and_original_images: path to directory of the original images and the final MIRCs
    """

    exp_dir_MIRCs_and_original_images = os.path.join(
        exp_dir, "MIRCs_and_original_images"
    )
    if not os.path.exists(exp_dir_MIRCs_and_original_images):
        os.makedirs(exp_dir_MIRCs_and_original_images)

    return exp_dir_MIRCs_and_original_images


def write_to_npz(
    exp_dir,
    img_identifier,
    reduced_res_counter,
    new_image_cuda,
    prob_most_predictive_crop,
    img_size_real_px_space,
    target_list,
):
    """Save data in uncompressed format .npz.

    Args:
        exp_dir:                      path to experiment directory
        img_identifier:               string describing the datapoint, e.g. 'plane_INclass404'
        reduced_res_counter:          counter indicating how many times the resolution has been reduced (but the size of the crop in real pixel space does not change)
        new_image_cuda.cpu().numpy(): numpy.ndarray
        prob_most_predictive_crop:    numpy.float64
        img_size_real_px_space:       int
    """
    # check that the folder in exp_dir that is specific to the img and the
    # class exists, otherwise create it
    img_class_dir = os.path.join(exp_dir, img_identifier)
    if not os.path.exists(img_class_dir):
        os.makedirs(img_class_dir)

    filename = f"{img_identifier}_{img_size_real_px_space}_{reduced_res_counter}.npz"
    path_to_file = os.path.join(img_class_dir, filename)
    np.savez(
        path_to_file,
        image=new_image_cuda.cpu().numpy(),
        probability=prob_most_predictive_crop.cpu().numpy(),
        crop_size=img_size_real_px_space,
        target_list=np.asarray(target_list),
    )


def save_to_csv(exp_dir, img_identifier, write_or_append, file_name, value):
    """save value to csv file

    Args:
        exp_dir:         path to experiment directory
        img_identifier:  string describing the datapoint, e.g. 'plane_INclass404'
        write_or_append: string determining whether the file is written to for the first time ("w") or appended to ("a")
        file_name:       name of csv-file
        value:           value to be saved in csv-file

    """

    with open(os.path.join(exp_dir, f"{file_name}"), write_or_append) as f:
        f.write(f"{img_identifier}, {value}\n")


def save_data_to_csv(
        exp_dir,
        img_identifier,
        write_or_append,
        pix_size_MIRC,
        prob_MIRC,
        prob_sub_MIRC):
    """Save data to csv files.
    This is repetetive given that the data is also stored to npz-files. However, the csv-format shows the results in a quickly readable format and is hence helpful for debugging.
    In theory, it would not have been necessary to save the three values recognition gap, probability of MIRC and probability of sub-MIRC. One of them could have been left out.

    Args:
        exp_dir:         path to experiment directory
        img_identifier:  string describing the datapoint, e.g. 'plane_INclass404'
        write_or_append: string determining whether the file is written to for the first time ("w") or appended to ("a")
        pix_size_MIRC:   pixel size in real pixel space of MIRC
        prob_MIRC:       probability of MIRC
        prob_sub_MIRC:   probability of sub-MIRC
    """

    rec_gap = prob_MIRC - prob_sub_MIRC
    save_to_csv(
        exp_dir,
        img_identifier,
        write_or_append,
        "rec_gap.csv",
        rec_gap.item())

    save_to_csv(
        exp_dir,
        img_identifier,
        write_or_append,
        "pix_size_MIRC.csv",
        pix_size_MIRC)

    prob_MIRC = prob_MIRC
    save_to_csv(
        exp_dir,
        img_identifier,
        write_or_append,
        "prob_MIRC.csv",
        prob_MIRC.item())

    save_to_csv(
        exp_dir,
        img_identifier,
        write_or_append,
        "prob_subMIRC.csv",
        prob_sub_MIRC.item(),
    )
