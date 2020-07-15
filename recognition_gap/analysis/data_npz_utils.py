# This script contains functions to deal with the data saved in npz files.

import os
import glob
from collections import OrderedDict
from collections import namedtuple
import numpy as np
import utils.util as util


def get_list_of_data_points_to_plot(exp_dir,
                                    all_datapoints_including_nonMIRC=True):
    """Create and return a list of paths to those directories
    whose data should be plotted. Only those data points that have a MIRC
    will be plotted.

    Args:
        exp_dir:                          path to directory w/ experiment
        all_datapoints_including_nonMIRC: flag to determine whether all datapoints shall be included, i.e. including those that did not yield MIRCs.
                                          default: True

    Returns:
        data_point_to_plot_list:          list of data points to plot
    """

    list_all_files_in_exp_dir = glob.glob(os.path.join(exp_dir, "*"))

    data_point_to_plot_list = []
    # loop through all files in the experiment directory
    for this_file_specific_to_img_class in list_all_files_in_exp_dir:
        # exclude all files - we're only interested in folders
        if not any(
            word in this_file_specific_to_img_class for word in [
                ".csv",
                ".png",
                ".svg"]):
            npz_files_list = glob.glob(
                os.path.join(
                    this_file_specific_to_img_class,
                    "*.npz"))
            # the default is that all datapoints (i.e. also nonMIRCs) shall be
            # included, hence set flag to append the current datapoint
            append_bool = True
            # if only datapoints that contain MIRCs shall be included, set flag
            # to False if there is no MIRC for the current datapoint
            if not all_datapoints_including_nonMIRC:
                # loop through all files
                for this_npz_file in npz_files_list:
                    # if one file contains "-1", then that means that that
                    # datapoint does not have a MIRC
                    if "-1" in this_npz_file:
                        append_bool = False
            # if a datapoint should be appendid, append its path
            if append_bool:
                data_point_to_plot_list.append(this_file_specific_to_img_class)
    return data_point_to_plot_list


# define namedtuple for dictionaries
Patch = namedtuple(
    "Patch",
    "image probability crop_size target_list")


def get_img_class_dict_all_data(
        data_point_to_plot_list,
        exp_dir,
        img_identifier):
    """obtain all data for one datapoint, and save all this data into an ordered dictionary.

    Args:
        data_point_to_plot_list: list of data points to plot
        exp_dir:                 path to experiment directory
        img_identifier:          string describing the datapoint, e.g. 'plane_INclass404'

    Returns:
        img_class_dict:          ordered dictionary
                                 keys are specific to img_identifier, pixel size of the crop and the reduced_res_counter, which is a counter indicating how many times the resolution has been reduced (but the size of the crop in real pixel space does not change)
                                 values are a namedtuple containing the image crop, the probability of that image crop, the crop's size in real pixel space and the target_list

    """

    # list of paths to all npz files
    npz_files_list = glob.glob(os.path.join(exp_dir, img_identifier, "*.npz"))
    size_in_real_pixels_list = util.get_size_in_original_px_space_list()
    img_class_dict = {}
    # loop through pixel sizes
    for this_size in size_in_real_pixels_list:
        # loop through reduced_res_counters
        for reduced_res_counter in range(10):
            imgs_dict_key = f"{img_identifier}_{this_size}_{reduced_res_counter}"
            npz_name = f"{imgs_dict_key}.npz"
            path_to_npz = os.path.join(exp_dir, img_identifier, npz_name)
            # if a file exists, load the data and pass it to dictionary
            if os.path.exists(path_to_npz):
                data = np.load(path_to_npz)
                img_class_dict[imgs_dict_key] = Patch(
                    image=data["image"],
                    probability=data["probability"],
                    crop_size=data["crop_size"],
                    target_list=data["target_list"]
                )

    return img_class_dict
