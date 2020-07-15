# This script contains functions that have to do with plotting.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

# custom imports
import data_csv_utils
import utils.util as util


def hide_right_and_top_spine(ax):
    """Hide the right and top spines

    Args:
        ax: axes of plot
    """
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


################################################
####### specific to figures showing bars #######
################################################

# width for bars
width = 0.35

# colors in analysis plots
colors = [
    "chartreuse",
    "seagreen",
    "deepskyblue",
    "royalblue",
    "darksalmon",
    "sienna",
    "white",
]


def plot_human_rec_gap_as_horizonal_bar(n_experimental_conditions):
    """plot the human data as a horizontal bar

    Args:
        n_experimental_conditions: value to determine the width of the horizontal bar
    """

    plt.plot(
        [0 - 0.5, n_experimental_conditions - 1 + 0.5],
        [data_csv_utils.recognitionGapsHuman, data_csv_utils.recognitionGapsHuman],
        "k-",
        label="humans",
    )
    plt.axhspan(
        data_csv_utils.recognitionGapsHuman -
        data_csv_utils.recognitionGapsHumanStd,
        data_csv_utils.recognitionGapsHuman +
        data_csv_utils.recognitionGapsHumanStd,
        0 -
        0.5,
        n_experimental_conditions -
        1 +
        0.5,
        color="gray",
        alpha=0.3,
    )


################################################
## specific to figures of probability vs crop ##
################################################

size = 0.251  # size of crops under x-ticks

color = ["orange", "green", "blue"]


def get_dict_of_dict_with_imagenet_number_wordnetID_word():
    """Create a dictionary with the following information:
    imagenetnumber, wordnetID, word.
    The structure is as follows:
    0: {"word": "tench, Tinca tinca", "wordnetID": "n01440764"},
    1: {"word": "goldfish, Carassius auratus", "wordnetID": "n01443537"}, ...
    """

    imagenetnumber_wordnetID_word_dict = {}
    for imagenetnumber, line in enumerate(
            open("categories.txt", "r")):
        wordnetID = line.split()[0]
        word = line.split(f"{wordnetID} ")[1].replace("\n", "")
        imagenetnumber_wordnetID_word_dict[imagenetnumber] = {
            "wordnetID": wordnetID, "word": word}

    return imagenetnumber_wordnetID_word_dict


def customize_axes(ax, crop_probability, xaxis_label_coord_x, orig_px_size):
    """Adjust the axes for the figure showing probability vs crop.

    Args:
        ax:                  axes of plot
        crop_probability:    list of each crop's probability that should be plotted
        xaxis_label_coord_x: position of x-coordinate for x-axis label
        orig_px_size:        list of original pixel sizes to plot as labels of x-ticks
    """

    ax.set_ylim([-0.07, 1.07])
    ax.set_ylabel("p(correct class)")
    ax.set_xlabel("patch size")
    ax.xaxis.set_label_coords(xaxis_label_coord_x, -0.17)
    ax.set_xticks(np.arange(len(crop_probability)))
    ax.set_xticklabels(orig_px_size)


def plot_recognition_criterion(ax, crop_probability):
    """plot the recognition criterion

    Arg:
        crop_probability: list of each crop's probability that should be plotted
    """

    ax.plot(
        [0, len(crop_probability) - 1],
        [0.5, 0.5],
        color="gray",
        linestyle="--",
        label="recognition criterion",
    )


def plot_crops_below_xaxis(
        fig,
        ax,
        img_class_dict,
        y_offset=0,
        space_between_ticks=0.037,
        color_counter=0,
        list_of_keys_for_plotting=[]):
    """add images below x-axis

    Args:
        fig:                       figure
        ax:                        axes of plot
        img_class_dict:            dictionary with all data, e.g. a key: "glasses_INclass836_224_0"
        y_offset:                  offset in y-direction. Only non-zero, when several datapoints are plotted into one plot
        space_between_ticks:       space between ticks to properly space the crops under the x-axis
        color_counter:             counter to index the color's array
        list_of_keys_for_plotting: list of keys whose data should be plotted (e.g. for Figure 3A (JOV_main_visualize_probability_vs_cropsize.ipynb) in the manuscript, only the first instance of a new pixel size is plotted)
    """

    # in case no list_of_keys_for_plotting was passed in the function, create
    # it with all keys of img_class_dict
    if len(list_of_keys_for_plotting) == 0:
        list_of_keys_for_plotting = list(img_class_dict.keys())

    # customize positions
    xl, yl, _, _ = np.array(ax.get_position()).ravel()
    x_new = xl - 0.09
    y_new = yl - 0.4 - y_offset

    # loop through the keys that should be plotted
    for number, cur_key in enumerate(list_of_keys_for_plotting):
        ax1 = fig.add_axes(
            [x_new + space_between_ticks * number, y_new, size, size])
        # customize axes
        for child in ax1.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_linestyle("--")
                child.set_linewidth(1.5)
                child.set_color(color[color_counter])
        ax1.set_xticks([])
        ax1.set_yticks([])
        # plot the crop
        imgplot = ax1.imshow(
            util.undo_normalization(
                img_class_dict[cur_key].image.transpose(
                    1, 2, 0)))


def plot_probabilities(
        ax,
        crop_probability,
        probability_label,
        color_counter=0):
    """plot the probabilities of the crops

    Args:
        ax:                axes of plot
        crop_probability:  list of each crop's probability that should be plotted
        probability_label: label that corresponds to the crop's probability of a certain datapoint
        color_counter:     counter to index the color's array
    """
    ax.plot(
        crop_probability,
        label=probability_label,
        color=color[color_counter])


def get_list_of_keys_for_plotting(img_class_dict):
    """obtain list that contains the zero'th entry of a new pixel size,
    except for the last pixel size. For the last pixel size, pick the last entry.

    Args:
        img_class_dict:            dictionary with all data, e.g. a key: "glasses_INclass836_224_0"

    Returns:
        list_of_keys_for_plotting: list of keys whose values will be plotted

    """
    key_list = list(img_class_dict.keys())
    # get last pixel size
    last_px_size = key_list[-1].split("_")[-2]
    # create list w/ pixel sizes to plot
    list_of_keys_for_plotting = []
    for key_cur in key_list:
        # For all but the last pixel size, pick the first entry ("0").
        if ((last_px_size not in key_cur) and (key_cur[-1] == "0")):
            list_of_keys_for_plotting.append(key_cur)
        # For the last pixel size, pick the last entry.
        elif (last_px_size in key_cur):
            list_of_keys_for_plotting.append(key_list[-1])
            break

    return list_of_keys_for_plotting


################################################
######## specific to saving single crop ########
################################################


def plot_and_save_singe_crop(
        crop,
        exp_dir_MIRCs_and_original_images,
        img_identifier,
        original_or_MIRC):
    """Plot and save original or MIRC image

    Args:
        crop:                              image to be plotted
        exp_dir_MIRCs_and_original_images: path to directory of the original images and the final MIRCs
        img_identifier:                    string to identify an image and its correct class
                                            e.g. glasses_INclass836 or n02834397_ILSVRC2012_val_00019315
        original_or_MIRC:                  string indicating whether "MIRC" or "original" image is plotted
    """

    fig, ax = plt.subplots(1, 1)
    ax.imshow(
        util.undo_normalization(
            crop.transpose(1, 2, 0)
        )
    )
    ax.set_axis_off()
    filename_MIRC = os.path.join(
        exp_dir_MIRCs_and_original_images,
        f"{img_identifier}_{original_or_MIRC}.png")
    fig.savefig(
        filename_MIRC,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
