# This code adds an image as background. The line-drawing has a size of 256 x 256px. A margin of 16px is added to the
# line-drawing. Thus, the resulting images have a size of 288 x 288px.
# author: Christina Funke

import numpy as np
import os
from PIL import Image
from pathlib import Path
from skimage import io, transform, color  # pip3 install scikit-image
import torchvision
from torchvision import transforms

margin = 16


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def change_contrast(lab, alpha):
    m = np.mean(lab, (0, 1))
    x_0 = np.array([50, 0, 0])
    x_alpha = (1 - alpha) * x_0 + alpha * m
    res = (lab - m) * alpha + x_alpha
    return res


def set_seed(method):
    """
    Set seed and number of repetitions. Both depend on the type of the data set. For example a different seed is used
    for test and training set
    :param method: type of the data set [string]
    :return: number of [int]
    """
    if method.endswith("val"):
        np.random.seed(0)
        num_elem = 1400
    elif method.endswith("test"):
        np.random.seed(1)
        num_elem = 2800
    elif method.endswith("train"):
        np.random.seed(2)
        num_elem = 14000
    return num_elem


def getbg_otf(image, contrast="random"):
    """
    This function can be used to add a background image for training data on-the-fly
    :param image: input image
    :param contrast: either 0 or "random"
    :return: output image
    """
    if contrast == "random":
        # be careful: slow! if you want to use random change this:
        # load imagenet in the main script and give the image to the function

        imagenet_data = torchvision.datasets.ImageFolder(
            "/gpfs01/bethge/data/imagenet-raw/raw-data/train"
        )
        my_transform = transforms.Compose(
            [
                transforms.Scale(256 + 2 * margin),
                transforms.CenterCrop(256 + 2 * margin),
            ]
        )
        imagenet_data.transform = my_transform

        rgb = np.array(imagenet_data[np.random.randint(0, 1261174)][0])
        lab = color.rgb2lab(rgb / 255)

        alpha = np.random.rand()
        lab_c = change_contrast(lab, alpha)
        rgb_c = color.lab2rgb(lab_c) * 255

    if contrast == 0:
        # to save time the computation was simplified for contrast0
        rgb_grey = np.array([118.91369957872172, 118.9127086726839, 118.91777938892962])
        rgb_c = np.tile(rgb_grey, (288, 288, 1))
    contour = np.array(image)

    contour = np.pad(
        contour,
        ((margin, margin), (margin, margin), (0, 0)),
        "constant",
        constant_values=255,
    )
    # get mask
    res_grey = contour[:, :, 0]
    img_grey = rgb2gray(rgb_c)
    mask = res_grey >= img_grey
    mask = np.dstack([mask] * 3)
    # add background to rgb_grey
    contour[mask] = rgb_c[mask]
    res = contour.astype(np.uint8)
    return res


def make_full_dataset(top_dir, set_num, debug, all_contrast_levels, imagenet_data):
    """
    generate and save the full data set for a specified variation
    :param top_dir: where to save the images [string]
    :param set_num: number that specifies the variation [int]
    :param debug: generate only seven images [bool]
    :param all_contrast_levels: do all contrast levels? [bool]
    :param imagenet_data: images
    """
    stim_folder = top_dir + "/set" + str(set_num) + "/"
    if set_num == 4:
        stim_folder_black = top_dir + "/set1/"
    if set_num == 5:
        set_num_black = 3
        stim_folder_black = top_dir + "/set" + str(set_num_black) + "/"
    if set_num == 1:
        methods = ["val", "test", "train"] # remove "train" from list to not generate training set
    else:
        methods = ["test"]

    for method in methods:
        if all_contrast_levels:
            if method == "test":
                levels = [0, 0.2, 0.4, 0.6, 0.8, 1]
            if method == "train" or method == "val" or method == "trainmany":
                levels = ["random", 0]
        else:
            levels = [0]

        for level in levels:
            num_elem = set_seed(method)
            if debug:
                num_elem = 2

            # make folder
            new_folder = os.path.join(stim_folder, "contrast" + str(level), method)
            if not Path(new_folder).is_dir():
                os.makedirs(new_folder)
            if not Path(new_folder + "/closed/").is_dir():
                os.mkdir(new_folder + "/closed/")
            if not Path(new_folder + "/open/").is_dir():
                os.mkdir(new_folder + "/open/")

            for num in (np.arange(num_elem)) * 2 + 1:
                if set_num == 4:
                    filename_closed = os.path.join(
                        stim_folder_black,
                        "linedrawing",
                        method,
                        "closed",
                        method + str(num) + ".png",
                    )
                    filename_open = os.path.join(
                        stim_folder_black,
                        "linedrawing",
                        method,
                        "open",
                        method + str(num + 1) + ".png",
                    )
                else:
                    filename_closed = os.path.join(
                        stim_folder,
                        "linedrawing",
                        method,
                        "closed",
                        method + str(num) + ".png",
                    )
                    filename_open = os.path.join(
                        stim_folder,
                        "linedrawing",
                        method,
                        "open",
                        method + str(num + 1) + ".png",
                    )
                if set_num == 5:
                    filename_closed_black = os.path.join(
                        stim_folder_black,
                        "linedrawing",
                        method,
                        "closed",
                        method + str(num) + ".png",
                    )
                    filename_open_black = os.path.join(
                        stim_folder_black,
                        "linedrawing",
                        method,
                        "open",
                        method + str(num + 1) + ".png",
                    )

                if (
                    level == 0
                ):  # shortcut for contrast 0 (grey background), this saves a lot of time
                    rgb_grey = np.array(
                        [118.91369957872172, 118.9127086726839, 118.91777938892962]
                    )
                    rgb_c = np.tile(rgb_grey, (288, 288, 1))
                else:
                    rgb = np.array(imagenet_data[np.random.randint(0, 1261174)][0])
                    # change contrast
                    lab = color.rgb2lab(rgb / 255)
                    if level == "random":
                        alpha = np.random.rand()
                    else:
                        alpha = level
                    lab_c = change_contrast(lab, alpha)
                    rgb_c = color.lab2rgb(lab_c) * 255

                if set_num == 5:
                    with Image.open(filename_closed) as im_closed:
                        with Image.open(filename_closed_black) as im_closed_black:
                            contour = np.array(im_closed)
                            contour = np.pad(
                                contour,
                                ((margin, margin), (margin, margin), (0, 0)),
                                "constant",
                                constant_values=255,
                            )
                            contour_black = np.array(im_closed_black)
                            contour_black = np.pad(
                                contour_black,
                                ((margin, margin), (margin, margin), (0, 0)),
                                "constant",
                                constant_values=255,
                            )

                            # get mask
                            res_grey = contour_black[:, :, 0]
                            img_grey = rgb2gray(rgb_c)
                            mask = res_grey >= img_grey
                            mask = np.dstack([mask] * 3)
                            # add background to rgb_grey
                            contour[mask] = rgb_c[mask]

                            # save it
                            res = contour.astype(np.uint8)
                            filename = os.path.basename(filename_closed)
                            io.imsave(os.path.join(new_folder, "closed", filename), res)

                    with Image.open(filename_open) as im_open:
                        with Image.open(filename_open_black) as im_open_black:

                            contour = np.array(im_open)
                            contour = np.pad(
                                contour,
                                ((margin, margin), (margin, margin), (0, 0)),
                                "constant",
                                constant_values=255,
                            )

                            contour_black = np.array(im_open_black)
                            contour_black = np.pad(
                                contour_black,
                                ((margin, margin), (margin, margin), (0, 0)),
                                "constant",
                                constant_values=255,
                            )

                            # get mask
                            res_grey = contour_black[:, :, 0]
                            img_grey = rgb2gray(rgb_c)
                            mask = res_grey >= img_grey
                            mask = np.dstack([mask] * 3)
                            # add background to rgb_grey
                            contour[mask] = rgb_c[mask]

                            # save it
                            res = contour.astype(np.uint8)
                            filename = os.path.basename(filename_open)
                            io.imsave(os.path.join(new_folder, "open", filename), res)

                else:
                    with Image.open(filename_closed) as im_closed:
                        contour = np.array(im_closed)
                        contour = np.pad(
                            contour,
                            ((margin, margin), (margin, margin), (0, 0)),
                            "constant",
                            constant_values=255,
                        )
                        if set_num == 4:
                            contour = 255 - contour

                        # get mask
                        res_grey = contour[:, :, 0]
                        img_grey = rgb2gray(rgb_c)
                        if set_num == 4:
                            mask = res_grey <= img_grey
                        else:
                            mask = res_grey >= img_grey

                        mask = np.dstack([mask] * 3)

                        # add background to rgb_grey
                        contour[mask] = rgb_c[mask]

                        # save it
                        res = contour.astype(np.uint8)
                        filename = os.path.basename(filename_closed)
                        io.imsave(os.path.join(new_folder, "closed", filename), res)

                    with Image.open(filename_open) as im_open:
                        contour = np.array(im_open)
                        contour = np.pad(
                            contour,
                            ((margin, margin), (margin, margin), (0, 0)),
                            "constant",
                            constant_values=255,
                        )
                        if set_num == 4:
                            contour = 255 - contour

                        # get mask
                        res_grey = contour[:, :, 0]
                        img_grey = rgb2gray(rgb_c)
                        if set_num == 4:
                            mask = res_grey <= img_grey
                        else:
                            mask = res_grey >= img_grey
                        mask = np.dstack([mask] * 3)

                        # add background to rgb_grey
                        contour[mask] = rgb_c[mask]

                        # save it
                        res = contour.astype(np.uint8)
                        filename = os.path.basename(filename_open)
                        io.imsave(os.path.join(new_folder, "open", filename), res)
