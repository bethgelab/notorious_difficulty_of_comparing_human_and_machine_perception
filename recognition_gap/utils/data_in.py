# This script contains all functions that have to do with the input data.

import os

# torch imports
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# custom imports
import configuration_for_experiment as config

# Dictionary for the images from Ullman et al.. It contains custom class names as keys and a list of numbers corresonding to the ImageNet class numbers (as for ResNets or BagNets).
# The classes were handpicked by Judy Borowski and Christina M. Funke to their best judgment.
# The selection is different from the one used in the machine experiments
# by Ullman et al..
imagenet_classes_for_images_from_Ullman_et_al = {
    "fly": [308],
    "ship": [403, 628, 510, 554, 625],
    "eagle": [22, 21],
    "glasses": [836, 837],
    # 836: sunglass = convex lens that focuses the rays
    # of the sun; used to start a fire. 837: spectacles
    # that are darkened or polarized to protect the
    # eyes from the glare of the sun.
    # source: http://wordnetweb.princeton.edu/
    # accessed in April 2020
    "bike": [444, 665, 671, 612, 870],
    "suit": [834, 906],
    "plane": [404],
    "horse": [339, 603],
    "car": [817, 468, 511, 609, 751, 627, 654, 656, 407, 436],
    # "eye":[] # excluded because the eye image from Ullman et al. does not have a corresponding class in ImageNet
}


class ImageFolderWithTargetListAndPaths(datasets.ImageFolder):
    """Custom data set that includes list of target(s) and image file paths. Extends torchvision.datasets.ImageFolder.
    """

    # override the __getitem__ method. This is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns: a tuple containing the
        # image and the target
        original_tuple = super(
            ImageFolderWithTargetListAndPaths,
            self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # create the target: a list with all ImageNet classes that shall be considered True for an image.
        # In the case of ImageNet, this is trivial: Just put the target label
        # into a list.
        if "imagenet" in path:
            target_list = [int(original_tuple[1])]
        # In the case of the images from Ullman et al., the target class(es)
        # is(are) obtained from a predefined dictionary
        elif "Ullman" in path:
            target_list = imagenet_classes_for_images_from_Ullman_et_al[
                path.split(os.path.sep)[-2]
            ]
        else:
            raise Exception(
                "You are using neither the data of Ullman et al. nor of ImageNet."
            )

        # make a new tuple that includes the image, the target_list and the
        # path
        tuple_with_target_list_and_with_path = (
            original_tuple[0], target_list, path)

        return tuple_with_target_list_and_with_path


def get_preprocessed_data_set(data_path):
    """Preprocess the data set and return it. Because a list of targets and the paths
    to the images are needed later, the class ImageFolderWithTargetListAndPaths
    is used. Preprocessing follows standard ResNet-preprocessing.

    Args:
        data_path:   path to folder where data is stored.

    Returns:
        data_set:    returns data set which includes image file paths
    """

    # standard ResNet-preprocessing
    data_set = ImageFolderWithTargetListAndPaths(
        data_path,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    return data_set


def get_data_loader(Ullman_or_ImageNet):
    """The goal of this function is to return the data loader. Therefore, the path to the images is obtained, the data set is preprocessed and the the loader is created.
    Note that the returned targets for the images from Ullman et al. are a subjective selection and is determined in imagenet_classes_for_images_from_Ullman_et_al.

    Args:
        Ullman_or_ImageNet: string indicating which images to use

    Returns:
        data_loader:        loader with data

    """
    data_set = get_preprocessed_data_set(config.data_path)

    # make data_loader deterministic despite shuffling
    torch.manual_seed(2809)

    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=1,  # only one image is processed at a time
        shuffle=True,  # to guarantee different classes from ImageNet
        num_workers=1,
        pin_memory=True,
    )

    return data_loader
