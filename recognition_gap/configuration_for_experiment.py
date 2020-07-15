"""
Conditions for experiment. These are variables that you can customize! As they are, they would correspond to the settings of the experiment in the main body.
"""

####################
### YOUR TODO!!! ###
####################
# path to image data. The path should either contain the keyword "Ullman" or "imagenet" depending on which data is represented.
# In case of ImageNet: This path ends in the validation directory which contains all 1000 classes.
# In case of images from Ullman et al. (2016): This path ends in a directory which contains a directory for each of the nine class classes. Each class directory contains the corresponding image from Ullman et al..
data_path = "/gpfs01/bethge/home/jborowski/CHAM_recognition_gap/JOV_publication_git_bethgelab/recognition_gap/images_from_Ullman_et_al_in_color_Judys_selection"
# data_path = "/gpfs01/bethge/data/imagenet-raw/raw-data/val/"

################################
### Conditions of experiment ###
################################
# string indicating which images to use - must match the data_path!
# Either "Ullman" or "ImageNet"
if "Ullman" in data_path:
    Ullman_or_ImageNet = "Ullman"
elif "imagenet" in data_path:
    Ullman_or_ImageNet = "ImageNet"
else:
    raise Exception(
        "You are using neither the data of Ullman et al. nor of ImageNet.")

# The following parameter is only relevant for the images from Ullman et al.: It determines whether all the probabilities of all ImageNet classes per input image are added (i.e. considered as one) or whether they are treated separately.
# Either True or False
list_as_one_class = True

# Which children are considered in the MIRC search? "stride1" menas that every crop with a difference of 1 pixel is considered. "Ullman4" means that only the four corner crops are considered.
# Either "stride1" or "Ullman4"
descendent_specifier = "Ullman4"

# starting and stopping index of images
# if ImageNet stimuli are used, pick a range
if Ullman_or_ImageNet == "ImageNet":
    start_idx = 0
    stop_idx = 1000
# if the images from Ullman et al. are used, the range gets picked
# automatically
else:
    start_idx = 0
    stop_idx = 9
