# This script contains the values from the human experiment from Ullman et
# al. (2016), pointers to the directories of the data of our experiments
# and functions to read in the csv files.

import os
import glob
import pandas as pd

# data from Ullman et al. (2016)
recognitionGapsHuman = 0.71
recognitionGapsHumanStd = 0.05

# dictionary with description of experimental condition as key and lists
# of directory paths for the corresponding data as values
exp_dir_dict = {
    "Ullman_stride-1_jointClasses": [
        os.path.join(
            "..",
            "figures_and_data_from_experiments",
            "exp_792020Ullman_list_as_one_classTrue_startidx0_stopidx9_stride1_v0")],
    "Ullman_Ullman4_jointClasses": [
        os.path.join(
            "..",
            "figures_and_data_from_experiments",
            "exp_792020Ullman_list_as_one_classTrue_startidx0_stopidx9_Ullman4_v0")],
    "Ullman_stride-1_separateClasses": [
        os.path.join(
            "..",
            "figures_and_data_from_experiments",
            "exp_792020Ullman_list_as_one_classFalse_startidx0_stopidx9_stride1_v0")],
    "Ullman_Ullman4_separateClasses": [
        os.path.join(
            "..",
            "figures_and_data_from_experiments",
            "exp_792020Ullman_list_as_one_classFalse_startidx0_stopidx9_Ullman4_v0")],
    "IN_stride-1": [
        os.path.join(
            "..",
            "figures_and_data_from_experiments",
            "exp_7102020ImageNet_list_as_one_classFalse_startidx0_stopidx1000_stride1_v0")],
    "IN_Ullman4": [
        os.path.join(
            "..",
            "figures_and_data_from_experiments",
            "exp_7102020ImageNet_list_as_one_classFalse_startidx0_stopidx1000_Ullman4_v0")],
}


# Functions
def load_csv_to_dataframe(path_to_file, metric_name):
    """Load csv to dataframe and return dataframe.abs

    Args:
        path_to_file:  relative path to csv file.
                       It contains the data in the format
                       img_identifier,value
        metric_name:   name of metric

    Returns:
        new_dataframe: dataframe with the column names "img_identifier" and "value"
    """
    new_dataframe = pd.read_csv(path_to_file, header=None)
    new_dataframe.columns = ["img_identifier", metric_name]

    return new_dataframe


def get_df_from_all_csv_files(exp_dir):
    """Return one dataframe with the data from all csv-files in exp_dir.

    Args:
        exp_dir:  directory from which to read all csv-files in

    Returns:
        final_df: final dataframe that contains the joined data from all csv files"""

    # list of paths to all csv files
    csv_files_list = glob.glob(os.path.join(exp_dir, "*.csv"))

    # iterate through all csv files
    for path_i, path_to_file in enumerate(csv_files_list):
        metric_name = path_to_file.split(os.path.sep)[-1][:-4]
        # load data from each csv file to a separate dataframe and join the
        # data.
        if path_i == 0:
            final_df = load_csv_to_dataframe(path_to_file, metric_name)
        else:
            new_df = load_csv_to_dataframe(path_to_file, metric_name)
            final_df = final_df.join(new_df[metric_name])

    return final_df


def get_df_from_exp_dir_list(exp_dir_list):
    """Get data for one experimental condition which is saved in the directory(ies) that are contained
    in the list of directories exp_dir_list.

    Args:
        exp_dir_list: list of paths to directories

    Returns:
        all_data_df:  one dataframe
    """

    # iterate over all folders within one experimental condition
    for exp_dir_i, exp_dir in enumerate(exp_dir_list):
        # get one dataframe with the values from all csv files
        # in case several folders have to be read in, concatenate the
        # dataframes.
        if exp_dir_i == 0:
            all_data_df = get_df_from_all_csv_files(exp_dir)
        else:
            another_all_data_df = get_df_from_all_csv_files(exp_dir)
            all_data_df = pd.concat(
                [all_data_df, another_all_data_df], ignore_index=True)

    return all_data_df


def get_df_with_data_from_real_MIRCs_only(all_data_df):
    """Clean the data such that only data from images which yielded MIRCs is contained

    Args:
        all_data_df: dataframe with data from one experimental conditions

    Returns:
        all_data_df_real_MIRCs: dataframe with the data of those images that yielded real MIRCs only
    """

    # create a mask to only consider those data points that contain real
    # MIRCs. This means that the recognition gap is larger than 0.
    mask_real_MIRCs = all_data_df.rec_gap > 0
    # create a dataframe with the data of those images that yielded real MIRCs
    # only.
    all_data_df_real_MIRCs = all_data_df[mask_real_MIRCs]
    return all_data_df_real_MIRCs
