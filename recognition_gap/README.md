# Recognition Gap - Case Study 3/3


## Requirements

- The code runs with python 3.6.3.
- Using a gpu to run the main notebook ```recognition_gap_experiment_JOV.ipynb``` is recommended.
- The required python packages are listed in requirements.txt. We used docker to set up the environment. You can either use the Dockerfile: docker/Dockerfile or load the image from dockerhub: cmfunke/notorious_difficulty_of_comparing_humans_and_machines:recognition_gap


## Usage

To run the **main experiment**, simply execute all cells in the notebook ```recognition_gap_experiment_JOV.ipynb```. It will use scripts from the folder utils, so make sure to download that one as well. Also, please make sure to provide the path to the input images in ```configuration_for_experiment.py```: The directory should contain a directory per class and each class directory should contain the corresponding image(s).

In order to obtain the **visualizations**, execute all cells in the notebooks ```JOV_main_plot_bar.ipynb```, ```JOV_main_visualize_probability_vs_cropsize.ipynb``` or ```JOV_appendix_analysis.ipynb``` in the analysis folder. They will need the scripts data_csv_utils.py and/or data_npz_utils.py and/or plot_utils.py. If you want to reproduce the plots from the manuscript, data_csv_utils.py already contains a dictionary pointing to the corresponding folders of the data. If instead, you would like to visualize data from your own experiments, substitute the corresponding directories.

In order to obtain additional visualizations of the original image and the MIRC, or the search process, execute all cells in the notebooks ```visualize_originalimage_and_MIRC.ipynb``` or ```visualize_probability_vs_cropsize.ipynb``` in the analysis folder respectively. Please note that for the experiments on the ImageNet data, only the csv files are available.


## FAQ

- **Where can I find the images from Ullman et al. (2016)?** The authors of the original study in “Atoms of recognition in human and computer vision” (Ullman et al., 2016) did not agree to providing the stimuli on this GitHub repo and instead ask researchers to contact them directly.


## Author

- Judy Borowski