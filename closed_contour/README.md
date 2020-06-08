# Closed Contour Detection - Case Study 1/3


## Requirements

- The code runs with python 3.6.3.
- The required python packages are listed in *requirements.txt*. We used docker to set up the environment. You can either use the Dockerfile: *docker/Dockerfile* or load the image from dockerhub: *cmfunke/notorious_difficulty_of_comparing_humans_and_machines:closed_contour*
- Gpu to train the networks is recommended.

## Usage

The following steps reproduce the experiments and figures.

**1. Generate dataset:**
Follow the instructions in *stimuli_gen/closed_contour_dataset.ipynb* to generate the closed contour dataset.

**2. Train the ResNet-50-based models:** 
```bash
cd network
./cc_training.sh
```
This script takes quite a while. Make sure to run it on a gpu.

**3. Generalisation performance:**
```bash
cd network
./cc_generalisation_csv.sh
```
This script computes the predictions for the different generalisation test sets. The results are provided in */results/imagelevel/* Use the notebook *analysis/generalisation_plots.ipynb* to make the figures.

**4. Adversarial examples:**
The notebook *analysis/adversarial_examples.ipynb* generates the adversarial examples.

**5. BagNets and heatmaps:**
```bash
cd network
./cc_training_BagNet.sh
```
Please note that training this BagNet takes quite a while. Make sure to run it on a gpu. When training BagNet it is not necessary to generate the training set, as it is generated online. Follow remark 1 in *stimuli_gen/closed_contour_dataset.ipynb* to generate only test and validation set.

The notebook *analysis/heatmaps_from_BagNet.ipynb* generates the heatmaps and the histogram of the logits.


## Authors
Christina Funke and Judy Borowski