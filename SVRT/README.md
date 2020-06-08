# Synthetic Visual Reasoning Test - Case Study 2/3


## Requirements

- The code runs with python 3.6.3.
- The required python packages are listed in *requirements.txt*. We used docker to set up the environment. You can either use the Dockerfile: *docker/Dockerfile* or load the image from dockerhub: *cmfunke/notorious_difficulty_of_comparing_humans_and_machines:svrt*
- Gpu to train the networks.


## Usage

The following steps reproduce the experiments and figures. In case you only want to generate the figures, you can skip steps 1 to 4 as all required files are provided in */results/*.

**1. Generate dataset:**
Follow the instructions in *stimuli_gen/doit_CHAM.sh* to generate the datasets for training, validation and testing.

**2. Train the models:**
```bash
cd network
./svrt_training.sh
```
This script takes quite a while. Make sure to run it on a gpu. The script trains ResNet-based models on all 23 problems. For each problem 2 x 3 x 3 = 18 different models are trained:
- 2 starting conditions (either **finetuned** (pretrained on ImageNet) or trained from **scratch** (no pretraining)) 
- 3 different training set sizes (28000, 1000, 100 images)
- 3 different learning rates (learning rates depend on the starting condition)

The resulting accuracies on training and validation set are saved in */results/exp_finetune.csv* and */results/exp_scratch.csv*.

**3. Find best learning rate:**
Execute the first four cells of the notebook *analysis/svrt_results.ipynb* to find the learning rate that leads to the highest accuracy on the validation set. The fourth cell outputs the index of the best learning rate for each problem given the number of training examples and starting condition. 

The resulting indices were manually copied into *network_training/svrt_test.py*, which is used in step 4 to evaluate the models on the test set.

**4. Evaluate on the test set:**
```bash
cd network
python3 svrt_test.py -net resnet50
python3 svrt_test.py -net resnet50 -pretrained 0
```
This script uses the models that were selected in the previous step and evaluates them on the test set. The resulting accuracies are saved in */results/testset_finetune.npy* and */results/testset_scratch.npy*.

**5. Make figures:**
Execute the remaining cells of the notebook *analysis/svrt_results.ipynb*. This reads the files generated in the previous steps and generates the figures shown in the manuscript. 


## Author
Christina Funke