#!/bin/bash

#  This file can be used to generate a test, trainings and validation set.
#  It relies on the code of Francois Fleuret <francois.fleuret@idiap.ch>.
#  First download their code base from http://www.idiap.ch/~fleuret/svrt/ 
#  and move it into this folder. Then run: ./doit_CHAM.sh

nb_samples_test=5600
nb_samples_train=28000
nb_samples_val=2800

problem_list=$*

[[ ${problem_list} ]] || problem_list=$(echo {1..23})

set -e

make -j -k vision_test

for problem_number in ${problem_list}; do
    result_dir_test=../stimuli/problem_${problem_number}/test/
    result_dir_train=../stimuli/problem_${problem_number}/train/
    result_dir_val=../stimuli/problem_${problem_number}/val/
    mkdir -p ${result_dir_train}
    mkdir -p ${result_dir_test}
    mkdir -p ${result_dir_val}


    ./vision_test \
        --problem_number=${problem_number} \
        --nb_train_samples=${nb_samples_train} \
        --result_path=${result_dir_train} \
        --random_seed=17 \
        write-samples

    ./vision_test \
        --problem_number=${problem_number} \
        --nb_train_samples=${nb_samples_val} \
        --result_path=${result_dir_val} \
        --random_seed=2 \
        write-samples

    ./vision_test \
        --problem_number=${problem_number} \
        --nb_train_samples=${nb_samples_test} \
        --result_path=${result_dir_test} \
        --random_seed=3 \
        write-samples
done
