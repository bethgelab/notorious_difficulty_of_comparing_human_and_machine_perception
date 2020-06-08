# Evaluate model performance on variations of the data set
# author: Christina Funke

# main experiment with ResNet50
python3 cc_generalisation_csv.py -net resnet50 -crop_margin 1 -exp_name resnet50_lr0.0003_numtrain14000_augment1_unique_batchsize64_optimizerAdam_contrast0_reg0_otf0_cropmargin1_5152019_v0

# appendix: varying contrast levels
python3 cc_generalisation_csv.py -net resnet50 -exp_name resnet50_lr0.0003_numtrain14000_augment1_unique_batchsize64_optimizerAdam_contrastrandom_1292019_v0