# bash scripts to train ResNets on closed contour data set
# author: Christina Funke

# main experiment with ResNet50
python3 cc_training.py -net resnet50 -lr 0.0003 -num_trainimages 14000 -unique unique -contrast contrast0 -otf 0 -crop_margin 1

# appendix: varying contrast levels
python3 cc_training.py -net resnet50 -lr 0.0003 -num_trainimages 14000 -unique unique -source share -contrast contrastrandom


