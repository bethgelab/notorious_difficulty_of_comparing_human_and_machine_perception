# bash script to train BagNet on closed contour data set
# author: Judy Borowski

# main experiment with ResNet50
python3 cc_training.py -net bagnet32 -lr 0.0001 -num_trainimages 280000 -unique pairs -batch_size 8 -optimizer adabound -n_epochs 200 -contrast contrast0 -regularization 0 -otf 1 -crop_margin 1