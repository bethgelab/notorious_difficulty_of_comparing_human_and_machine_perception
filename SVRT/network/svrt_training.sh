# script to train resnet on svrt problems

# finetune
for num_train in 28000 1000 100;
	do
	for set in {1..23}
        	do
        	python3 svrt_training.py -net resnet50 -lr 3e-4 -num_trainimages $num_train -set_num $set -save 1 -epoch_multiplier 10
        	python3 svrt_training.py -net resnet50 -lr 1e-4 -num_trainimages $num_train -set_num $set -save 1 -epoch_multiplier 10
        	python3 svrt_training.py -net resnet50 -lr 6e-5 -num_trainimages $num_train -set_num $set -save 1 -epoch_multiplier 10
	done
done


# train from scratch
for num_train in 28000 1000 100;
        do
        for set in {1..23}
                do
                python3 svrt_training.py -net resnet50 -lr 1e-3 -num_trainimages $num_train -set_num $set -save 1 -epoch_multiplier 10 -pretrained 0     
                python3 svrt_training.py -net resnet50 -lr 6e-4 -num_trainimages $num_train -set_num $set -save 1 -epoch_multiplier 10 -pretrained 0
                python3 svrt_training.py -net resnet50 -lr 3e-4 -num_trainimages $num_train -set_num $set -save 1 -epoch_multiplier 10 -pretrained 0
        done
done

