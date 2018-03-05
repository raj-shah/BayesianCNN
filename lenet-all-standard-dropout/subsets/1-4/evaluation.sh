
#!/bin/bash

for n in `seq 0 10000 10000000`
do
	check=/home/si318/Desktop/MLSALT4/lenet-all-standard-dropout/checkpoints/model.ckpt-$n
	python3 mc_drop.py --checkpoint_file_path $check --T 50 
done

