#!/bin/bash

for n in `seq 0 5000 245000`
do
	check=/home/si318/Desktop/MLSALT4/lenet-all-standard-dropout/trained_onnoise_02/checkpoints/model.ckpt-$n
	python std_drop_eval.py --checkpoint_file_path $check
done

