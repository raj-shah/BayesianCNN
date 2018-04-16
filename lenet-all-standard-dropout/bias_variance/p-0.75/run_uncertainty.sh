#!/bin/bash
for f in 1 2 3 4 5
do
for t in 50
do
for p in 0.25 .5 .6 .75 .9
do
    
    python uncertainty.py --checkpoint_file_path checkpoints/model.ckpt-240000 --T $t --prob $p
done
done
done
