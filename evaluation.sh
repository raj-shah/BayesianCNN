
#USE ABSOLUTE PATHS THROUGHOUT, add this to cron tab, set it to run every couple minutes, ie */2 * * * * ./~/MLSALT4/evaluation.sh
#THIS SHOULD WORK IF checkpoints are saved to: checkpoints/model.ckpt+"-"+str(i), needs a bit of cleaning up to succesfully remove and move checkpoints
#!/bin/bash

for f in $(ls -1v checkpoints/*.index)
do
	check=${f%.*}
	n=${check#checkpoints/model.ckpt-}
	echo $check
	echo $n >> trainAccuracies.log
	python evaluate_mc_dropout.py --checkpoint_file_path $check >> trainAccuracies.log
done
  #echo "\n"
	#if (( $n % 100000 == 0 )) 
	#then
	#	mv $f checkpoints.save/
	#else
		#rm $f
	#fi
#done
