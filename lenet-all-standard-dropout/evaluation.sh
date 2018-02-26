
#USE ABSOLUTE PATHS THROUGHOUT, add this to cron tab, set it to run every couple minutes, ie */2 * * * * ./~/MLSALT4/evaluation.sh
#THIS SHOULD WORK IF checkpoints are saved to: checkpoints/model.ckpt+"-"+str(i), needs a bit of cleaning up to succesfully remove and move checkpoints
#!/bin/bash

mkdir checkpoints.save

for f in $(ls -1v checkpoints/*.index)
do
	check=${f%.*}
	n=${check#checkpoints/model.ckpt-}
	echo $check
	echo $n >> trainingAccuracies.log
	python3 evaluate_mc_dropout_py3.py --checkpoint_file_path $check >> process.log
	#if (( $n % 100000 == 0 )) 
	#then
	#	for file in $(ls -1v $check*)
		#	do
		#	mv $file checkpoints.save/
		#	echo $check | mail -s "Evaluations script" david.r.burt94@gmail.com # PLEASE CHANGE EMAIL
		#	done
	#else
	#	for file in $(ls -1v $check*)
	#		do
	#		rm $file
	#		done
	#fi
done
