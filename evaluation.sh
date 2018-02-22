
#USE ABSOLUTE PATHS THROUGHOUT, add this to cron tab, set it to run every couple minutes, ie */2 * * * * ./~/MLSALT4/evaluation.sh

for f in $(ls -1v checkpoints/*)
do
	n=${f#checkpoints/model.ckpt-}
	echo $n >> trainAccuracies.log
	python evaluate_mc_dropout.py --checkpoint_file_path $f >> trainAccuracies.log
  echo "\n"
	if (( $n % 100000 == 0 )) 
	then
		mv $f checkpoints.save/
	else
		rm $f
	fi
done
