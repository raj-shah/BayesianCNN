
#1 Make the directory (using 'mkdir checkpoints.save') checkpoints.save in the same directory as this file
#2 in command line type 'EDITOR=nano crontab -e', this will open a buffer
#2 in the buffer, type */3 * * * * ./PATH-TO/evaulation.sh, use the absolute path to this script here, save and close the crontab buffer, this will run the script every 3 minutes
#3 type in crontab -l, this should return */3 * * * * ./evaulation.sh, indicating theat the cronjob has been set
#4 Train the network, the evaluation script will be run every 3 minutes, moving stuff over to a new directory
#!/bin/bash

#mkdir checkpoints.save

for f in $(ls -1v /home/si318/Desktop/MLSALT4/lenet-all-standard-dropout/checkpoints/*.index)
do
	check=${f%.*}
	echo $f
	n=${check#/home/si318/Desktop/MLSALT4/lenet-all-standard-dropout/checkpoints/model.ckpt-}	echo $check
	echo $n >> trainingAccuracies.log
	python3 mc_drop.py --checkpoint_file_path $check >> process.log
	#if (( $n % 10000 == 0 )) #This is currently trivially sa
	#then
		#for file in $(ls -1v $check*)
			#do
	#mv -f $file checkpoints.save/
			#echo $check | mail -s "Evaluations script" david.r.burt94@gmail.com # PLEASE CHANGE EMAIL
			#done
	#else
	#	for file in $(ls -1v $check*)
	#		do
	#		rm $file
	#		done
	#fi
done
