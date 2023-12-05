#!/bin/bash
device=2
round=stochastic
source .venv/bin/activate

for width in 3 5; do
	for round in stochastic nearest ; do
		for round2 in stochastic nearest ; do
			#for run in {1..10}; do
				CUDA_VISIBLE_DEVICES=$device python mix_precision_train.py -w $width -e $width -g $width -a $width \
					--weight-round $round \
					--error-round $round2 \
					--gradient-round $round2 \
					--activation-round $round2 \
					--learning-rate 0.1 \
					--momentum 0 \
					--batch-size 64 \
					--mix-precision False 
			#done
		done
	done
done

for bs in 8 16 32; do
	#for lr in 0.1 0.03 0.01; do
		for ww in 1 2 3; do
		 for aw in 1 2 3; do
		 for ew in 1 2 3; do
		 for gw in 1 2 3; do
			# for run in {1..10}; do
				for round in nearest stochastic; do
					CUDA_VISIBLE_DEVICES=$device python mix_precision_train.py -w $ww -e $ew -g $gw -a $aw \
						--weight-round $round \
						--error-round $round \
						--gradient-round $round \
						--activation-round $round \
						--learning-rate 0.05 \
						--momentum 0 \
						--batch-size $bs \
						--mix-precision True 
				done
			# done
		 done
		 done
		 done
		done
	# done
done

