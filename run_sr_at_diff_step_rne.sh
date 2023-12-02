#!/bin/bash
device=0
round2=nearest
source .venv/bin/activate

for width in 3 5; do
	for round in stochastic nearest ; do
			for run in {1..5}; do
				CUDA_VISIBLE_DEVICES=$device python mix_precision_train.py -w $width -e $width -g $width -a $width \
					--weight-round $round \
					--error-round $round2 \
					--gradient-round $round2 \
					--activation-round $round2 \
					--learning-rate 0.1 \
					--momentum 0 \
					--batch-size 64 \
					--mix-precision False \
					--epochs 150 \
					--log-path sr_at_diff_stage
			done
	done
done

