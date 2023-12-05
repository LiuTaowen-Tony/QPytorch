#!/bin/bash
round=stochastic
source .venv/bin/activate


for man_width in 3; do
for round2 in stochastic nearest ; do
for loss_scale in 1 8 64 256 1024; do
for batch_size in 64; do
for lr in 0.05; do
for exp_width in 2 3 5 8; do
for run in 0; do
	device=0
	if [ $exp_width -lt 4 ] 
	then
		device=1
	fi
	CUDA_VISIBLE_DEVICES=$device python mix_precision_train.py \
		-w $man_width -e $man_width -g $man_width -a $man_width \
		--weight-ew $exp_width \
		--error-ew $exp_width \
		--gradient-ew $exp_width \
		--activation-ew $exp_width \
		--loss-scale $loss_scale \
		--log-path loss_scale_log \
		--epochs 150 \
		--weight-round $round2 \
		--error-round $round2 \
		--gradient-round $round2 \
		--activation-round $round2 \
		--learning-rate 0.1 \
		--momentum 0 \
		--batch-size $batch_size \
		--mix-precision True &
done
done
	wait
done
done
done
done
done

