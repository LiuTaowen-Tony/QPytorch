#!/bin/bash
round=stochastic
source .venv/bin/activate


for man_width in 3; do
for round2 in stochastic nearest ; do
for loss_scale in 256 1024 2048 4096; do
for batch_size in 64; do
for lr in 0.05; do
for bk_exp_width in 3 4; do
for fw_exp_width in 2 3 4; do
for run in 0; do
	device=0
	if [ $bk_exp_width -lt 4 ] 
	then
		device=1
	fi
	CUDA_VISIBLE_DEVICES=$device python mix_precision_train.py \
		-w $man_width -e $man_width -g $man_width -a $man_width \
		--weight-ew $fw_exp_width \
		--error-ew $bk_exp_width \
		--gradient-ew $fw_exp_width \
		--activation-ew $bk_exp_width \
		--loss-scale $loss_scale \
		--log-path loss_scale_finegrain_log \
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
done
	wait
done
done
done
done
done 

