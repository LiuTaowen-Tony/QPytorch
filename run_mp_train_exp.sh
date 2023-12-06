#!/bin/bash
round=stochastic
source .venv/bin/activate


for man_width in 3; do
for round2 in stochastic nearest ; do
for loss_scale in 256 1024 2048 4096; do
for batch_size in 64; do
for lr in 0.1; do
for bk_exp_width in 3; do
for fw_exp_width in 3; do
for run in 0 1 2; do
	# device=0
	# if [ $run -lt 2 ] 
	# then
	# 	device=1
	# fi
	CUDA_VISIBLE_DEVICES=$run python mix_precision_train.py \
		-w $man_width -e $man_width -g $man_width -a $man_width \
		--weight-ew $fw_exp_width \
		--error-ew $bk_exp_width \
		--gradient-ew $fw_exp_width \
		--activation-ew $bk_exp_width \
		--loss-scale $loss_scale \
		--log-path e3m3_momentum_log \
		--epochs 200 \
		--weight-round $round2 \
		--error-round $round2 \
		--gradient-round $round2 \
		--activation-round $round2 \
		--learning-rate $lr \
		--momentum 0.9 \
		--batch-size $batch_size \
		--mix-precision True &
done
	wait
done
done
done
done
done
done
done 

