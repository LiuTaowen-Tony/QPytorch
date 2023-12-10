#!/bin/bash
round=stochastic
source .venv/bin/activate


for man_width in 3; do
for loss_scale in 250 500 1000 1500 2000 3000 4000 5000 6000 7000 8000; do
for clip in 10 5 1 0.5 0.1 ; do
for batch_size in 64; do
for lr in 0.1; do
for bk_exp_width in 3; do
for fw_exp_width in 3; do
for run in 0 1 2; do
for round2 in stochastic nearest ; do
	CUDA_VISIBLE_DEVICES=$run python mix_precision_train.py \
		-w $man_width -e $man_width -g $man_width -a $man_width \
		--seed $run \
		--weight-ew $fw_exp_width \
		--error-ew $bk_exp_width \
		--gradient-ew $fw_exp_width \
		--activation-ew $bk_exp_width \
		--loss-scale $loss_scale \
		--log-path e4m3_log_${run}_clip \
		--seed $run \
		--epochs 200 \
		--weight-round $round2 \
		--error-round $round2 \
		--gradient-round $round2 \
		--activation-round $round2 \
		--learning-rate $lr \
		--momentum 0 \
		--batch-size $batch_size \
		--clip $clip \
		--mix-precision True &
done
done
	wait
done
done
done
done
done
done 
done
