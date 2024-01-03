#!/bin/bash
round=stochastic
source .venv/bin/activate

clip=100000000000000


for man_width in 3; do
for loss_scale in 1 10 25 100 200 1000 2000 3000 5000 8000 16000 32000 64000 128000 256000 512000 1024000; do
for batch_size in 64; do
for lr in 0.1; do
for bk_exp_width in 4; do
for fw_exp_width in 2; do
for run in 1 2; do
for round2 in stochastic nearest ; do
	CUDA_VISIBLE_DEVICES=$run python mix_precision_train.py \
		-w $man_width -e $man_width -g $man_width -a $man_width \
		--seed $run \
		--weight-ew $fw_exp_width \
		--error-ew $bk_exp_width \
		--gradient-ew $fw_exp_width \
		--activation-ew $bk_exp_width \
		--loss-scale $loss_scale \
		--log-path results_new/f${fw_exp_width}_b${bk_exp_width}_log_${run}_clip \
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
